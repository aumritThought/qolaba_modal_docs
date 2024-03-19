from src.data_models.ModalAppSchemas import TaskResponse, TimeData
from modal import Image as MIM
from modal import Secret
from PIL import Image
from PIL.Image import Image as Imagetype
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import torch, time, os, requests, re, io, datetime, uuid
from src.utils.Constants import BASE_IMAGE_COMMANDS, PYTHON_VERSION, REQUIREMENT_FILE_PATH, MEAN_HEIGHT, SDXL_REFINER_MODEL_PATH, google_credentials_info, OUTPUT_IMAGE_EXTENSION, SECRET_NAME, content_type, MAX_UPLOAD_RETRY
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from requests import Response
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
#Safety Checker Utils

class SafetyChecker:
    def __init__(self) -> None:
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor()

    def check_nsfw_content(self, image : Imagetype) -> list[bool]:
        image = image.convert("RGB")
        safety_checker_input = self.feature_extractor(np.array(image), return_tensors="pt").to("cuda")

        image, has_nsfw_concept = self.safety_checker(
            images= np.array(image), clip_input=safety_checker_input.pixel_values
        )

        return has_nsfw_concept
    
def download_safety_checker():
    SafetyChecker()

#Modal image Utils
def get_base_image() -> MIM:
    return MIM.debian_slim(python_version = PYTHON_VERSION).run_commands(BASE_IMAGE_COMMANDS).pip_install_from_requirements(REQUIREMENT_FILE_PATH).run_function(download_safety_checker, gpu = "t4", secrets = [Secret.from_name(SECRET_NAME)])

#Modal app utils
def get_refiner(pipe : StableDiffusionXLPipeline) -> DiffusionPipeline:
    return DiffusionPipeline.from_pretrained(
            SDXL_REFINER_MODEL_PATH,
            text_encoder_2 = pipe.text_encoder_2,
            vae = pipe.vae,
            torch_dtype = torch.float16,
            use_safetensors = True,
            variant = "fp16",
        ).to("cuda")


#Modal App Output relataed utils
def generate_image_urls(image_data, safety_checker : SafetyChecker) -> tuple[list[str], list[bool]]:
    image_urls = []
    has_nsfw_content = []
    for im in range(0, len(image_data)):
        nsfw_content = safety_checker.check_nsfw_content(image_data[im])
        if nsfw_content[0]:
            has_nsfw_content.append(nsfw_content[0])
        else:
            im_url = upload_data_gcp(image_data[im], OUTPUT_IMAGE_EXTENSION)
            image_urls.append(im_url)
            has_nsfw_content.append(nsfw_content[0])
    return image_urls, has_nsfw_content


def prepare_response(result: list[str] | dict, Has_NSFW_content : list[bool], time : float, runtime : float) -> dict:
    task_response = TaskResponse(
        result = result,
        Has_NSFW_Content = Has_NSFW_content,
        time = TimeData(startup_time = time, runtime = runtime)
    )
    return task_response.model_dump()

#Completing request 
def make_request(url: str, method: str, json_data: dict = None, headers: dict = None, files : dict = None, json : dict = None) -> Response:

    method = method.upper()

    if method not in ["GET", "POST"]:
        raise Exception(
            "Invalid request method, Please check method input given with URL"
        )

    response = None
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, data=json_data, headers=headers, files = files, json = json)

    if(response.status_code != 200):
        raise Exception(str(response.text))

    return response

#Cloudinary
def upload_data_gcp(data : Imagetype | str, extension : str) -> str:
    for i in range(0, MAX_UPLOAD_RETRY):
        url = upload_to_gcp(data, extension)
        if(url != None and url != ""):
            break
    if(url == "" or url == None):
        raise Exception("Received an empty URL")
    return url
            
def upload_to_gcp(data : Imagetype | str, extension : str) -> str:
    try:
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        random_string = str(uuid.uuid4())

        destination_blob_name = f"{current_time}_{random_string}.{extension}"
        if(type(data) == Imagetype):
            with io.BytesIO() as buffer:
                data.save(buffer, format="PNG")
                data = buffer.getvalue()

        byte_data = io.BytesIO(data)

        credentials = service_account.Credentials.from_service_account_info(google_credentials_info)

        storage_client = storage.Client(credentials=credentials, project=google_credentials_info['project_id'])

        bucket = storage_client.bucket(os.environ["BUCKET_NAME"])

        blob = bucket.blob(destination_blob_name)

        blob.content_type = content_type[extension]

        blob.content_disposition = "inline"

        blob.upload_from_file(byte_data)

        return blob.public_url

    except Exception as e:
        raise Exception(f"Error uploading to GCP: {e}")


#Image operations
def resize_image(img: Imagetype) -> Imagetype:
    img = img.resize((64 * round(img.size[0] / 64), 64 * round(img.size[1] / 64)))
    if(img.size[0]*img.size[1] > MEAN_HEIGHT*MEAN_HEIGHT):
        # if ( img.size[0] > MAX_HEIGHT or img.size[0] < MIN_HEIGHT or img.size[1] > MAX_HEIGHT or img.size[1] < MIN_HEIGHT):
            if img.size[1] >= img.size[0]:
                height = MEAN_HEIGHT
                width = ((int(img.size[0]* MEAN_HEIGHT/ img.size[1]))// 64) * 64
            else:
                width = MEAN_HEIGHT
                height = ((int(img.size[1]*MEAN_HEIGHT/ img.size[0]))// 64) * 64

            img = img.resize((width, height))
    return img



def get_image_from_url(url: str) -> Imagetype:

    response : Response = make_request(url, method = "GET") 
    image_data = io.BytesIO(response.content)
    image = Image.open(image_data).convert("RGB")
    image = resize_image(image)
    return image

def invert_bw_image_color(img: Imagetype) -> Imagetype:
    mask_array = np.array(img)

    inverted_mask_array = 255 - mask_array

    inverted_mask = Image.fromarray(inverted_mask_array)
    
    return inverted_mask


def get_seed_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator




    

    
#API app utils
def timing_decorator(func: callable) -> callable:
    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)

        runtime = time.time() - start_time

        result = TaskResponse(**result)

        result.time.runtime = runtime

        return result.model_dump()

    return wrapper


def check_token(api_key: HTTPAuthorizationCredentials):
    if api_key.credentials != os.environ["API_KEY"]:

        response = requests.post(
            url=os.environ["QOLABA_B2B_API_URL"],
            data={"keysHashedValue": api_key.credentials},
        )

        response.raise_for_status()

        data = response.json()
        if data["response"] == False:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
            )

def get_clean_name(name : str) -> str:
    pattern = re.compile('[^a-zA-Z0-9]')
    cleaned_string = pattern.sub('', name)
    return cleaned_string.lower()


