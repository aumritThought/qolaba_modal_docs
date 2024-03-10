from src.data_models.ModalAppSchemas import TaskResponse, TimeData
from modal import Image as MIM
import cloudinary
import cloudinary.uploader
from PIL import Image, ImageOps
from PIL.Image import Image as Imagetype
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import torch, time, os, requests, re, io
from src.utils.Constants import BASE_IMAGE_COMMANDS, PYTHON_VERSION, REQUIREMENT_FILE_PATH, MAX_HEIGHT, MIN_HEIGHT, SDXL_REFINER_MODEL_PATH
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from requests import Response


class SafetyChecker:
    def __init__(self) -> None:
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor()

    def check_nsfw_content(self, image : Imagetype) -> list[bool]:
        
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")

        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_checker_input.pixel_values
        )

        return has_nsfw_concept

def generate_image_urls(image_data, safety_checker : SafetyChecker, format : str = "JPEG") -> tuple[list[str], list[bool]]:
    image_urls = []
    has_nsfw_content = []
    for im in range(0, len(image_data)):
        nsfw_content = safety_checker.check_nsfw_content(image_data[im])
        if nsfw_content[0]:
            has_nsfw_content.append(nsfw_content[0])
        else:
            im_url = upload_cloudinary_image(image_data[im], format)
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


def upload_cloudinary_image(image : Imagetype | str, format : str = "JPEG") -> str:
    cloudinary.config(
        cloud_name=os.environ["CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

    try:
        if(type(image) == Imagetype):
            with io.BytesIO() as buffer:
                image.save(buffer, format=format)
                image_bytes = buffer.getvalue()
        else:
            image_bytes = io.BytesIO(image)

        response = cloudinary.uploader.upload(
            file=image_bytes,
        )
        return response["secure_url"]

    except Exception as e:
        raise Exception(f"Error uploading image to Cloudinary: {e}")

def upload_cloudinary_video(video_path : str) -> str:
    cloudinary.config(
        cloud_name=os.environ["CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

    try:
        response = cloudinary.uploader.upload(
                file=video_path,  resource_type="video"
            )
        return response["secure_url"]

    except Exception as e:
        raise Exception(f"Error uploading image to Cloudinary: {e}")


def resize_image(img: Imagetype) -> Imagetype:
    img = img.resize((64 * round(img.size[0] / 64), 64 * round(img.size[1] / 64)))

    if ( img.size[0] > MAX_HEIGHT or img.size[0] < MIN_HEIGHT or img.size[1] > MAX_HEIGHT or img.size[1] < MIN_HEIGHT):

        if img.size[1] >= img.size[0]:
            height = MAX_HEIGHT
            width = ((int(img.size[0]* MAX_HEIGHT/ img.size[1]))// 64) * 64
        else:
            width = MAX_HEIGHT
            height = ((int(img.size[1]*MAX_HEIGHT/ img.size[0]))// 64) * 64

        img = img.resize((width, height))
    return img

def make_request(url: str, method: str, json_data: dict = None, headers: dict = None) -> Response:

    method = method.upper()

    if method not in ["GET", "POST"]:
        raise Exception(
            "Invalid request method", "Please check method input given with URL"
        )

    response = None
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, json=json_data, headers=headers)

    response.raise_for_status()

    return response

def get_image_from_url(url: str, resize : bool) -> Imagetype:

    response : Response = make_request(url) 
    image_data = io.BytesIO(response.content)
    image = Image.open(image_data).convert("RGB")
    if(resize == True):
        image = resize_image(image)
    return image

def invert_bw_image_color(img: Imagetype) -> Imagetype:
    white_background = Image.new("RGB", img.size, (255, 255, 255))

    white_background.paste(img, mask=img.split()[3])

    img = ImageOps.invert(white_background)

    return img


def get_seed_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def download_safety_checker():
    SafetyChecker()

def get_base_image() -> MIM:
    return MIM.debian_slim(python_version = PYTHON_VERSION).run_commands(BASE_IMAGE_COMMANDS).pip_install_from_requirements(REQUIREMENT_FILE_PATH).run_function(download_safety_checker, gpu = "t4")

    
def get_refiner(pipe : StableDiffusionXLPipeline) -> DiffusionPipeline:
    return DiffusionPipeline.from_pretrained(
            SDXL_REFINER_MODEL_PATH,
            text_encoder_2 = pipe.text_encoder_2,
            vae = pipe.vae,
            torch_dtype = torch.float16,
            use_safetensors = True,
            variant = "fp16",
        ).to("cuda")
    

def timing_decorator(func: callable) -> callable:
    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)

        execution_time = time.time() - start_time

        time_data = {"runtime": execution_time, "startup_time": 0}
        result["time"] = time_data

        return result

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


