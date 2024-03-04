from src.data_models.ModalAppSchemas import TaskResponse, TimeData
from modal import Image
import cloudinary
import cloudinary.uploader
from io import BytesIO
import os, PIL
import requests
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import torch
from src.utils.Constants import BASE_IMAGE_COMMANDS, PYTHON_VERSION, REQUIREMENT_FILE_PATH, MAX_HEIGHT, MIN_HEIGHT, SDXL_REFINER_MODEL_PATH

class SafetyChecker:
    def __init__(self) -> None:
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor()

    def check_nsfw_content(self, image : PIL.Image) -> list[bool]:
        
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")

        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_checker_input.pixel_values
        )

        return has_nsfw_concept

def generate_image_urls(image_data: list[PIL.Image], safety_checker : SafetyChecker) -> tuple[list[str], list[bool]]:
    image_urls = []
    has_nsfw_content = []
    for im in range(0, len(image_data)):
        nsfw_content = safety_checker.check_nsfw_content(image_data[im])
        if nsfw_content[0]:
            has_nsfw_content.append(nsfw_content[0])
        else:
            im_url = upload_cloudinary_image(image_data[im])
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


def upload_cloudinary_image(image : PIL.Image) -> str:
    cloudinary.config(
        cloud_name=os.environ["CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

    try:
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

        response = cloudinary.uploader.upload(
            file=image_bytes,
        )
        return response["secure_url"]

    except Exception as e:
        raise Exception(f"Error uploading image to Cloudinary: {e}")


def resize_image(img: Image) -> PIL.Image:
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


def get_image_from_url(url: str, resize : bool) -> PIL.Image:

    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    image_data = BytesIO(response.content)
    image = PIL.Image.open(image_data)
    if(resize == True):
        image = resize_image(image)
    return image



def get_seed_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def download_safety_checker():
    SafetyChecker()

def get_base_image() -> Image:
    return Image.debian_slim(python_version = PYTHON_VERSION).run_commands(BASE_IMAGE_COMMANDS).pip_install_from_requirements(REQUIREMENT_FILE_PATH).run_function(download_safety_checker, gpu = "t4")



    
def get_refiner(pipe : StableDiffusionXLPipeline):
    return DiffusionPipeline.from_pretrained(
            SDXL_REFINER_MODEL_PATH,
            text_encoder_2 = pipe.text_encoder_2,
            vae = pipe.vae,
            torch_dtype = torch.float16,
            use_safetensors = True,
            variant = "fp16",
        ).to("cuda")
    
