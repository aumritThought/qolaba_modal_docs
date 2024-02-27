from src.data_models.ImageToImage import Inference, ImageData, ImageInferenceInput
from src.utils.Strings import QOLABA_SERVER_UPLOAD_CLOUDINARY_URL
from modal import Image, Stub
import cloudinary
import cloudinary.uploader
from io import BytesIO
import os
import numpy as np
from io import BytesIO
import requests
from PIL import Image as pim
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch


def generate_image_urls(image_data: ImageData) -> list:
    image_urls = []
    for im in range(0, len(image_data.images)):
        if image_data.has_nsfw_content[im]:
            pass
        else:
            im_url = upload_cloudinary_image(image_data.images[im])
            image_urls.append(im_url)
    return image_urls


def image_to_image_inference(images: list, nsfw, time, runtime):
    image_data = ImageData(images=images, has_nsfw_content=nsfw)
    inference = Inference()
    inference.result = generate_image_urls(image_data)
    inference.has_nsfw_content = nsfw
    inference.startup_time = time
    inference.runtime = runtime
    return dict(inference)


def upload_cloudinary_image(image):
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
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
        print(f"Error uploading image to Cloudinary: {e}")


def create_stub(name: str, run_commands: list, run_function) -> Stub:
    stub = Stub(name)
    image = (
        Image.debian_slim(python_version="3.11").run_commands(run_commands)
    ).run_function(run_function, gpu="a10g")

    stub.image = image
    return stub


def get_image_array_from_url(url: str):

    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    image_data = BytesIO(response.content)
    image = pim.open(image_data)
    return np.array(image.convert("RGB"))


def get_images_from_controlnet_pipe(
    inference_input: ImageInferenceInput,
    pipe,
    image: np.array,
    generator: torch.Generator,
) -> list:
    images = []
    for _ in range(inference_input.batch):
        result = pipe(
            prompt=inference_input.prompt,
            image=image,
            num_inference_steps=inference_input.num_inference_steps,
            guidance_scale=inference_input.guidance_scale,
            negative_prompt=inference_input.negative_prompt,
            controlnet_conditioning_scale=inference_input.controlnet_conditioning_scale,
            generator=generator,
        ).images
        images.append(result[0])

    return images


def get_seed_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
