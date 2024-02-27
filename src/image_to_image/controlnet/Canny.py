import time
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from modal import method, Secret
from src.data_models.ImageToImage import CannyModels, ImageInferenceInput, ImageData
from src.utils.Globals import (
    image_to_image_inference,
    create_stub,
    get_image_array_from_url,
    get_images_from_controlnet_pipe,
    get_seed_generator,
)
from src.utils.Strings import (
    CLOUDINARY_SECRET,
    CANNY_CONTROLNET_IMAGETOIMAGE,
    STARLIGHT_SAFETENSORS_PATH,
)
import torch
import cv2
from PIL import Image
import numpy as np
from src.utils.Constants import CONTROLNET_COMMANDS


models = CannyModels()


# Function to download and prepare Canny models
def download_models() -> None:
    controlnet = ControlNetModel.from_pretrained(
        models.controlnet_model, torch_dtype=torch.float16
    )
    StableDiffusionXLControlNetPipeline.from_single_file(
        STARLIGHT_SAFETENSORS_PATH,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    StableDiffusionSafetyChecker.from_pretrained(models.sd_safety_checker)
    CLIPFeatureExtractor()


# Create a stub for the service
stub = create_stub(
    CANNY_CONTROLNET_IMAGETOIMAGE,
    CONTROLNET_COMMANDS,
    download_models,
)


# Function to create the controlnet pipeline
def create_controlnet_pipe():
    controlnet = ControlNetModel.from_pretrained(
        models.controlnet_model, torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        STARLIGHT_SAFETENSORS_PATH,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def get_canny_edge_image(inference_input: ImageInferenceInput):
    image = get_image_array_from_url(inference_input.image_url)
    edges = cv2.Canny(
        image, inference_input.low_threshold, inference_input.high_threshold
    )
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    image = Image.fromarray(edges)
    return image


# Define the StableDiffusion class with modal methods
@stub.cls(
    gpu="a10g",
    container_idle_timeout=200,
    memory=10240,
    secrets=[Secret.from_name(CLOUDINARY_SECRET)],
)
class StableDiffusion:
    def __enter__(self):
        self.start_time = time.time()
        self.pipe = create_controlnet_pipe()
        self.container_execution_time = time.time() - self.start_time

    @method()
    def run_inference(self, inference_input: dict) -> dict:

        inference_input = ImageInferenceInput(**inference_input)
        start_time = time.time()
        generator = get_seed_generator(inference_input.seed)
        image = get_canny_edge_image(inference_input)
        images = get_images_from_controlnet_pipe(
            inference_input, self.pipe, image, generator
        )

        torch.cuda.empty_cache()

        return image_to_image_inference(
            images,
            [False] * inference_input.batch,
            self.container_execution_time,
            time.time() - start_time,
        )
