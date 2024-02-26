import time
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from modal import method, Secret
from src.data_models.ImageToImage import CannyModels, ImageInferenceInput
from src.utils.Globals import (
    image_to_image_inference,
    generate_image_urls,
    create_stub,
    get_image_array_from_url,
)
from src.utils.Strings import STARLIGHT_SAFETENSORS
import torch


canny_models = CannyModels()


# Function to download and prepare Canny models
def download_canny_models() -> None:
    controlnet = ControlNetModel.from_pretrained(
        canny_models.controlnet_model, torch_dtype=torch.float16
    )
    StableDiffusionXLControlNetPipeline.from_single_file(
        STARLIGHT_SAFETENSORS,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    StableDiffusionSafetyChecker.from_pretrained(canny_models.sd_safety_checker)
    CLIPFeatureExtractor()


# Create a stub for the service
stub = create_stub(
    "canny_controlnet_image2image",
    [
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "wget https://civitai.com/api/download/models/182077",
        "pip install diffusers --upgrade",
        "pip install controlnet_aux invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf pydantic cloudinary",
        "mv 182077 Starlight.safetensors",
    ],
    download_canny_models,
)


# Function to create the controlnet pipeline
def create_controlnet_pipe():
    controlnet = ControlNetModel.from_pretrained(
        canny_models.controlnet_model, torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        STARLIGHT_SAFETENSORS,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


# Define the StableDiffusion class with modal methods
@stub.cls(
    gpu="a10g",
    container_idle_timeout=200,
    memory=10240,
    secrets=[Secret.from_name("clodinary-secrets")],
)
class StableDiffusion:
    def __enter__(self):
        self.start_time = time.time()
        self.pipe = create_controlnet_pipe()
        self.container_execution_time = time.time() - self.start_time

    @method()
    def run_inference(self, inference_input: dict) -> dict:
        import cv2

        # from cv2 import Canny
        from PIL import Image
        import numpy as np

        inference_input = ImageInferenceInput(**inference_input)
        start_time = time.time()
        image = get_image_array_from_url(inference_input.image_url)
        generator = torch.Generator()
        generator.manual_seed(inference_input.seed)
        # Apply Canny edge detection
        edges = cv2.Canny(
            image, inference_input.low_threshold, inference_input.high_threshold
        )
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        image = Image.fromarray(edges)

        images = []
        for _ in range(inference_input.batch):
            result = self.pipe(
                prompt=inference_input.prompt,
                image=image,
                num_inference_steps=inference_input.num_inference_steps,
                guidance_scale=inference_input.guidance_scale,
                negative_prompt=inference_input.negative_prompt,
                controlnet_conditioning_scale=inference_input.controlnet_conditioning_scale,
                generator=generator,
            ).images
            images.append(result[0])

        torch.cuda.empty_cache()

        image_data = {
            "images": images,
            "has_nsfw_content": [False] * inference_input.batch,
        }
        image_urls = generate_image_urls(image_data)
        self.runtime = time.time() - start_time

        return image_to_image_inference(
            image_urls,
            [False] * inference_input.batch,
            self.container_execution_time,
            self.runtime,
        )
