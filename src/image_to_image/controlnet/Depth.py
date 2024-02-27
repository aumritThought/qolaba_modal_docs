import time
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from modal import method, Secret, Image
from src.data_models.ImageToImage import DepthModels, ImageInferenceInput, ImageData
from src.utils.Globals import (
    image_to_image_inference,
    create_stub,
    get_image_array_from_url,
    get_seed_generator,
    get_images_from_controlnet_pipe,
)
from src.utils.Strings import (
    STARLIGHT_SAFETENSORS_PATH,
    CLOUDINARY_SECRET,
    DEPTH_CONTROLNET_IMAGETOIMAGE,
)
from src.utils.Constants import CONTROLNET_COMMANDS
import torch

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
import numpy as np
from PIL import Image as pim

models = DepthModels()


# Function to download and prepare Canny models
def download_models() -> None:
    DPTForDepthEstimation.from_pretrained(models.depth_estimator).to("cuda")
    DPTFeatureExtractor.from_pretrained(models.depth_estimator)
    controlnet = ControlNetModel.from_pretrained(
        models.controlnet_model,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    StableDiffusionXLControlNetPipeline.from_single_file(
        STARLIGHT_SAFETENSORS_PATH,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    StableDiffusionSafetyChecker.from_pretrained(models.sd_safety_checker)
    CLIPFeatureExtractor()


# Create a stub for the service
stub = create_stub(
    DEPTH_CONTROLNET_IMAGETOIMAGE,
    CONTROLNET_COMMANDS,
    download_models,
)


# Function to create the controlnet pipeline
def create_controlnet_pipe():
    controlnet = ControlNetModel.from_pretrained(
        models.controlnet_model,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        STARLIGHT_SAFETENSORS_PATH,
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
    secrets=[Secret.from_name(CLOUDINARY_SECRET)],
)
class StableDiffusion:
    def __enter__(self):
        self.start_time = time.time()
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            models.depth_estimator
        ).to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            models.depth_estimator
        )
        self.pipe = create_controlnet_pipe()
        self.container_execution_time = time.time() - self.start_time

    def get_depth_map_image(self, image):

        image = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = pim.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        image = image.resize((image.size))
        return image

    @method()
    def run_inference(self, inference_input: dict) -> dict:

        inference_input = ImageInferenceInput(**inference_input)
        start_time = time.time()
        generator = get_seed_generator(inference_input.seed)
        image = get_image_array_from_url(inference_input.image_url)
        image = self.get_depth_map_image(image)
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
