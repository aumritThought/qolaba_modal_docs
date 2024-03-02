import time
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from modal import method, Secret, Image
from src.data_models.ImageToImage import NormalModels, ImageInferenceInput, ImageData
from src.utils.Globals import (
    image_to_image_inference,
    create_stub,
    get_image_array_from_url,
    get_seed_generator,
    get_images_from_controlnet_pipe,
)
from src.utils.Strings import (
    SD_REVANIM,
    CLOUDINARY_SECRET,
    NORMAL_CONTROLNET_IMAGETOIMAGE,
)
from src.utils.Constants import NORMAL_CONTROLNET_COMMANDS
from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
import torch
from transformers import pipeline
import numpy as np
from PIL import Image as pim
import cv2

models = NormalModels()


# Function to download and prepare Canny models
def download_models() -> None:
    pipeline(models.depth_estimation_name, model=models.depth_estimator)
    controlnet = ControlNetModel.from_pretrained(
        models.controlnet_model,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        SD_REVANIM,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    StableDiffusionSafetyChecker.from_pretrained(models.sd_safety_checker)
    CLIPFeatureExtractor()


# Create a stub for the service
stub = create_stub(
    NORMAL_CONTROLNET_IMAGETOIMAGE,
    NORMAL_CONTROLNET_COMMANDS,
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
        SD_REVANIM,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config
        )

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
        self.depth_estimator = pipeline(
            models.depth_estimation_name, model=models.depth_estimator
        )
        # self.feature_extractor = DPTFeatureExtractor.from_pretrained(
        #     models.depth_estimator
        # )
        self.pipe = create_controlnet_pipe()
        self.container_execution_time = time.time() - self.start_time

    def get_depth_map_image(self, image):

        image = self.depth_estimator(image)["predicted_depth"][0]

        image = image.numpy()

        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.4

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize((image.size[0], image.size[1]))

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
