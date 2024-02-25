from modal import Image, Stub, method
from src.data_models.ImageToImage import Inference, Models
from src.utils.Globals import image_to_image_inference, generate_image_urls, create_stub
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from src.utils.Strings import STARLIGHT_SAFETENSORS
import time

canny_models = Models()


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


stub = create_stub(
    "Canny" + "_controlnet_" + "_image2image",
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

@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:
    def __enter__(self):

        st = time.time()
        self.pipe = create_controlnet_pipe()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(
        self, file_url, prompt, guidance_scale, negative_prompt, batch, strength=None
    ) -> Inference:
        import cv2, time, torch
        from PIL import Image
        import numpy as np

        st = time.time()

        image = np.array(file_url)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        images = []
        for i in range(0, batch):
            img = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=20,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=0.5,
            ).images
            images.append(img[0])

        torch.cuda.empty_cache()

        image_data = {"images": images, "Has_NSFW_Content": [False] * batch}

        image_urls = generate_image_urls(image_data)
        self.runtime = time.time() - st

        return image_to_image_inference(
            image_urls, [False] * batch, self.container_execution_time, self.runtime
        )
