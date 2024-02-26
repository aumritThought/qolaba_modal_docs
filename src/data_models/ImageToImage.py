from pydantic import BaseModel, Field, validator, HttpUrl
from uuid import uuid4, UUID
from PIL import Image
from typing import Any, Iterator


class Time(BaseModel):
    startup_time: str = None
    runtime: str = None


class Inference(BaseModel):
    id: UUID = Field(default=uuid4())
    result: list = None
    has_nsfw_content: list[bool] = None
    startup_time: str = None
    runtime: str = None


class CannyModels(BaseModel):
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    sd_safety_checker: str = "CompVis/stable-diffusion-safety-checker"
    sdxl_vae_autoencoder: str = "madebyollin/sdxl-vae-fp16-fix"


class ImageInferenceInput(BaseModel):
    image_url: str = None
    prompt: str = ""
    batch: int = 1
    seed: int = 0
    low_threshold: int = Field(default=100, ge=0, le=255)
    high_threshold: int = Field(default=200, ge=0, le=255)
    num_inference_steps: int = Field(default=20, gt=0)
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    controlnet_conditioning_scale: float = Field(default=0.5, gt=0)
