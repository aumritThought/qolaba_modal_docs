from pydantic import BaseModel
from uuid import uuid4


class Time(BaseModel):
    startup_time: str = None
    runtime: str = None


class Inference(BaseModel):
    # id: uuid4 = uuid4()
    result: list = None
    has_nsfw_content: list[bool] = None
    startup_time: str = None
    runtime: str = None


class Models(BaseModel):
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    sd_safety_checker: str = "CompVis/stable-diffusion-safety-checker"
    sdxl_vae_autoencoder: str = "madebyollin/sdxl-vae-fp16-fix"
