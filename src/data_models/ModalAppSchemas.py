from pydantic import BaseModel, Field, field_validator
from fastapi import Query
from typing import  Optional, Any
from src.utils.Constants import (
    MIN_HEIGHT, MAX_HEIGHT, 
    MAX_INFERENCE_STEPS, 
    MIN_INFERENCE_STEPS, 
    MAX_BATCH, MIN_BATCH, 
    MAX_GUIDANCE_SCALE, 
    MIN_GUIDANCE_SCALE, 
    MAX_STRENGTH,
    MIN_STRENGTH,
    MAX_COLOR, MIN_COLOR,
    MAX_FPS, MIN_FPS)
from src.utils.Constants import sdxl_model_string, controlnet_models


class StubNames(BaseModel):
    sdxl_text_to_image: str = "SDXL_Text_To_Image"
    sdxl_image_to_image : str = "SDXL_Image_To_Image"
    sdxl_controlnet : str = "SDXL_controlnet"
    ultrasharp_upscaler : str = "Ultrasharp_Upscaler"
    image_variation : str = "IPAdapter_image_variation"
    face_consistent : str = "IPAdapter_face_consistent"
    background_removal : str = "BackGround_Removal"
    stable_video_diffusion : str = "Stable_Video_Diffusion"
    illusion_diffusion : str = "Illusion_Diffusion"

class StubConfiguration(BaseModel):
    memory : int
    container_idle_timeout : int 
    gpu : str 

class TimeData(BaseModel):
    startup_time : int | float
    runtime : int | float

class TaskResponse(BaseModel):
    result : list[str | dict] 
    Has_NSFW_Content : list[bool]
    time : TimeData


# class SDXLParameters(BaseModel):
#     model : str = Field(pattern = sdxl_model_string)  
#     lora_model : Optional[str] = None
#     image : Optional[str] = None
#     controlnet_models : Optional[list[str]] = None
#     height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
#     width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
#     num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 
#     guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
#     batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
#     prompt: str
#     negative_prompt: Optional[str] = " "
#     lora_scale : float = Query(default = 0.5, gt = 0, le = 1)
    


class SDXLText2ImageParameters(BaseModel):
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 
    guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale : float = Query(default = 0.5, gt = 0, le = 1)

class SDXLImage2ImageParameters(BaseModel):
    image : str | Any
    strength : float = Query(gt = MIN_STRENGTH, le = MAX_STRENGTH)
    guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale : float = Query(default = 0.5, gt = 0, le = 1)

class SDXLControlNetParameters(SDXLImage2ImageParameters):
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS)

class UpscaleParameters(BaseModel):
    image : str | Any

class VariationParameters(SDXLControlNetParameters):
    prompt: Optional[str] = None

class FaceConsistentParameters(SDXLText2ImageParameters):
    image : str | Any
    strength : float = Query(gt = MIN_STRENGTH, le = MAX_STRENGTH) 

class InitParameters(BaseModel):
    model : str = Field(pattern = sdxl_model_string)  
    lora_model : Optional[str] = None
    controlnet_model : Optional[str] = Field(default=None, pattern = controlnet_models)

class BackGroundRemoval(BaseModel):
    image : str | Any
    bg_img : Optional[str] = None
    bg_color : Optional[bool] = False 
    r_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
    g_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
    b_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
    blur: Optional[bool] = False  

class StableVideoDiffusion(BaseModel):
    image : str | Any
    fps : int = Query(default=MIN_FPS,ge=MIN_FPS, le=MAX_FPS)

class IllusionDuiffusion(SDXLImage2ImageParameters):
    controlnet_scale : float = Query(ge=0, le=4)
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 



    