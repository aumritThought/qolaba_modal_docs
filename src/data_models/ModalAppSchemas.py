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
    MIN_STRENGTH)
from src.utils.Constants import sdxl_model_string, controlnet_models


class StubNames(BaseModel):
    sdxl_text_to_image: str = "SDXL_Text_To_Image"
    sdxl_image_to_image : str = "SDXL_Image_To_Image"
    sdxl_controlnet : str = "SDXL_controlnet"
    ultrasharp_upscaler : str = "Ultrasharp_Upscaler"

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
    strength : float = Query(ge = MIN_STRENGTH, le = MAX_STRENGTH)
    guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale : float = Query(default = 0.5, gt = 0, le = 1)

class SDXLControlNetParameters(BaseModel):
    image : str | Any
    strength : float = Query(ge = MIN_STRENGTH, le = MAX_STRENGTH)
    guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale : float = Query(default = 0.5, gt = 0, le = 1)
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS)

class UpscaleParameters(BaseModel):
    image : str | Any

class InitParameters(BaseModel):
    model : str = Field(pattern = sdxl_model_string)  
    lora_model : Optional[str] = None
    controlnet_model : str = Field(pattern = controlnet_models)





    