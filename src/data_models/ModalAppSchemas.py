from pydantic import BaseModel, constr
from fastapi import Query
from typing import  Optional
from src.utils.Constants import (
    MIN_HEIGHT, MAX_HEIGHT, 
    MAX_INFERENCE_STEPS, 
    MIN_INFERENCE_STEPS, 
    MAX_BATCH, MIN_BATCH, 
    MAX_GUIDANCE_SCALE, 
    MIN_GUIDANCE_SCALE)
from src.utils.Constants import sdxl_model_string


class StubNames(BaseModel):
    sdxl_text_to_image: str = "SDXL_Text_To_Image"
    

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


class SDXLText2ImageParameters(BaseModel):
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 
    guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale : float = Query(default = 0.5, gt = 0, le = 1)

class InitParameters(BaseModel):
    model_name : sdxl_model_string  # type: ignore
    lora_model : Optional[str] = None