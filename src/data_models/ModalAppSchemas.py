from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import Query
from typing import  Optional, Any, List
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
    MAX_FPS, MIN_FPS,
    MIN_INCREASE_SIDE, MAX_INCREASE_SIDE,
    MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS, MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS,
    elevenlabs_accent_list, elevenlabs_age_list, elevenlabs_gender_list, dalle_supported_quality, sdxl_preset_list)
from src.utils.Constants import sdxl_model_string, controlnet_models
from elevenlabs import voices, Voice, set_api_key
import os


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
    qr_code_generation : str = "QRCode_Generation"

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

class ClipDropUncropParameters(BaseModel):
    image : str | Any
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    right: int = Query(ge = MIN_INCREASE_SIDE, le = MAX_INCREASE_SIDE)
    left: int = Query(ge = MIN_INCREASE_SIDE, le = MAX_INCREASE_SIDE)
    top: int = Query(ge = MIN_INCREASE_SIDE, le = MAX_INCREASE_SIDE)
    bottom: int = Query(ge = MIN_INCREASE_SIDE, le = MAX_INCREASE_SIDE)

class ClipDropCleanUpParameters(BaseModel):
    image : str | Any
    mask_image : str | Any

class ClipDropReplaceBackgroundParameters(BaseModel):
    image : str | Any
    prompt : str

class ClipDropRemoveTextParameters(BaseModel):
    image : str | Any

class DIDVideoParameters(BaseModel):
    image: str | Any
    expression : Optional[did_expression_list] = "neutral" # type: ignore
    expression_intesity : float = Query(default=1, ge=0, le=1)
    voice_id: Optional[str] = "d7bbcdd6964c47bdaae26decade4a933"
    prompt: Optional[str] = "Hey, welcome to Qolaba"

class ElevenLabsParameters(BaseModel):
    prompt: str = Query(default="Hi, we are in Qolaba", max_length=2500, min_length=1) 
    clone: Optional[bool]= False
    name: Optional[str] = "Cloned Voice"
    description : Optional[str] = "description"
    list_of_files: List[str] = Query(default=["None"], max_length = MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS, min_length = MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS)
    voice_design: Optional[bool] = False 
    gender: Optional[elevenlabs_gender_list]="female" # type: ignore
    age: Optional[elevenlabs_age_list]="young" # type: ignore
    accent: Optional[elevenlabs_accent_list]="american" # type: ignore
    accent_strength: float = Query(default = 1, ge = 0.3, le = 2)
    generate_audio : Optional[bool]= True
    voice_id : Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    stability: float = Query(default=0.5,ge=0, le=1)
    similarity_boost: float = Query(default=0.75,ge=0, le=1)
    style: float = Query(default=0.0,ge=0, le=1)
    use_speaker_boost : Optional[bool] = True 

    
    @model_validator(mode='after')
    def validate_params(self):
        total_sum=sum([self.clone, self.voice_design, self.generate_audio])
        if(not(total_sum==1)):
            raise ValueError("Only one of 'clone', 'voicedesign','list_of_voices', or 'generate_audio' must be True")
        return self
    
    @field_validator("voice_id")
    def validate_voice_id(cls, v):
        set_api_key(os.environ["ELEVENLABS_API_KEY"])
        voices_data : List[Voice] = voices()
        voice_dict=[]
        for i in voices_data:
            voice_dict.append(i.voice_id)
        if v not in voice_dict:
            raise ValueError("Invalid input. The parameter must be one of: " + ", ".join(voice_dict))
        return v
    
class DalleParameters(SDXLText2ImageParameters):
    quality : Optional[dalle_supported_quality] = "hd"  # type: ignore

class SDXLAPITextToImageParameters(SDXLText2ImageParameters):
    style_preset = Field(pattern=sdxl_preset_list)

class SDXLAPIImageToImageParameters(SDXLImage2ImageParameters):
    style_preset = Field(pattern=sdxl_preset_list)
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 

class APITaskResponse(BaseModel):
    result : list[str | dict] 
    Has_NSFW_Content : list[bool]
    time : TimeData

class APIInput(BaseModel):
    parameters : dict
    init_parameters : dict
    ref_id: Optional[str] = ""
    celery: Optional[bool] = False 

