from pydantic import BaseModel, Field, field_validator, model_validator, constr
from fastapi import Query
from typing import  Optional, Any, List, Literal
from src.utils.Constants import (
    ELEVENLABS_ERROR, 
    VOICE_ID_ERROR_MSG,
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
    HW_MULTIPLE,
    MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS, MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS, gender_word,
    elevenlabs_accent_list, elevenlabs_age_list, elevenlabs_gender_list, dalle_supported_quality, sdxl_preset_list, did_expression_list)
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
    frnd_face_consistent : str = "IPAdapter_FRND_face_consistent"
    stable_cascade_text_to_image : str = "Stable_Cascade"
    oot_diffusion : str = "OOTDiffusion"

class StubConfiguration(BaseModel):
    memory : int
    container_idle_timeout : int 
    gpu : str 
    num_containers : int
    

class TimeData(BaseModel):
    startup_time : int | float
    runtime : int | float

class TaskResponse(BaseModel):
    result : Any
    Has_NSFW_Content : list[bool]
    low_res_urls : Optional[list[str]] = []
    time : TimeData
    extension : str | None


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

    @model_validator(mode='after')
    def validate_params(self):
        self.height = self.height - (self.height % HW_MULTIPLE)
        self.width = self.width - (self.width % HW_MULTIPLE)
        return self


class SDXLImage2ImageParameters(BaseModel):
    file_url : str | Any
    strength : float = Query(default = 0.7, gt = MIN_STRENGTH, le = MAX_STRENGTH)
    guidance_scale:  float = Query( ge = MIN_GUIDANCE_SCALE, le = MAX_GUIDANCE_SCALE)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale : float = Query(default = 0.5, gt = 0, le = 1)

class SDXLControlNetParameters(SDXLImage2ImageParameters):
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS)

class UpscaleParameters(BaseModel):
    file_url : str | Any
    scale : Literal[2, 4, 8]
    check_nsfw : Optional[bool] = True

class VariationParameters(SDXLControlNetParameters):
    prompt: Optional[str] = None

class FaceConsistentParameters(SDXLText2ImageParameters):
    file_url : str | Any
    strength : float = Query(gt = MIN_STRENGTH, le = MAX_STRENGTH) 

    @model_validator(mode='after')
    def validate_params(self):
        self.height = self.height - (self.height % HW_MULTIPLE)
        self.width = self.width - (self.width % HW_MULTIPLE)
        return self

class FRNDFaceAvatarParameters(BaseModel):
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=1024)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=1024)
    num_inference_steps: int = Query(default=30, ge=MIN_INFERENCE_STEPS, le=40)
    guidance_scale: float = Query(default=7.5, ge=MIN_GUIDANCE_SCALE, le=MAX_GUIDANCE_SCALE)
    batch: int = Query(default=1, ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str | None = f"((establishing shot)), (((smiling studio photo))), joyous indian dating aesthetic, 3D pixar, looking at camera, enjoyable pleasurable {gender_word}, ((black winter t-shirt)), proper eyes, Relaxed, Charming, Cordial,  Relaxed, Charming, Cordial, Gracious, 3d bitmoji avatar render, dark background"
    negative_prompt: str | None = " "
    file_url: str | Any
    strength: float = Query(default= 1, gt=MIN_STRENGTH, le=MAX_STRENGTH)
    gender : Literal["male", "female"]
    remove_background : bool
    bg_color : Optional[str] = None

    @model_validator(mode='after')
    def validate_params(self):
        self.height = self.height - (self.height % HW_MULTIPLE)
        self.width = self.width - (self.width % HW_MULTIPLE)
        return self


class InitParameters(BaseModel):
    model : str = Field(pattern = sdxl_model_string)  
    lora_model : Optional[str] = None
    controlnet_model : Optional[str] = Field(default=None, pattern = controlnet_models)

class BackGroundRemoval(BaseModel):
    file_url : str | Any
    bg_img : Optional[str] = None
    bg_color : Optional[bool] = False 
    r_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
    g_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
    b_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
    blur: Optional[bool] = False  

class StableVideoDiffusion(BaseModel):
    file_url : str | Any
    fps : int = Query(default=MIN_FPS,ge=MIN_FPS, le=MAX_FPS)

class IllusionDuiffusion(SDXLImage2ImageParameters):
    num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 

class DIDVideoParameters(BaseModel):
    file_url: str | Any
    expression : Optional[did_expression_list] = "neutral" # type: ignore
    expression_intesity : float = Query(default=1, ge=0, le=1)
    voice_id: Optional[str] = "d7bbcdd6964c47bdaae26decade4a933"
    prompt: Optional[str] = "Hey, welcome to Qolaba"

class CloneParameters(BaseModel):
    name: Optional[str] = "Cloned Voice"
    description : Optional[str] = "description"
    list_of_files: List[str] = Query(default=["None"], max_length = MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS, min_length = MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS)

class DesignParameters(BaseModel):
    name: Optional[str] = "Designed Voice"
    description : Optional[str] = "description"
    gender: Optional[elevenlabs_gender_list]="female" # type: ignore
    age: Optional[elevenlabs_age_list]="young" # type: ignore
    accent: Optional[elevenlabs_accent_list]="american" # type: ignore
    accent_strength: float = Query(default = 1, ge = 0.3, le = 2)
    

class AudioParameters(BaseModel):
    voice_id : Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    stability: float = Query(default=0.5,ge=0, le=1)
    similarity_boost: float = Query(default=0.75,ge=0, le=1)
    style: float = Query(default=0.0,ge=0, le=1)
    use_speaker_boost : Optional[bool] = True 

    @field_validator("voice_id")
    def validate_voice_id(cls, v):
        set_api_key(os.environ["ELEVENLABS_API_KEY"])
        voices_data : List[Voice] = voices()
        voice_dict=[]
        for i in voices_data:
            voice_dict.append(i.voice_id)
        if v not in voice_dict:
            raise Exception(ELEVENLABS_ERROR, VOICE_ID_ERROR_MSG)
        return v

class VoiceData(BaseModel):
    category: Optional[List[str]]=["premade"]
    description : Optional[str] = "Voice description"
    @field_validator("category")
    def validate_category(cls, v):
        choices = set(["cloned", "generated", "premade"])
        v=set(v)
        if not(v.issubset(choices)):
            raise Exception("Invalid input. The parameter must be one of: " + ", ".join(choices))
        return list(v)

class ElevenLabsParameters(BaseModel):
    prompt: str = Query(default="Hi, we are in Qolaba", max_length=2500, min_length=1) 
    clone: Optional[bool]= False
    clone_parameters : Optional[CloneParameters] = Field(default_factory=CloneParameters)
    voice_design: Optional[bool] = False 
    design_parameters : Optional[DesignParameters] = Field(default_factory=DesignParameters)
    generate_audio : Optional[bool]= True
    audio_parameters : Optional[AudioParameters] = Field(default_factory=AudioParameters)

    @model_validator(mode='after')
    def validate_params(self):
        total_sum=sum([self.clone, self.voice_design, self.generate_audio])
        if(not(total_sum==1)):
            raise Exception("Only one of 'clone', 'voicedesign','list_of_voices', or 'generate_audio' must be True")
        return self


# class ElevenLabsParameters(BaseModel):
#     prompt: str = Query(default="Hi, we are in Qolaba", max_length=2500, min_length=1) 
#     clone: Optional[bool]= False
#     name: Optional[str] = "Cloned Voice"
#     description : Optional[str] = "description"
#     list_of_files: List[str] = Query(default=["None"], max_length = MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS, min_length = MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS)
#     voice_design: Optional[bool] = False 
#     gender: Optional[elevenlabs_gender_list]="female" # type: ignore
#     age: Optional[elevenlabs_age_list]="young" # type: ignore
#     accent: Optional[elevenlabs_accent_list]="american" # type: ignore
#     accent_strength: float = Query(default = 1, ge = 0.3, le = 2)
#     generate_audio : Optional[bool]= True
#     voice_id : Optional[str] = "21m00Tcm4TlvDq8ikWAM"
#     stability: float = Query(default=0.5,ge=0, le=1)
#     similarity_boost: float = Query(default=0.75,ge=0, le=1)
#     style: float = Query(default=0.0,ge=0, le=1)
#     use_speaker_boost : Optional[bool] = True 

    
#     @model_validator(mode='after')
#     def validate_params(self):
#         total_sum=sum([self.clone, self.voice_design, self.generate_audio])
#         if(not(total_sum==1)):
#             raise ValueError("Only one of 'clone', 'voicedesign','list_of_voices', or 'generate_audio' must be True")
#         return self
    
#     @field_validator("voice_id")
#     def validate_voice_id(cls, v):
#         set_api_key(os.environ["ELEVENLABS_API_KEY"])
#         voices_data : List[Voice] = voices()
#         voice_dict=[]
#         for i in voices_data:
#             voice_dict.append(i.voice_id)
#         if v not in voice_dict:
#             raise ValueError("Invalid input. The parameter must be one of: " + ", ".join(voice_dict))
#         return v
    
class DalleParameters(BaseModel):
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    quality : Optional[dalle_supported_quality] = "standard"  # type: ignore

class OpenAITTSParameters(BaseModel):
    prompt : str

class TTSOutput(BaseModel):
    output : str | None
    cost : float

class SDXLAPITextToImageParameters(SDXLText2ImageParameters):
    style_preset : Optional[sdxl_preset_list] = "enhance" # type: ignore
    # seed : Optionalint


class SDXL3APITextToImageParameters(BaseModel):
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt : Optional[str]


class SDXL3APIImageToImageParameters(BaseModel):
    file_url : str 
    batch:  int = Query( ge = MIN_BATCH, le = MAX_BATCH)
    prompt: str
    negative_prompt : Optional[str]
    strength : float = Query(default = 0.7, gt = MIN_STRENGTH, le = MAX_STRENGTH)

class SDXLAPIImageToImageParameters(SDXLImage2ImageParameters):
    style_preset : Optional[sdxl_preset_list] = "enhance"# type: ignore
    height: int = Query(ge = MIN_HEIGHT, le = MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le = MAX_HEIGHT)
    # num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS) 

class SDXLAPIInpainting(BaseModel):
    file_url : str | Any
    mask_url : Optional[str | Any] = None
    prompt : str
    batch:  int = Query(default = 1, ge = MIN_BATCH, le = MAX_BATCH)
    negative_prompt: Optional[str] = " "

class PromptParrotParameters(BaseModel):
    prompt : str
    batch : int


class OOTDiffusionParameters(BaseModel):
    file_url : str | Any
    bg_img : str | Any
    batch:  int = Query(default = 1, ge = MIN_BATCH, le = MAX_BATCH)
    num_inference_steps: int = Query(default=30, ge=MIN_INFERENCE_STEPS, le=40)
    scale : int = Query(default=2, ge=1, le=5)
    category : Optional[Literal[0, 1, 2]] = 0

class APITaskResponse(BaseModel):
    time_required : Optional[dict] = {}
    error: Optional[str] = None
    error_data : Optional[str | dict] = None
    input: Optional[dict] = {}
    output: Optional[dict | list] = {}
    task_id: Optional[str] = None
    status: Literal["SUCCESS", "FAILED", "PENDING"] = "PENDING"

class APIInput(BaseModel):
    app_id : str
    parameters : Optional[dict] = {}
    init_parameters : Optional[dict] = {}
    ref_id: Optional[str] = ""
    celery: Optional[bool] = False
    inference_type : Optional[Literal["a10g", "a100", "h100"]] = "a10g"
    upscale : Optional[bool] = False

class TaskStatus(BaseModel):
    task_id: Optional[str] = None
    ref_id: Optional[str] = ""

