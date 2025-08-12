from typing import Any, List, Literal, Optional, Dict # Added Dict

from fastapi import Query
from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.Constants import *
from src.utils.Constants import (
    MAX_GUIDANCE_SCALE,
    MIN_GUIDANCE_SCALE,
    controlnet_models,
    sdxl_model_string,
)  # Ensure these are imported if not already


class StubNames(BaseModel):
    sdxl_text_to_image: str = "SDXL_Text_To_Image"
    sdxl_image_to_image: str = "SDXL_Image_To_Image"
    sdxl_controlnet: str = "SDXL_controlnet"
    ultrasharp_upscaler: str = "Ultrasharp_Upscaler"
    image_variation: str = "IPAdapter_image_variation"
    face_consistent: str = "IPAdapter_face_consistent"
    background_removal: str = "BackGround_Removal"
    stable_video_diffusion: str = "Stable_Video_Diffusion"
    illusion_diffusion: str = "Illusion_Diffusion"
    qr_code_generation: str = "QRCode_Generation"
    frnd_face_consistent: str = "IPAdapter_FRND_face_consistent"
    stable_cascade_text_to_image: str = "Stable_Cascade"
    oot_diffusion: str = "OOTDiffusion"
    hair_fast: str = "HairFast"
    flux_refiner: str = "flux_refiner"


class StubConfiguration(BaseModel):
    memory: int
    container_idle_timeout: int
    gpu: str
    num_containers: int


class TimeData(BaseModel):
    startup_time: int | float
    runtime: int | float


class TaskResponse(BaseModel):
    result: Any
    Has_NSFW_Content: list[bool]
    Has_copyrighted_Content: Optional[list[bool]] = None
    low_res_urls: Optional[list[str]] = []
    time: TimeData
    extension: str | None


class HairFastParameters(BaseModel):
    face_url: str
    shape_url: str
    color_url: str
    align_images: bool = True


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


class MusicGenParameters(BaseModel):
    prompt: str


class PromptImage(BaseModel):
    uri: str
    position: Literal["first", "last"] = "first"


class LumaLabsVideoParameters(BaseModel):
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"] = "1:1"
    prompt: str
    loop: Optional[bool] = False
    file_url: Optional[list[PromptImage]] = None
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    model: str = "ray-2"


class RunwayImage2VideoParameters(BaseModel):
    prompt: str
    file_url: Optional[list[PromptImage]] = None
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    duration: Literal[5, 10] = 5
    aspect_ratio: Literal[
        "1280:720",
        "720:1280",
        "960:960",
        "1104:832",
        "832:1104",
        "1584:672"
    ] = "1280:720"


class Kling15Video(BaseModel):
    prompt: str
    file_url: Optional[list[PromptImage]] = None
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    duration: Literal[5, 10] = 5
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"


class MinimaxVideo(BaseModel):
    prompt: str
    file_url: Optional[list[PromptImage]] = None
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)


class HunyuanVideo(BaseModel):
    prompt: str
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    aspect_ratio: Optional[Literal["16:9", "9:16"]] = "16:9"


class IdeoGramText2ImageParameters(BaseModel):
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: Optional[str] = " "
    negative_prompt: Optional[str] = " "
    aspect_ratio: Optional[str] = "1:1"
    style_type: Literal[
        "AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"
    ] = "AUTO"
    magic_prompt_option: Literal["AUTO", "ON", "OFF"] = "AUTO"
    file_url: Optional[str | Any] = None
    mask_url: Optional[str | Any] = None


class IdeoGramRemixParameters(BaseModel):
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: Optional[str] = " "
    aspect_ratio: Optional[str] = "1:1"
    magic_prompt_option: Literal["AUTO", "ON", "OFF"] = "AUTO"
    file_url: Optional[str | Any] = None
    color_palette: Literal[
        "EMBER", "FRESH", "JUNGLE", "MAGIC", "MELON", "MOSAIC", "PASTEL", "ULTRAMARINE"
    ] = "FRESH"
    strength: float = Query(default=0.7, gt=MIN_STRENGTH, le=MAX_STRENGTH)
    negative_prompt: Optional[str] = " "
    style_type: Literal[
        "AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"
    ] = "AUTO"


class FluxText2ImageParameters(BaseModel):
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    # num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS)
    # guidance_scale:  float = Query(ge = 2, le = 5)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    interval: float = Query(default=2, ge=1, le=4)
    safety_tolerance: float = Query(default=2, ge=1, le=5)
    aspect_ratio: Optional[str] = "1:1"
    file_url: Optional[str] = None
    output_quality: Optional[int] = 100


class RecraftV3Text2ImageParameters(BaseModel):
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    style: Optional[recraft_v3_style_cond] = "any"  # type:ignore


class FluxImage2ImageParameters(BaseModel):
    num_inference_steps: int = Query(
        default=30, ge=MIN_INFERENCE_STEPS, le=MAX_INFERENCE_STEPS
    )
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    interval: float = Query(default=2, ge=1, le=4)
    safety_tolerance: float = Query(default=2, ge=1, le=5)
    aspect_ratio: Optional[str] = "1:1"
    file_url: Optional[str] = None
    output_quality: Optional[int] = 100
    strength: float = Query(default=0.7, gt=MIN_STRENGTH, le=MAX_STRENGTH)
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    negative_prompt: Optional[str] = " "
    # @model_validator(mode='before')
    # def validate_params(self):
    #     self["guidance_scale"] = 2 + ((self["guidance_scale"] - 4)/ 8) * 3
    #     return self


class OmnigenParameters(BaseModel):
    num_inference_steps: int = Query(
        default=50, ge=MIN_INFERENCE_STEPS, le=MAX_INFERENCE_STEPS
    )
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    interval: float = Query(default=2, ge=1, le=4)
    safety_tolerance: float = Query(default=2, ge=1, le=5)
    aspect_ratio: Optional[str] = "1:1"
    file_url: Optional[list[str] | str] = None
    output_quality: Optional[int] = 100
    strength: float = Query(default=0.7, gt=MIN_STRENGTH, le=MAX_STRENGTH)
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)


class SDXLText2ImageParameters(BaseModel):
    height: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    num_inference_steps: int = Query(ge=MIN_INFERENCE_STEPS, le=MAX_INFERENCE_STEPS)
    guidance_scale: float = Query(ge=MIN_GUIDANCE_SCALE, le=MAX_GUIDANCE_SCALE)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale: float = Query(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def validate_params(self):
        self.height = self.height - (self.height % HW_MULTIPLE)
        self.width = self.width - (self.width % HW_MULTIPLE)
        return self


class HairStyleParameters(BaseModel):
    file_url: str | Any
    hair_style: Optional[available_hairstyles] = "ManBun"  # type:ignore
    color: Optional[available_haircolors] = "brown"  # type:ignore
    batch: int = Query(ge=MIN_BATCH, le=4)


class SDXLImage2ImageParameters(BaseModel):
    file_url: str | Any
    strength: float = Query(default=0.7, gt=MIN_STRENGTH, le=MAX_STRENGTH)
    guidance_scale: float = Query(ge=MIN_GUIDANCE_SCALE, le=MAX_GUIDANCE_SCALE)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str] = " "
    lora_scale: float = Query(default=0.5, gt=0, le=1)


class SDXLControlNetParameters(SDXLImage2ImageParameters):
    num_inference_steps: int = Query(ge=MIN_INFERENCE_STEPS, le=MAX_INFERENCE_STEPS)


class UpscaleParameters(BaseModel):
    file_url: str | Any
    scale: Literal[2, 4, 8]
    check_nsfw: Optional[bool] = True
    strength: float = Query(default=0.5, gt=MIN_STRENGTH, le=MAX_STRENGTH)


class VariationParameters(SDXLControlNetParameters):
    prompt: Optional[str] = None


class FaceConsistentParameters(SDXLText2ImageParameters):
    file_url: str | Any
    strength: float = Query(gt=MIN_STRENGTH, le=MAX_STRENGTH)

    @model_validator(mode="after")
    def validate_params(self):
        self.height = self.height - (self.height % HW_MULTIPLE)
        self.width = self.width - (self.width % HW_MULTIPLE)
        return self


class FRNDFaceAvatarParameters(BaseModel):
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=1024)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=1024)
    num_inference_steps: int = Query(default=30, ge=MIN_INFERENCE_STEPS, le=40)
    guidance_scale: float = Query(
        default=7.5, ge=MIN_GUIDANCE_SCALE, le=MAX_GUIDANCE_SCALE
    )
    batch: int = Query(default=1, ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str | None = (
        f"((establishing shot)), (((smiling studio photo))), joyous indian dating aesthetic, 3D pixar, looking at camera, enjoyable pleasurable {gender_word}, ((black winter t-shirt)), proper eyes, Relaxed, Charming, Cordial,  Relaxed, Charming, Cordial, Gracious, 3d bitmoji avatar render, dark background"
    )
    negative_prompt: str | None = " "
    file_url: str | Any
    strength: float = Query(default=1, gt=MIN_STRENGTH, le=MAX_STRENGTH)
    gender: Literal["male", "female"]
    remove_background: bool
    bg_color: Optional[str] = None

    @model_validator(mode="after")
    def validate_params(self):
        self.height = self.height - (self.height % HW_MULTIPLE)
        self.width = self.width - (self.width % HW_MULTIPLE)
        return self


class InitParameters(BaseModel):
    model: str = Field(pattern=sdxl_model_string)
    lora_model: Optional[str] = None
    controlnet_model: Optional[str] = Field(default=None, pattern=controlnet_models)


class BackGroundRemoval(BaseModel):
    file_url: str | Any
    bg_img: Optional[str] = None
    bg_color: Optional[bool] = False
    r_color: int = Query(default=MIN_COLOR, ge=MIN_COLOR, le=MAX_COLOR)
    g_color: int = Query(default=MIN_COLOR, ge=MIN_COLOR, le=MAX_COLOR)
    b_color: int = Query(default=MIN_COLOR, ge=MIN_COLOR, le=MAX_COLOR)
    blur: Optional[bool] = False


class StableVideoDiffusion(BaseModel):
    file_url: str | Any
    fps: int = Query(default=MIN_FPS, ge=MIN_FPS, le=MAX_FPS)


class IllusionDuiffusion(SDXLImage2ImageParameters):
    num_inference_steps: int = Query(ge=MIN_INFERENCE_STEPS, le=MAX_INFERENCE_STEPS)


class DIDVideoParameters(BaseModel):
    file_url: str | Any
    expression: Optional[did_expression_list] = "neutral"  # type: ignore
    expression_intesity: float = Query(default=1, ge=0, le=1)
    voice_id: Optional[str] = "d7bbcdd6964c47bdaae26decade4a933"
    prompt: Optional[str] = "Hey, welcome to Qolaba"


class CloneParameters(BaseModel):
    name: Optional[str] = "Cloned Voice"
    description: Optional[str] = "description"
    list_of_files: List[str] = Query(
        default=["None"],
        max_length=MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS,
        min_length=MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS,
    )


class DesignParameters(BaseModel):
    name: Optional[str] = "Designed Voice"
    description: Optional[str] = "description"
    gender: Optional[elevenlabs_gender_list] = "female"  # type: ignore
    age: Optional[elevenlabs_age_list] = "young"  # type: ignore
    accent: Optional[elevenlabs_accent_list] = "american"  # type: ignore
    accent_strength: float = Query(default=1, ge=0.3, le=2)


class AudioParameters(BaseModel):
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    public_id: Optional[str] = None
    stability: float = Query(default=0.5, ge=0, le=1)
    similarity_boost: float = Query(default=0.75, ge=0, le=1)
    style: float = Query(default=0.0, ge=0, le=1)
    use_speaker_boost: Optional[bool] = True


class VoiceData(BaseModel):
    category: Optional[List[str]] = ["premade"]
    description: Optional[str] = "Voice description"

    @field_validator("category")
    def validate_category(cls, v):
        choices = set(["cloned", "generated", "premade"])
        v = set(v)
        if not (v.issubset(choices)):
            raise Exception(
                "Invalid input. The parameter must be one of: " + ", ".join(choices)
            )
        return list(v)


class ElevenLabsParameters(BaseModel):
    prompt: str = Query(max_length=69216, min_length=1)
    clone: Optional[bool] = False
    clone_parameters: Optional[CloneParameters] = Field(default_factory=CloneParameters)
    voice_design: Optional[bool] = False
    design_parameters: Optional[DesignParameters] = Field(
        default_factory=DesignParameters
    )
    generate_audio: Optional[bool] = True
    audio_parameters: Optional[AudioParameters] = Field(default_factory=AudioParameters)

    @model_validator(mode="after")
    def validate_params(self):
        total_sum = sum([self.clone, self.voice_design, self.generate_audio])
        if not (total_sum == 1):
            raise Exception(
                "Only one of 'clone', 'voicedesign','list_of_voices', or 'generate_audio' must be True"
            )
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
    height: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    quality: Optional[dalle_supported_quality] = "standard"  # type: ignore


class GPTImageParameters(BaseModel):
    height: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    quality: Optional[dalle_supported_quality] = "auto"  # type: ignore
    image: Optional[str] = None


class OpenAITTSParameters(BaseModel):
    prompt: str


class TTSOutput(BaseModel):
    output: str | None
    cost: float


class SDXLAPITextToImageParameters(SDXLText2ImageParameters):
    style_preset: Optional[sdxl_preset_list] = "enhance"  # type: ignore
    # seed : Optionalint


class SDXL3APITextToImageParameters(BaseModel):
    height: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str]


class SDXL3APIImageToImageParameters(BaseModel):
    file_url: str
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    prompt: str
    negative_prompt: Optional[str]
    strength: float = Query(default=0.7, gt=MIN_STRENGTH, le=MAX_STRENGTH)


class SDXLAPIImageToImageParameters(SDXLImage2ImageParameters):
    style_preset: Optional[sdxl_preset_list] = "enhance"  # type: ignore
    height: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(ge=MIN_HEIGHT, le=MAX_HEIGHT)
    # num_inference_steps: int = Query(ge = MIN_INFERENCE_STEPS, le = MAX_INFERENCE_STEPS)


class SDXLAPIInpainting(BaseModel):
    file_url: str | Any
    mask_url: Optional[str | Any] = None
    prompt: str
    batch: int = Query(default=1, ge=MIN_BATCH, le=MAX_BATCH)
    negative_prompt: Optional[str] = " "


class PromptParrotParameters(BaseModel):
    prompt: str
    batch: int


class OOTDiffusionParameters(BaseModel):
    file_url: str | Any
    bg_img: str | Any
    batch: int = Query(default=1, ge=MIN_BATCH, le=MAX_BATCH)
    num_inference_steps: int = Query(default=30, ge=MIN_INFERENCE_STEPS, le=40)
    scale: int = Query(default=2, ge=1, le=5)
    category: Optional[Literal[0, 1, 2]] = 0


class APITaskResponse(BaseModel):
    time_required: Optional[dict] = {}
    error: Optional[str] = None
    error_data: Optional[str | dict] = None
    input: Optional[dict] = {}
    output: Optional[dict | list] = {}
    task_id: Optional[str] = None
    status: Literal["SUCCESS", "FAILED", "PENDING"] = "PENDING"


class APIInput(BaseModel):
    app_id: str
    parameters: Optional[dict] = {}
    init_parameters: Optional[dict] = {}
    ref_id: Optional[str] = ""
    celery: Optional[bool] = False
    inference_type: Optional[Literal["a10g", "a100", "h100"]] = "a10g"
    upscale: Optional[bool] = False
    check_copyright_content: Optional[bool] = True


class TaskStatus(BaseModel):
    task_id: Optional[str] = None
    ref_id: Optional[str] = ""


class ChunkSummary(BaseModel):
    summary: str
    topic: str


class Timestamp(BaseModel):
    start: float
    end: float


class SubtopicTimestamps(BaseModel):
    start: float
    chapter: str


class Subtopics(BaseModel):
    topics: list[SubtopicTimestamps]


class TranslationContent(BaseModel):
    start: float
    end: float
    translated_content: str


class ListofTranslation(BaseModel):
    translation_list: list[TranslationContent]


class NSFWSchema(BaseModel):
    NSFW_content: bool


class Veo2Parameters(BaseModel):
    prompt: str | None = "A lego chef cooking eggs"
    image: Optional[str | None] = None  # Changed default to None for clarity
    file_url: Optional[List | None] = None
    # Keep the Literal type for the final validated field
    duration: Literal["5s", "6s", "7s", "8s"] = "5s"
    aspect_ratio: Literal["16:9", "9:16", "auto_prefer_portrait", "auto"] = "16:9"
    cfg_scale: float = Query(default=0.5, ge=MIN_GUIDANCE_SCALE, le=MAX_GUIDANCE_SCALE)

    @field_validator("duration", mode="before")
    @classmethod
    def format_duration(cls, v):
        allowed_ints = {5, 6, 7, 8}
        allowed_strs = {"5s", "6s", "7s", "8s"}

        if isinstance(v, int):
            if v in allowed_ints:
                return f"{v}s"
            else:
                raise ValueError(f"Integer duration must be one of {allowed_ints}")
        elif isinstance(v, str):
            # Allow valid strings directly or try converting string digits
            if v in allowed_strs:
                return v
            elif v.isdigit():
                int_v = int(v)
                if int_v in allowed_ints:
                    return f"{int_v}s"
                else:
                    raise ValueError(
                        f"String duration '{v}' must represent one of {allowed_ints}"
                    )
            else:
                raise ValueError(
                    f"String duration '{v}' is not in the allowed format {allowed_strs}"
                )
        # Raise error for any other type
        raise TypeError(
            f"Duration must be an integer {allowed_ints} or string {allowed_strs}"
        )

class Veo3Parameters(BaseModel):
    prompt: str
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    duration: Literal["8s"] = "8s"
    negative_prompt: Optional[str] = None
    enhance_prompt: bool = True
    seed: Optional[int] = None
    generate_audio: bool = True

    @field_validator("duration", mode="before")
    @classmethod
    def format_duration(cls, v):
        if isinstance(v, int):
            if v == 8:
                return "8s"
            else:
                raise ValueError("Duration must be 8 seconds")
        elif isinstance(v, str):
            if v == "8s":
                return v
            elif v.isdigit() and int(v) == 8:
                return "8s"
            else:
                raise ValueError("Duration must be '8s'")
        raise TypeError("Duration must be an integer 8 or string '8s'")

class Kling2MasterParameters(BaseModel):
    prompt: str | None = (
        "slow-motion sequence captures the catastrophic implosion of a skyscraper, dust and debris billowing outwards in a chaotic ballet of destruction, while a haunting, orchestral score underscores the sheer power and finality of the event."
    )
    negative_prompt: str = ""
    file_url: Optional[str] = None
    # Keep Literal type for the final validated field
    duration: Literal["5", "10"] = "5"
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    cfg_scale: float = Query(
        default=0.7,  # Or 1.0, check Fal AI docs for typical/recommended value
        ge=0.0,  # Assuming minimum is 0.0, adjust if needed
        le=1.0,  # Set maximum allowed value to 1.0
    )

    @field_validator("duration", mode="before")
    @classmethod
    def format_duration(cls, v):
        allowed_ints = {5, 10}
        allowed_strs = {"5", "10"}

        if isinstance(v, int):
            if v in allowed_ints:
                return str(v)  # Convert valid int to string
            else:
                raise ValueError(f"Integer duration must be one of {allowed_ints}")
        elif isinstance(v, str):
            # Allow valid strings directly or check if it's a digit string
            if v in allowed_strs:
                return v
            elif v.isdigit():
                int_v = int(v)
                if int_v in allowed_ints:
                    return str(int_v)  # Return the valid string representation
                else:
                    raise ValueError(
                        f"String duration '{v}' must represent one of {allowed_ints}"
                    )
            else:
                raise ValueError(
                    f"String duration '{v}' is not in the allowed format {allowed_strs}"
                )
        # Raise error for any other type
        raise TypeError(
            f"Duration must be an integer {allowed_ints} or string {allowed_strs}"
        )


class Lyria2MusicGenerationParameters(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    sample_count: Optional[int] = None
    seed: Optional[int] = None
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)

    @model_validator(mode="after")
    def check_exclusive_fields(self):
        if self.sample_count is not None and self.seed is not None:
            raise ValueError("sample_count and seed cannot be set at the same time.")
        if self.sample_count is None and self.seed is None:
            # Default to sample_count = 1 if neither is provided, as per notebook examples
            # However, the notebook sometimes calls with only prompt.
            # The API might default sample_count or handle it.
            # For now, let's not enforce one if not present, to match notebook flexibility.
            pass
        return self


class FluxKontextMaxMultiInputParameters(BaseModel):
    prompt: str = Field(...)
    seed: Optional[int] = Field(default=None)
    guidance_scale: Optional[float] = Field(default=3.5)
    sync_mode: Optional[bool] = Field(default=None)
    num_images: Optional[int] = Field(default=1)
    safety_tolerance: Optional[str] = Field(default="2")
    output_format: Optional[str] = Field(default="png")
    aspect_ratio: Literal["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"] = "16:9"
    batch: int = Query(ge=MIN_BATCH, le=MAX_BATCH)
    file_urls: List[str] = None
    height: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    width: int = Query(default=1024, ge=MIN_HEIGHT, le=MAX_HEIGHT)


