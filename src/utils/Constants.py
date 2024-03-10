# CONTROLNET_COMMANDS = [
#     "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
#     "apt-get update && apt-get install wget -y",
#     "wget https://civitai.com/api/download/models/182077",
#     "pip install diffusers --upgrade",
#     "pip install controlnet_aux invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf pydantic cloudinary",
#     "mv 182077 Starlight.safetensors",
# ]

# NORMAL_CONTROLNET_COMMANDS = [
#             "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
#             "pip install diffusers transformers accelerate opencv-python Pillow xformers controlnet_aux matplotlib cloudinary",
#         ]

# STARLIGHT_SAFETENSORS_PATH = "../Starlight.safetensors"
# QOLABA_SERVER_UPLOAD_CLOUDINARY_URL = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
# CLOUDINARY_SECRET = "cloudinary-secrets"
# CANNY_CONTROLNET_IMAGETOIMAGE = "canny_controlnet_image2image"
# DEPTH_CONTROLNET_IMAGETOIMAGE = "depth_controlnet_image2image"
# SD_REVANIM = "stablediffusionapi/rev-anim"
# NORMAL_CONTROLNET_IMAGETOIMAGE = "normal_controlnet_image2image"

from pydantic import constr

#Volume variables
VOLUME_NAME = "SDXL-LORA-Volume"
VOLUME_PATH = "/SDXL_models"

#Environment variables
PYTHON_VERSION = "3.11.8"

BASE_IMAGE_COMMANDS = [
    "apt-get update && apt-get install ffmpeg libsm6 libxext6 git curl wget pkg-config libssl-dev openssl git-lfs -y",
    "git lfs install",
    "pip install torch torchvision torchaudio"
]

REQUIREMENT_FILE_PATH = "requirements.txt"
SECRET_NAME = "environment_configuration"

SDXL_ANIME_MODEL = "/SDXL_models/Animagine.safetensors"
SDXL_ANIME_2_MODEL = "/SDXL_models/AstreaPixie.safetensors"
SDXL_REALISTIC_MODEL = "/SDXL_models/NewRealityXL.safetensors"
SDXL_REALISTIC_2_MODEL = "/SDXL_models/RealismEngine.safetensors"
SDXL_PIXELA_MODEL = "/SDXL_models/ProtoVision.safetensors"
SDXL_COLORFUL_MODEL = "/SDXL_models/Starlight.safetensors"
SDXL_REVANIME_MODEL = "/SDXL_models/FaeTastic.safetensors"
SDXL_CARTOON_MODEL = "/SDXL_models/DeepBlue.safetensors"
SDXL_3DCARTOON_MODEL = "/SDXL_models/DynaVision.safetensors"
SDXL_TURBO_MODEL = "/SDXL_models/TurboVisionXL.safetensors"

sdxl_model_list = {
    "rev-anim" : SDXL_REVANIME_MODEL,
    "Vibrant" : SDXL_PIXELA_MODEL,
    "Colorful" : SDXL_COLORFUL_MODEL,
    "Realistic" : SDXL_REALISTIC_MODEL,
    "Realistic 2" : SDXL_REALISTIC_2_MODEL, 
    "Anime" : SDXL_ANIME_MODEL,
    "Anime 2" : SDXL_ANIME_2_MODEL,
    "Cartoon" : SDXL_CARTOON_MODEL,
    "3D Cartoon" : SDXL_3DCARTOON_MODEL,
    "SDXL Turbo" : SDXL_TURBO_MODEL
}

sdxl_model_string = "|".join(sdxl_model_list.keys())



SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
SDXL_REFINER_MODEL_PATH = "/SDXL_models/sdxl_model_refiner"

OPENPOSE_PATH = "/SDXL_models/openpose"
SKETCH_PATH = "/SDXL_models/sketch"
CANNY_PATH = "/SDXL_models/canny"
DEPTH_PATH = "/SDXL_models/depth"

CANNY = "canny"
OPENPOSE = "openpose"
SKETCH = "sketch"
DEPTH = "depth"

controlnet_model_list = {
    OPENPOSE : OPENPOSE_PATH,
    SKETCH : SKETCH_PATH,
    CANNY : CANNY_PATH,
    DEPTH : DEPTH_PATH
}

controlnet_models = "|".join(controlnet_model_list.keys())

ULTRASHARP_MODEL = "/SDXL_models/4x-UltraSharp.pth"


# Text_To_Image Configuration
MAX_HEIGHT = 2048
MIN_HEIGHT = 512
MAX_INFERENCE_STEPS = 50
MIN_INFERENCE_STEPS = 5
MAX_GUIDANCE_SCALE = 30
MIN_GUIDANCE_SCALE = 0
MAX_BATCH = 8
MIN_BATCH = 1

#Image_To_Image Configuration
MAX_STRENGTH = 1
MIN_STRENGTH = 0

#Background removal Configuration
MAX_COLOR = 255
MIN_COLOR = 0

#stable video configuration
MAX_FPS = 30
MIN_FPS = 5

#Clipdrop side increase parameters
MIN_INCREASE_SIDE = 0
MAX_INCREASE_SIDE = 2048

#Did api configuration
DID_TALK_API = "https://api.d-id.com/talks"
DID_AVATAR_STYLES = ["circle", "normal", "closeUp"]
did_avatar_styles = constr(pattern = '|'.join(DID_AVATAR_STYLES))
DID_EXPRESSION_LIST = ["surprise", "happy", "serious", "neutral" ]
did_expression_list = constr(pattern = '|'.join(DID_EXPRESSION_LIST))

#SDXL API configuration
STABILITY_API = "https://api.stability.ai"
SDXL_ENGINE_ID = "stable-diffusion-xl-1024-v1-0"
SDXL_DEFAULT_PRESET = "enhance"
SDXL_STYLE_PRESET_LIST = ["3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk", "origami", "photographic", "pixel-art", "tile-texture"]
sdxl_preset_list = constr(pattern = '|'.join(SDXL_STYLE_PRESET_LIST))

#clipdrop api configuration
CLIPDROP_UNCROP_URL = "https://clipdrop-api.co/uncrop/v1"
CLIPDROP_CLEANUP_URL = "https://clipdrop-api.co/cleanup/v1"
CLIPDROP_REPLACE_BACKGROUND_URL = "https://clipdrop-api.co/replace-background/v1"
CLIPDROP_REMOVE_TEXT_URL = "https://clipdrop-api.co/remove-text/v1"


#Elevenlabs configuration
MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS = 3
MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS = 1
ELEVENLABS_GENDER_LIST = ["female","male"]
ELEVENLABS_AGE_LIST = ['young', 'middle_aged', 'old']
ELEVENLABS_ACCENT_LIST = ['british', 'american', 'african', 'australian', 'indian']
elevenlabs_age_list = constr(pattern = '|'.join(ELEVENLABS_AGE_LIST))
elevenlabs_accent_list = constr(pattern = '|'.join(ELEVENLABS_ACCENT_LIST))
elevenlabs_gender_list = constr(pattern = '|'.join(ELEVENLABS_GENDER_LIST))

#Dalle configuration
DALLE_SUPPORTED_HW = ["1024x1024", "1024x1792", "1792x1024"]
DALLE_SUPPORTED_QUALITY = ["hd", "standard"]
dalle_supported_quality = constr(pattern = '|'.join(DALLE_SUPPORTED_QUALITY))


# Celery configuration
CELERY_RESULT_EXPIRATION_TIME = 14400
REDIS_URL = "redis://localhost:6379/0"
CELERY_MAX_RETRY = 1
CELERY_SOFT_LIMIT = 7200

#Modal app cache configuration
MAX_TIME_MODAL_APP_CACHE =  3600

#