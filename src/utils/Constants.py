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
PYTHON_VERSION = "3.11"

BASE_IMAGE_COMMANDS = [
    "apt-get update && apt-get install ffmpeg libsm6 libxext6 git curl wget pkg-config libssl-dev openssl -y",
    "pip3 install torch torchvision torchaudio"
]

REQUIREMENT_FILE_PATH = "requirements.txt"

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

sdxl_model_string = constr(pattern="|".join(sdxl_model_list.keys()))

SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
SDXL_REFINER_MODEL_PATH = "/SDXL_models/sdxl_model_refiner"

OPENPOSE_MODEL = "TencentARC/t2i-adapter-openpose-sdxl-1.0"
OPENPOSE_PATH = "/SDXL_models/openpose"

SKETCH_MODEL = "TencentARC/t2i-adapter-sketch-sdxl-1.0"
SKETCH_PATH = "/SDXL_models/sketch"

CANNY_MODEL = "TencentARC/t2i-adapter-canny-sdxl-1.0"
CANNY_PATH = "/SDXL_models/canny"

DEPTH_MODEL = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
DEPTH_PATH = "/SDXL_models/depth"


# Text_To_Image Configuration
MAX_HEIGHT = 2048
MIN_HEIGHT = 512
MAX_INFERENCE_STEPS = 50
MIN_INFERENCE_STEPS = 5
MAX_GUIDANCE_SCALE = 30
MIN_GUIDANCE_SCALE = 0
MAX_BATCH = 8
MIN_BATCH = 1
