from pydantic import constr
import os

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
MAX_HEIGHT = 1536
MIN_HEIGHT = 256
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


#GOOGLE credential info
google_credentials_info = {
    "type": os.environ["GCP_TYPE"],
    "project_id": os.environ["GCP_PROJECT_ID"],
    "private_key_id": os.environ["GCP_PRIVATE_KEY_ID"],
    "private_key": os.environ["GCP_PRIVATE_KEY"].encode('utf-8').decode('unicode_escape'),
    "client_email": os.environ["GCP_CLIENT_EMAIL"],
    "client_id": os.environ["GCP_CLIENT_ID"],
    "auth_uri": os.environ["GCP_AUTH_URI"],
    "token_uri": os.environ["GCP_TOKEN_URI"],
    "auth_provider_x509_cert_url": os.environ["GCP_AUTH_PROVIDER_X509_CERT_URL"],
    "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"],
    # "universe_domain": os.environ["GCP_UNIVERSE_DOMAIN"]
}

BUCKET_NAME = "qolaba"
OUTPUT_IMAGE_EXTENSION = "png"
OUTPUT_AUDIO_EXTENSION = "mp3"
OUTPUT_VIDEO_EXTENSION = "mp4"

content_type = {
    OUTPUT_IMAGE_EXTENSION : "image/jpeg",
    OUTPUT_AUDIO_EXTENSION : "audio/mpeg",
    OUTPUT_VIDEO_EXTENSION : "video/mp4"
}

extra_negative_prompt="disfigured, kitsch, ugly, oversaturated, greain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye"

gender_word = "gender"

#Parameter dictionary
app_dict = {
    "ap-JOsvgUBfInC2UQnz0FQFkG": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "SDXL Turbo"
        },
        "model_name" : "sdxl_turbo_text2image" 
    },
    "ap-9ccz5olHjWxCVWjGTpLhr6": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "rev-anim"
        },
        "model_name" : "revanim_text2image" 
    },
    "ap-PZYd1Bb5QH57Rw4BF0dPA4": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Vibrant"
        },
        "model_name" : "pixela_text2image" 
    },
    "ap-RmIIykpGUHWzTZRmF0U4bH": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "3D Cartoon"
        },
        "model_name" : "3D_cartoon_text2image" 
    },
    "ap-VbAWaTpRHAcfqywOmcjN2y": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Cartoon"
        },
        "model_name" : "cartoon_text2image" 
    },
    "ap-z5opn6mGtcHtSh5kCTje1C": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Colorful"
        },
        "model_name" : "colorful_text2image" 
    },
    "ap-7vTmulaxjyiCzQEj1BOOvJ": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Anime"
        },
        "model_name" : "anime_text2image" 
    },
    "ap-onCEEZdTFZF4X8XJjYOHGR": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Anime 2"
        },
        "model_name" : "anime_2_text2image" 
    },
    "ap-uU4Xo4z6KgFHFjW9gnAl3q": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Realistic"
        },
        "model_name" : "realistic_text2image" 
    },
    "ap-9TlukeWaO8uJmjZsPzqmKB": {
        "app_id" : "SDXL_Text_To_Image_modal",
        "init_parameters" : {
            "model" : "Realistic 2"
        },
        "model_name" : "realistic_2_text2image" 
    },

    "ap-oDVXRQ7bCaWW05O52P1mlA": {
        "app_id" : "sdxltext2image_api",
        "init_parameters" : {
        },
        "model_name" : "SDXL_text2image" 
    },
    "ap-sdSyd0idsndjnsnsndjsds": {
        "app_id" : "dalletext2image_api",
        "init_parameters" : {
        },
        "model_name" : "dalle_text2image" 
    },

    "ap-9QVYLtWxXRzGH0qk97iG03": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "rev-anim"
        },
        "model_name" : "revanim_image2image" 
    },
    "ap-vcbg1l7bBuScbDuUOk3Shl": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Vibrant"
        },
        "model_name" : "pixela_image2image" 
    },
    "ap-1dONyJCjssrvvXLufmvGWJ": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "3D Cartoon"
        },
        "model_name" : "3D_cartoon_image2image" 
    },
    "ap-Rgtph3LFS9rRT5X7XREazo": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Cartoon"
        },
        "model_name" : "cartoon_image2image" 
    },
    "ap-wVf86rypMYb2pyMY46ikxw": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Colorful"
        },
        "model_name" : "colorful_image2image" 
    },
    "ap-Zj2zfXKq52zvpib7jGqYFp": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Anime"
        },
        "model_name" : "anime_image2image" 
    },
    "ap-ATG5IQ5s9jjrILhBbRNp4h": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Anime 2"
        },
        "model_name" : "anime_2_image2image" 
    },
    "ap-nM68b4WwQ6VzdNwwXBrOHE": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Realistic"
        },
        "model_name" : "realistic_image2image" 
    },
    "ap-6xdUez7nEduEYqDan7NTrh": {
        "app_id" : "SDXL_Image_To_Image_modal",
        "init_parameters" : {
            "model" : "Realistic 2"
        },
        "model_name" : "realistic_2_image2image" 
    },

    "ap-eRSyl0imzrUrbvgdAYwDTl": {
        "app_id" : "sdxlimage2image_api",
        "init_parameters" : {
        },
        "model_name" : "SDXL_image2image" 
    },

    "ap-ssndjnidsierrsnsnd223d": {
        "app_id" : "didvideo_api",
        "init_parameters" : {
        },
        "model_name" : "did_video" 
    },

    "ap-m9XvuyMHQI0W5K0aE13N9t": {
        "app_id" : "Stable_Video_Diffusion_modal",
        "init_parameters" : {
        },
        "model_name" : "stable_diffusion_video" 
    },

    "ap-a1Syd0inzrUrbvgdAbcser": {
        "app_id" : "elvenlabsaudio_api",
        "init_parameters" : {
        },
        "model_name" : "Elven_labs_audio" 
    },
    "ap-a1b2c3d4e5f6g7h8i9j0kq": {
        "app_id" : "SDXL_controlnet_modal",
        "init_parameters" : {
            "model" : "Colorful",
            "controlnet_model" : "openpose"
        },
        "model_name" : "Openpose_controlnet__image2image" 
    },
    "ap-1us0FK21Ach6eiWxo22is8": {
        "app_id" : "SDXL_controlnet_modal",
        "init_parameters" : {
            "model" : "Colorful",
            "controlnet_model" : "canny"
        },
        "model_name" : "Canny_controlnet__image2image" 
    },
    "ap-WrXnJBXy23XpPh6IlH5tRX": {
        "app_id" : "SDXL_controlnet_modal",
        "init_parameters" :{
            "model" : "Colorful",
            "controlnet_model" : "depth"
        },
        "model_name" : "depth_controlnet__image2image" 
    },
    "ap-ebD8V0XxzAuKSu696A11nd": {
        "app_id" : "SDXL_controlnet_modal",
        "init_parameters" : {
            "model" : "Colorful",
            "controlnet_model" : "sketch"
        },
        "model_name" : "sketch_controlnet__image2image" 
    },
    "ap-7yMwQ4XtC3ZrP2jLHJ8bB1": {
        "app_id" : "clipdropuncropimage2image_api",
        "init_parameters" : {
        },
        "model_name" : "clipdrop_uncrop_image2image" 
    },
    "ap-7XmHk4LtVgFJq2cQoE3yB8": {
        "app_id" : "clipdropcleanupimage2image_api",
        "init_parameters" : {
        },
        "model_name" : "clipdrop_cleanup_image2image" 
    },
    "ap-L4vHj7YbXeT2qKoU1fW3G7": {
        "app_id" : "clipdropreplacebackgroundimage2image_api",
        "init_parameters" : {
        },
        "model_name" : "clipdrop_Replace_background_image2image" 
    },
    "ap-9k8P3v6zBq2s5T1rXwQm45": {
        "app_id" : "clipdropremovetextimage2image_api",
        "init_parameters" : {
        },
        "model_name" : "clipdrop_remove_text_image2image" 
    },
    "ap-TYMECaYPMCcOyK2ZgNHgqQ": {
        "app_id" : "QRCode_Generation_modal", 
        "init_parameters" : {
        },
        "model_name" : "QR_code_image2image" 
    },
    "ap-lpocU0cB9szyuvta9lZ83B": {
        "app_id" : "Illusion_Diffusion_modal",
        "init_parameters" : {
        },
        "model_name" : "Illusion_Diffusion_image2image"  
    },
    "ap-SwsTOsD7STxJ0obCkskUHw": {
        "app_id" : "IPAdapter_face_consistent_modal",
        "init_parameters" : {
            "model" : "Colorful"
        },
        "model_name" : "face_consistent_image2image"
    },
    "ap-jL9u1wFpY5sRv2XmN3aQ44": {
        "app_id" : "IPAdapter_FRND_face_consistent_modal",
        "init_parameters" : {
        },
        "model_name" : "face_consistent_FRND_image2image"
    },
    "ap-kV3kwXL9QXJxYS6jV1VtkT": {
        "app_id" : "BackGround_Removal_modal",
        "init_parameters" : {
        },
        "model_name" : "Background_removal_image2image"
    },
    "ap-kNvCpegQY2HiDHmk5X7X6F": {
        "app_id" : "IPAdapter_image_variation_modal",
        "init_parameters" : {
            "model" : "Colorful"
        },
        "model_name" : "variation_image2image" 
    },
    "ap-NFDMz1Rn9UlpS6zMl2FNpm": {
        "app_id" : "Ultrasharp_Upscaler_modal",
        "init_parameters" : {
        },
        "model_name" : "upscaling_image2image" 
    }
}

