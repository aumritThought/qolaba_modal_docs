from modal import Stub, Volume, Image, Secret
from src.utils.Constants import *
import os
from diffusers import DiffusionPipeline
import torch
from diffusers import ControlNetModel, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL



stub = Stub("volume-stub", secrets=[Secret.from_name("environment_configuration")])

image = (
    Image.debian_slim(python_version = PYTHON_VERSION)
    .run_commands(BASE_IMAGE_COMMANDS)
    .pip_install_from_requirements(REQUIREMENT_FILE_PATH)
)

vol = Volume.persisted(VOLUME_NAME)

@stub.function(volumes={VOLUME_PATH: vol} ,  image = image, timeout = 7200)
def download_models():
    #3D Cartoon model
    os.system("wget https://civitai.com/api/download/models/297740")
    os.system(f"mv 297740 {SDXL_3DCARTOON_MODEL}")

    #Cartoon model
    os.system("wget https://civitai.com/api/download/models/365017")
    os.system(f"mv 365017 {SDXL_CARTOON_MODEL}")

    #Pixela Model
    os.system("wget https://civitai.com/api/download/models/265938")
    os.system(f"mv 265938 {SDXL_PIXELA_MODEL}")

    #Colorful model
    os.system("wget https://civitai.com/api/download/models/182077")
    os.system(f"mv 182077 {SDXL_COLORFUL_MODEL}")

    #rev-anime model
    os.system("wget https://civitai.com/api/download/models/291443")
    os.system(f"mv 291443 {SDXL_REVANIME_MODEL}")

    #Realistic Model
    os.system("wget https://civitai.com/api/download/models/312982")
    os.system(f"mv 312982 {SDXL_REALISTIC_MODEL}")

    #Realistic 2 model
    os.system("wget https://civitai.com/api/download/models/293240")
    os.system(f"mv 293240 {SDXL_REALISTIC_2_MODEL}")

    #SDXL Turbo model
    os.system("wget https://civitai.com/api/download/models/273102")
    os.system(f"mv 273102 {SDXL_TURBO_MODEL}")

    #Anime model
    os.system("wget https://civitai.com/api/download/models/133163")
    os.system(f"mv 133163 {SDXL_ANIME_MODEL}")

    #Anime 2 model
    os.system("wget https://civitai.com/api/download/models/293564")
    os.system(f"mv 293564 {SDXL_ANIME_2_MODEL}")

    
    os.mkdir(SDXL_REFINER_MODEL_PATH)

    refiner = DiffusionPipeline.from_pretrained(
            SDXL_REFINER_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    refiner.save_pretrained(SDXL_REFINER_MODEL_PATH)


    T2IAdapter.from_pretrained(
        OPENPOSE_MODEL, torch_dtype=torch.float16
    ).save_pretrained(OPENPOSE_PATH)

    T2IAdapter.from_pretrained(
        SKETCH_MODEL, torch_dtype=torch.float16
    ).save_pretrained(SKETCH_PATH)

    ControlNetModel.from_pretrained(
        CANNY_MODEL, torch_dtype=torch.float16
    ).save_pretrained(CANNY_PATH)

    ControlNetModel.from_pretrained(
        DEPTH_MODEL, torch_dtype=torch.float16
    ).save_pretrained(DEPTH_PATH)

    vol.commit()



@stub.local_entrypoint()
def run():
    download_models.remote()