from modal import Stub, Volume, Secret
from src.utils.Constants import *
from src.utils.Globals import get_base_image
import os
from diffusers import DiffusionPipeline
import torch
from diffusers import T2IAdapter



stub = Stub("volume-stub", secrets=[Secret.from_name("environment_configuration")])

image = get_base_image()

vol = Volume.persisted(VOLUME_NAME)

@stub.function(volumes={VOLUME_PATH: vol} ,  image = image, timeout = 72000)
def download_models():
    #3D Cartoon model
    os.system(f"wget -O {SDXL_3DCARTOON_MODEL} https://civitai.com/api/download/models/297740")

    #Cartoon model
    os.system(f"wget -O {SDXL_CARTOON_MODEL} https://civitai.com/api/download/models/365017")

    #Pixela Model
    os.system(f"wget -O {SDXL_PIXELA_MODEL} https://civitai.com/api/download/models/265938")

    #Colorful model
    os.system(f"wget -O {SDXL_COLORFUL_MODEL} https://civitai.com/api/download/models/182077")

    #rev-anime model
    os.system(f"wget -O {SDXL_REVANIME_MODEL} https://civitai.com/api/download/models/291443")

    #Realistic Model
    os.system(f"wget -O {SDXL_REALISTIC_MODEL} https://civitai.com/api/download/models/312982")

    #Realistic 2 model
    os.system(f"wget -O {SDXL_REALISTIC_2_MODEL} https://civitai.com/api/download/models/293240")

    #SDXL Turbo model
    os.system(f"wget -O {SDXL_TURBO_MODEL} https://civitai.com/api/download/models/273102")

    #Anime model
    os.system(f"wget -O {SDXL_ANIME_MODEL} https://civitai.com/api/download/models/133163")

    # #Anime 2 model
    os.system(f"wget -O {SDXL_ANIME_2_MODEL} https://civitai.com/api/download/models/293564")

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

    T2IAdapter.from_pretrained(
        CANNY_MODEL, torch_dtype=torch.float16
    ).save_pretrained(CANNY_PATH)

    T2IAdapter.from_pretrained(
        DEPTH_MODEL, torch_dtype=torch.float16
    ).save_pretrained(DEPTH_PATH)

    vol.commit()



@stub.local_entrypoint()
def run():
    download_models.remote()