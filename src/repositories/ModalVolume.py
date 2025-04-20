import os

import torch
from diffusers import DiffusionPipeline
from modal import App, Secret, Volume

from src.utils.Constants import *
from src.utils.Globals import get_base_image

app = App("volume-stub", secrets=[Secret.from_name("environment_configuration")])

image = get_base_image()

vol = Volume.from_name(VOLUME_NAME)


@app.function(volumes={VOLUME_PATH: vol}, image=image, timeout=72000)
def download_models():
    if not os.path.isfile(SDXL_3DCARTOON_MODEL):
        # 3D Cartoon model
        os.system(
            f"wget -O {SDXL_3DCARTOON_MODEL} https://civitai.com/api/download/models/297740"
        )
        vol.commit()

    if not os.path.isfile(SDXL_CARTOON_MODEL):
        # Cartoon model
        os.system(
            f"wget -O {SDXL_CARTOON_MODEL} https://civitai.com/api/download/models/310711"
        )
        vol.commit()

    if not os.path.isfile(SDXL_PIXELA_MODEL):
        # Pixela Model
        os.system(
            f"wget -O {SDXL_PIXELA_MODEL} https://civitai.com/api/download/models/265938"
        )
        vol.commit()

    if not os.path.isfile(SDXL_COLORFUL_MODEL):
        # Colorful model
        os.system(
            f"wget -O {SDXL_COLORFUL_MODEL} https://civitai.com/api/download/models/182077"
        )
        vol.commit()

    if not os.path.isfile(SDXL_REVANIME_MODEL):
        # rev-anime model
        os.system(
            f"wget -O {SDXL_REVANIME_MODEL} https://civitai.com/api/download/models/291443"
        )
        vol.commit()

    if not os.path.isfile(SDXL_REALISTIC_MODEL):
        # Realistic Model
        os.system(
            f"wget -O {SDXL_REALISTIC_MODEL} https://civitai.com/api/download/models/312982"
        )
        vol.commit()

    if not os.path.isfile(SDXL_REALISTIC_2_MODEL):
        # Realistic 2 model
        os.system(
            f"wget -O {SDXL_REALISTIC_2_MODEL} https://civitai.com/api/download/models/293240"
        )
        vol.commit()

    if not os.path.isfile(SDXL_TURBO_MODEL):
        # SDXL Turbo model
        os.system(
            f"wget -O {SDXL_TURBO_MODEL} https://civitai.com/api/download/models/273102"
        )
        vol.commit()

    if not os.path.isfile(SDXL_ANIME_MODEL):
        # Anime model
        os.system(
            f"wget -O {SDXL_ANIME_MODEL} https://civitai.com/api/download/models/133163"
        )
        vol.commit()

    if not os.path.isfile(SDXL_ANIME_2_MODEL):
        # #Anime 2 model
        os.system(
            f"wget -O {SDXL_ANIME_2_MODEL} https://civitai.com/api/download/models/293564"
        )
        vol.commit()

    if not os.path.isdir(SDXL_REFINER_MODEL_PATH):
        os.mkdir(SDXL_REFINER_MODEL_PATH)

        refiner = DiffusionPipeline.from_pretrained(
            SDXL_REFINER_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.save_pretrained(SDXL_REFINER_MODEL_PATH)
        vol.commit()

    if not os.path.isdir(OPENPOSE_PATH):
        os.system(
            f"git clone https://huggingface.co/TencentARC/t2i-adapter-openpose-sdxl-1.0 {OPENPOSE_PATH}"
        )
        vol.commit()

    if not os.path.isdir(SKETCH_PATH):
        os.system(
            f"git clone https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0 {SKETCH_PATH}"
        )
        vol.commit()

    if not os.path.isdir(DEPTH_PATH):
        os.system(
            f"git clone https://huggingface.co/TencentARC/t2i-adapter-depth-midas-sdxl-1.0 {DEPTH_PATH}"
        )
        vol.commit()

    if not os.path.isdir(CANNY_PATH):
        os.system(
            f"git clone https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0 {CANNY_PATH}"
        )
        vol.commit()

    if not os.path.isdir(ULTRASHARP_MODEL):
        os.system(
            f"wget -O {ULTRASHARP_MODEL} https://huggingface.co/lokCX/4x-Ultrasharp/resolve/1856559b50de25116a7c07261177dd128f1f5664/4x-UltraSharp.pth"
        )
        vol.commit()

    vol.commit()


@app.local_entrypoint()
def run():
    download_models.remote()
