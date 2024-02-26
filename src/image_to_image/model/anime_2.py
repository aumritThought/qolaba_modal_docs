from modal import Image, Stub, method
from Common_code import *

model_schema = get_schema()
model_schema["name"] = "anime_2_image2image"
model_schema["model_id"] = "../AstreaPixie.safetensors"


def download_models():
    download_models_(model_schema["model_id"])


stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
            "apt-get update && apt-get install wget -y",
            "wget https://civitai.com/api/download/models/133163",
            "pip install diffusers --upgrade",
            "pip install invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf",
            "mv 133163 AstreaPixie.safetensors",
        ]
    )
).run_function(download_models, gpu="t4")

stub.image = image


@stub.cls(
    gpu=model_schema["gpu"],
    memory=model_schema["memory"],
    container_idle_timeout=model_schema["container_idle_timeout"],
)
class stableDiffusion:
    def __enter__(self):
        self.generator = stableDiffusion_(model_schema["model_id"])

    @method()
    def run_inference(
        self, file_url, prompt, guidance_scale, negative_prompt, batch, strength
    ):
        return self.generator.run_inference(
            file_url, prompt, guidance_scale, negative_prompt, batch, strength
        )
