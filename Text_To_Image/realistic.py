from modal import Image, Stub, method
from Common_code import *

model_schema= get_schema()
model_schema["name"] = "realistic_text2image"
model_schema["model_id"] = "../XXMix_9realisticSDXL.safetensors"

def download_models():
    download_models_(model_schema["model_id"])

stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "wget https://civitai.com/api/download/models/163192",
        "pip install diffusers --upgrade",
        "pip install invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf",
        "mv 163192 XXMix_9realisticSDXL.safetensors",
        ])
    ).run_function(
            download_models,
            gpu="t4",
        )

stub.image = image

@stub.cls(gpu="a100", memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"],concurrency_limit=model_schema["concurrency_limit"] )
class stableDiffusion:   
    def __enter__(self):
        self.generator = stableDiffusion_(model_schema["model_id"])

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):
        return self.generator.run_inference(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch)
