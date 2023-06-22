from pathlib import Path

from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import  Depends, HTTPException, status, Query
from typing import  Optional, Annotated
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer()


def download_models():
    from kandinsky2 import get_kandinsky2
    model = get_kandinsky2('cuda', task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)

stub = Stub("kandinsky")
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["git"])
    .run_commands([
        "pip install 'git+https://github.com/ai-forever/Kandinsky-2.git'",
        "pip install git+https://github.com/openai/CLIP.git",
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"
                   ])
    .pip_install("opencv-python")
).run_function(
        download_models,
        gpu="a10g"
    )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:   
    def __enter__(self):
        from kandinsky2 import get_kandinsky2
        self.pipe = get_kandinsky2('cuda', task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):

        images = self.pipe.generate_text2img(prompt, num_steps=num_inference_steps,
                                batch_size=batch, guidance_scale=guidance_scale,
                                h=height, w=width, negative_prior_prompt=negative_prompt,sampler='p_sampler', prior_cf_scale=4,
                                    prior_steps="5")


        return images
