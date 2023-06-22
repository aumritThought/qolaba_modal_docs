from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import Depends, HTTPException, UploadFile, status, Request
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

stub = Stub("karlo-image-variation")
cache_path = "/vol/cache"


def download_models():
    from diffusers import UnCLIPImageVariationPipeline
    import torch

    pipe = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16, cache_dir=cache_path)

    pipe.save_pretrained(cache_path, safe_serialization=True)

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "diffusers", "transformers", "accelerate", "safetensors"
    )
    .run_function(
        download_models,
    )
)
stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:   
    def __enter__(self):
        from diffusers import UnCLIPImageVariationPipeline
        import torch
        self.pipe = UnCLIPImageVariationPipeline.from_pretrained(cache_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to('cuda')


    @method()
    def run_inference(self,file, decoder_num_inference_steps, super_res_num_inference_steps, decoder_guidance_scale, num_imgs):
        import numpy
        from PIL import Image
        list_img=[".jpg",".png"]
        if any([x in file.filename for x in list_img]):
            content=file.file.read()
            image = Image.open(io.BytesIO(content))
            image = numpy.array(image) 
                # Convert RGB to BGR 
            image = image[:, :, ::-1].copy() 

            image = self.pipe([image]*num_imgs, decoder_num_inference_steps=decoder_num_inference_steps,
                    super_res_num_inference_steps=super_res_num_inference_steps,
                    decoder_guidance_scale=decoder_guidance_scale).images
            return image
        else:
            return "invalid file"




