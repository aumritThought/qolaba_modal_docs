from modal import Image, Secret, Stub, web_endpoint

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

auth_scheme = HTTPBearer()

@stub.function(gpu="a10g",secret=Secret.from_name("API_UPSCALING_KEY"))
@web_endpoint(label="karlo", method="POST")
def image_upscale(file : UploadFile,
    decoder_num_inference_steps : Optional[int]= 30,
    super_res_num_inference_steps: Optional[int] = 7,
    decoder_guidance_scale: Optional[int] = 4,
    num_imgs: Optional[int] = 1,
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):

    import os
    from diffusers import UnCLIPImageVariationPipeline
    import torch
    import numpy
    from PIL import Image

    if api_key.credentials != os.environ["API_UPSCALING_KEY"]:
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:   
        list_img=[".jpg",".png"]
        if any([x in file.filename for x in list_img]):
            content=file.file.read()
            image = Image.open(io.BytesIO(content))
            image = numpy.array(image) 
                # Convert RGB to BGR 
            image = image[:, :, ::-1].copy() 
            pipe = UnCLIPImageVariationPipeline.from_pretrained(cache_path, torch_dtype=torch.float16)
            pipe = pipe.to('cuda')
            image = pipe([image]*num_imgs, decoder_num_inference_steps=decoder_num_inference_steps,
                    super_res_num_inference_steps=super_res_num_inference_steps,
                    decoder_guidance_scale=decoder_guidance_scale).images
            filtered_image = io.BytesIO()
            image[0].save(filtered_image, "JPEG")
            filtered_image.seek(0)
            return StreamingResponse(filtered_image, media_type="image/jpeg")
        else:
            return "invalid file"




