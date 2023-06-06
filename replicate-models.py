from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import Depends, HTTPException, UploadFile, status, Request
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Union

stub = Stub("karlo-image-variation")
cache_path = "/vol/cache"


def download_models():
    from diffusers import StableDiffusionPipeline
    import torch
    from torch import inference_mode
    import deepspeed

    print(torch.cuda.is_available())
    model_id = "nitrosocke/archer-diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # with torch.inference_mode():
    #     deepspeed.init_inference(
    #     model=getattr(pipe,"model", pipe),      # Transformers models
    #     # mp_size=1,        # Number of GPU
    #     dtype=torch.float16, # dtype of the weights (fp16)
    #     # replace_method="auto", # Lets DS autmatically identify the layer to replace
    #     replace_with_kernel_inject=True, # replace the model with the kernel injector
    # )
        
    model_id = "Lykon/DreamShaper"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")

    # with torch.inference_mode():
    #     deepspeed.init_inference(
    #     model=getattr(pipe,"model", pipe),      # Transformers models
    #     # mp_size=1,        # Number of GPU
    #     dtype=torch.float16, # dtype of the weights (fp16)
    #     # replace_method="auto", # Lets DS autmatically identify the layer to replace
    #     replace_with_kernel_inject=True, # replace the model with the kernel injector
    # )
    

image=Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_dockerfile_commands=[ "RUN apt-get update --fix-missing",
                                "RUN apt-get install -y python3-pip",
                                "RUN apt install python-is-python3",
                                "RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117",
                                "RUN apt-get -y install git",
                                "RUN pip install git+https://github.com/microsoft/deepspeed@3a2dc40d54489b176981cf24c7c1f296c8fc5d30 ",
                                "RUN pip install --upgrade diffusers==0.11.0 transformers==4.25.1 safetensors scipy triton==2.0.0.dev20221031 accelerate ftfy tensorflow",
                                "RUN pip install tensorrt onnxruntime onnxruntime-gpu tf2onnx"
                               ]
).run_function(
        download_models,
        gpu="a10g"
    )

stub.image = image

auth_scheme = HTTPBearer()

@stub.cls(gpu="a10g")
class archerDiffusion:
    def __enter__(self):
        from diffusers import StableDiffusionPipeline
        import torch
        import deepspeed

        model_id = "nitrosocke/archer-diffusion"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

        with torch.inference_mode():
            x= deepspeed.init_inference(
            model=getattr(self.pipe,"model", self.pipe),      # Transformers models
            # mp_size=1,        # Number of GPU
            dtype=torch.float16, # dtype of the weights (fp16)
            # replace_method="auto", # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=True, # replace the model with the kernel injector
        )

    @method()
    def run_inference(
        self,
        prompt: Optional[str] = "dog",
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[str] = "extra-hands",
        batch_size: Optional[int] = 1):

        image = self.pipe(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size).images

        return image



@stub.function(secret=Secret.from_name("API_UPSCALING_KEY"))
@web_endpoint(label="replicate-models", method="POST")
def archer(
    prompt: Optional[str] = "dog",
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[str] = "extra-hands",
    batch_size: Optional[int] = 1,
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):

    import os, io

    if api_key.credentials != os.environ["API_UPSCALING_KEY"]:
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:  
        sd = archerDiffusion()
        image = sd.run_inference.call(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size)
        filtered_image = io.BytesIO()
        image[0].save(filtered_image, "JPEG")
        filtered_image.seek(0)
        return StreamingResponse(filtered_image, media_type="image/jpeg")
    

# @stub.cls(gpu="a10g")
# class dreamshaper:
#     def __enter__(self):
#         from diffusers import StableDiffusionPipeline
#         import torch
#         import deepspeed

#         model_id = "Lykon/DreamShaper"
#         self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#         self.pipe = self.pipe.to("cuda")

#         with torch.inference_mode():
#             deepspeed.init_inference(
#             model=getattr(self.pipe,"model", self.pipe),      # Transformers models
#             # mp_size=1,        # Number of GPU
#             dtype=torch.float16, # dtype of the weights (fp16)
#             # replace_method="auto", # Lets DS autmatically identify the layer to replace
#             replace_with_kernel_inject=True, # replace the model with the kernel injector
#         )

#     @method()
#     def run_inference(
#         self,
#         prompt: Optional[str] = "dog",
#         height: Optional[int] = 512,
#         width: Optional[int] = 512,
#         num_inference_steps: Optional[int] = 50,
#         guidance_scale: Optional[float] = 7.5,
#         negative_prompt: Optional[str] = "extra-hands",
#         batch_size: Optional[int] = 1):

#         image = self.pipe(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size).images

#         return image



# @stub.function(secret=Secret.from_name("API_UPSCALING_KEY"))
# @web_endpoint(label="dreamshaper", method="POST")
# def dreamshaperf(
#     prompt: Optional[str] = "dog",
#     height: Optional[int] = 512,
#     width: Optional[int] = 512,
#     num_inference_steps: Optional[int] = 50,
#     guidance_scale: Optional[float] = 7.5,
#     negative_prompt: Optional[str] = "extra-hands",
#     batch_size: Optional[int] = 1,
#     api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):

#     import os, io

#     if api_key.credentials != os.environ["API_UPSCALING_KEY"]:
#         raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Incorrect bearer token",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
#     else:  
#         sd = dreamshaper()
#         image = sd.run_inference.call(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch_size)
#         filtered_image = io.BytesIO()
#         image[0].save(filtered_image, "JPEG")
#         filtered_image.seek(0)
#         return StreamingResponse(filtered_image, media_type="image/jpeg")
