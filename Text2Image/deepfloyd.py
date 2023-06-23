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
    from diffusers import DiffusionPipeline
    import torch
    from huggingface_hub import login

    login("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")

    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    stage_1.enable_model_cpu_offload()

    stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
    stage_2.enable_model_cpu_offload()

stub = Stub("deepfloyd")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "pip install diffusers transformers accelerate scipy safetensors sentencepiece"
                   ])
).run_function(
        download_models,
        gpu="a10g"
    )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600)
class stableDiffusion:   
    def __enter__(self):
        from diffusers import DiffusionPipeline
        import torch

        self.stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        self.stage_1.enable_model_cpu_offload()

        self.stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
        self.stage_2.enable_model_cpu_offload()

    @method()
    def run_inference(self,prompt,ratio,hrns,imns,guidance_scale,negative_prompt,num_out):
        from diffusers.utils import pt_to_pil

        if ratio == "1:1":
            height, width = 64, 64
        elif ratio == "3:2":
            height, width = 96, 64
        elif ratio == "2:1":
            height, width = 64, 32
        elif ratio == "2:3":
            height, width = 64,96
        elif ratio == "1:2":
            height, width = 32, 64
        else :
            height, width = 64, 64
            

        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt=[prompt]*num_out, negative_prompt=[negative_prompt]*num_out)
        
        image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", num_inference_steps=imns, guidance_scale=guidance_scale, height=height, width=width).images

        image = self.stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", num_inference_steps=hrns, guidance_scale=guidance_scale).images
        image=pt_to_pil(image)    
            
        for i in range(0,len(image)):
            image[i]=image[i].resize((width*4,height*4))
        
        return image

