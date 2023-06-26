from pathlib import Path

from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import  Depends, HTTPException, status, Query
from typing import  Optional, Annotated
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer()

model_schema={
    "model_id":"prompthero/openjourney-v4",
    "memory":10240,
    "container_idle_timeout":60,
    "name":"openjourney_text2image",
    "gpu":"a10g"
}

def download_models():
    import os, sys
    sys.path.append("/AITemplate/examples/05_stable_diffusion/")
    os.chdir("/AITemplate/examples/05_stable_diffusion/")
    os.system("python3 scripts/download_pipeline.py --model prompthero/openjourney-v4")

    from scripts.compile_alt import compile_diffusers

    compile_diffusers("./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2")
    class stableDiffusion:   
        def __init__(self):
            import sys,os
            sys.path.append("/AITemplate/examples/05_stable_diffusion/")
            os.chdir("/AITemplate/examples/05_stable_diffusion/")
            from aitemplate.utils.import_path import import_parent
            if __name__ == "__main__":
                import_parent(filepath=__file__, level=1)
            from src.pipeline_stable_diffusion_ait_alt import StableDiffusionAITPipeline
            from diffusers import  EulerDiscreteScheduler

            hf_hub_or_path="prompthero/openjourney-v4"
            self.pipe = StableDiffusionAITPipeline(
                hf_hub_or_path=hf_hub_or_path,
                ckpt=None,
            )
            self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
                "prompthero/openjourney-v4", subfolder="scheduler"
            )
    
    a=stableDiffusion()
    
def download_models1():
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub(model_schema["name"])
image=Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_dockerfile_commands=[ 
                                "RUN apt-get update --fix-missing",
                                "RUN apt install -y python3 python3-dev python3-pip",
                                "RUN apt install python-is-python3",
                                "RUN apt-get -y install git",
                                "RUN git clone --recursive https://github.com/qolaba/AITemplate.git",
                                "WORKDIR /AITemplate/docker",
                                "RUN bash install/install_basic_dep.sh",
                                "RUN bash install/install_test_dep.sh",
                                "RUN bash install/install_doc_dep.sh",
                                "RUN pip3 install torch torchvision torchaudio",
                                "RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata",
                                "RUN bash install/install_detection_deps.sh",
                                # "RUN bash install/install_ait.sh",
                                "RUN pip3 install --upgrade diffusers transformers accelerate scipy click ftfy",
                                "WORKDIR /AITemplate/python",
                                "RUN pwd",
                                "RUN python3 setup.py bdist_wheel",
                                "RUN pip3 install dist/*.whl --force-reinstall", 
                                "RUN pip3 install numpy==1.22",
                                "WORKDIR /AITemplate/examples/05_stable_diffusion/",
                                
                               ]
).run_function(
        download_models,
        gpu="a10g"
    ).run_function(
        download_models1,
    )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"])
class stableDiffusion:   
    def __enter__(self):
        import sys,os
        sys.path.append("/AITemplate/examples/05_stable_diffusion/")
        os.chdir("/AITemplate/examples/05_stable_diffusion/")
        from aitemplate.utils.import_path import import_parent
        if __name__ == "__main__":
            import_parent(filepath=__file__, level=1)
        from src.pipeline_stable_diffusion_ait_alt import StableDiffusionAITPipeline
        from diffusers import  EulerDiscreteScheduler
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor

        hf_hub_or_path=model_schema["model_id"]
        self.pipe = StableDiffusionAITPipeline(
            hf_hub_or_path=hf_hub_or_path,
            ckpt=None,
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
            hf_hub_or_path, subfolder="scheduler"
        )
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPFeatureExtractor()

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):

        import torch
        import numpy as np
        from PIL import Image
        prompt=prompt+", mdjrny-v4 style"
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        with torch.autocast("cuda"):
            image = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images

        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        return image
