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
    import os, sys
    sys.path.append("/AITemplate/examples/05_stable_diffusion/")
    os.chdir("/AITemplate/examples/05_stable_diffusion/")
    os.system("python3 scripts/download_pipeline.py --model theintuitiveye/HARDblend")

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

            hf_hub_or_path="theintuitiveye/HARDblend"
            self.pipe = StableDiffusionAITPipeline(
                hf_hub_or_path=hf_hub_or_path,
                ckpt=None,
            )
            self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
                "theintuitiveye/HARDblend", subfolder="scheduler"
            )
    
    a=stableDiffusion()

stub = Stub("HARDblend")
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
    )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
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

        hf_hub_or_path="theintuitiveye/HARDblend"
        self.pipe = StableDiffusionAITPipeline(
            hf_hub_or_path=hf_hub_or_path,
            ckpt=None,
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
            "theintuitiveye/HARDblend", subfolder="scheduler"
        )

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):

        import torch 
        
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


        return image
