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
    os.system("python3 scripts/download_pipeline.py --model Lykon/DreamShaper")

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

            hf_hub_or_path="Lykon/DreamShaper"
            self.pipe = StableDiffusionAITPipeline(
                hf_hub_or_path=hf_hub_or_path,
                ckpt=None,
            )
            self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
                "Lykon/DreamShaper", subfolder="scheduler"
            )
    
    a=stableDiffusion()

stub = Stub("DreamShaper")
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

@stub.cls(gpu="a10g", container_idle_timeout=1200, memory=10240)
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

        hf_hub_or_path="Lykon/DreamShaper"
        self.pipe = StableDiffusionAITPipeline(
            hf_hub_or_path=hf_hub_or_path,
            ckpt=None,
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
            "Lykon/DreamShaper", subfolder="scheduler"
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


@stub.function(secrets=(Secret.from_name("API_UPSCALING_KEY"), Secret.from_name("cloudinary_url")))
@web_endpoint(label="dreamshaper", method="POST")
def get_image(
    
    height: int = Query(default=512,ge=256, le=1024),
    width: int = Query(default=512,ge=256, le=1024),
    num_inference_steps: int = Query(default=50,ge=10, le=100),
    guidance_scale:  float = Query(default=7.5,ge=2, le=25),
    batch:  int = Query(default=1,ge=1, le=4),
    prompt: Optional[str] = "Cute dog",
    negative_prompt: Optional[str] = "ugly",
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):

    import io,os
    import requests
    import base64, json

    if api_key.credentials != os.environ["API_UPSCALING_KEY"]:
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        image = stableDiffusion().run_inference.call(prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch)
        url = os.environ["cloudinary_url"]
        image_urls=[]
        for im in image:
            filtered_image = io.BytesIO()
            im.save(filtered_image, "JPEG")
            myobj = {
                    "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
            }
            rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
            im_url=rps.json()["data"]["secure_url"]
            image_urls.append(im_url)
        return {"urls":image_urls}
