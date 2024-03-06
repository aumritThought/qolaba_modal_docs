from modal import Image, Stub, method

model_schema={
    "memory":10240,
    "container_idle_timeout":600,
    "name":"stable_diffusion_video",
    "gpu":"a100"
}

def download_models():
    import torch
    from stable_diffusion_videos import StableDiffusionWalkPipeline

    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "stablediffusionapi/revanimated",
        torch_dtype=torch.float16,
    ).to("cuda")


stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "pip install diffusers --upgrade",
        "pip install invisible_watermark transformers accelerate safetensors xformers omegaconf stable_diffusion_videos",
        ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"], timeout=9000)
class stableDiffusion:   
    def __enter__(self):
        from stable_diffusion_videos import StableDiffusionWalkPipeline
        import torch

        self.pipe = StableDiffusionWalkPipeline.from_pretrained(
            "stablediffusionapi/revanimated",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()   
        

    @method()
    def run_inference(self,List_of_prompts_sd,height,width,guidance_scale,negative_prompt,fps, Num_of_sec):
        import requests, base64, random, shutil
        video_path = self.pipe.walk(
                prompts=List_of_prompts_sd,
                seeds=random.sample(range(1, 1000000000), len(List_of_prompts_sd)),
                num_interpolation_steps=int(fps*Num_of_sec/(len(List_of_prompts_sd)-1)),
                fps=fps,
                height=height, 
                width=width,   
                output_dir='dreams',      
                guidance_scale=guidance_scale,        
                num_inference_steps=45,  
                negative_prompt=negative_prompt   
            )
        with open(video_path, "rb") as f:
            url="https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
            byte=f.read()
            myobj = {"image":"data:audio/mpeg;base64,"+(base64.b64encode(byte).decode("utf8"))}
            
            rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
            video_url=rps.json()["data"]["secure_url"]
        

        shutil.rmtree("dreams")

        return {"video_url":video_url}
