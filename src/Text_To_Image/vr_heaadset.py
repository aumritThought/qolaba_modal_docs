from modal import Image, Stub, method
from Common_code import *

model_schema= get_schema()
model_schema["name"] = "vr_headset_text2image"

def download_models():
    from huggingface_hub import login
    login("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")
    from diffusers import StableDiffusionXLPipeline, LCMScheduler, DiffusionPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
    pipe.load_lora_weights("Qolaba/Lora_Models",  weight_name="Lora_yogasanavectorart_QPTc0sa1.safetensors")
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.enable_xformers_memory_efficient_attention()

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git wget -y",
        "apt-get install git-lfs",
        "git lfs install",
        "git clone https://github.com/tencent-ailab/IP-Adapter.git",
        "pip install --upgrade diffusers transformers accelerate safetensors torch xformers onnxruntime einops insightface omegaconf",
        "wget https://civitai.com/api/download/models/182077",
        "mv 182077 Starlight.safetensors",
        ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", memory=model_schema["memory"], container_idle_timeout=200)
class stableDiffusion:   
    def __enter__(self):
        import time
        st= time.time()
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        from diffusers import StableDiffusionXLPipeline, LCMScheduler, DiffusionPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.pipe.load_lora_weights("Qolaba/Lora_Models",  weight_name="Lora_yogasanavectorart_QPTc0sa1.safetensors")

        self.pipe.to(device="cuda", dtype=torch.float16)
        self.pipe.enable_xformers_memory_efficient_attention()

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        self.refiner.enable_xformers_memory_efficient_attention()
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPImageProcessor()   
        self.container_execution_time=time.time()-st

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):
        import torch, io, requests, base64
        import numpy as np
        from PIL import Image
        import time

        st=time.time()
        torch.cuda.empty_cache()
        prompt="Blue_Striped_Blazer, "+prompt
        images=[]
        for i in range(0,batch):
            image = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_end=0.8,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                    cross_attention_kwargs={"scale": 0.8}
                ).images[0]
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_start=0.8,
                image=image,
            ).images[0]
            torch.cuda.empty_cache()
            images.append(image)

        safety_checker_input = self.feature_extractor(
                images, return_tensors="pt"
            ).to("cuda")
        images, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in images], clip_input=safety_checker_input.pixel_values
                    )
        images=[ Image.fromarray(np.uint8(i)) for i in images] 
        
        

        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls=[]
        for im in range(0, len(images)):
            filtered_image = io.BytesIO()
            if(has_nsfw_concept[im]):
                pass
            else:
                images[im].save(filtered_image, "JPEG")
                myobj = {
                        "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                    }
                rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
                im_url=rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        self.runtime=time.time()-st

        return {"result":image_urls,  
                "Has_NSFW_Content":has_nsfw_concept, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}