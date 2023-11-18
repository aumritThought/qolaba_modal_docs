from modal import Image, Stub, method


def download_models():
    import torch, os, sys
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
    from PIL import Image
    os.chdir("../IP-Adapter")
    sys.path.insert(0, "../IP-Adapter")
    from ip_adapter import IPAdapterPlus


    os.system("git clone https://huggingface.co/h94/IP-Adapter")

    base_model_path = "Lykon/dreamshaper-8"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "IP-Adapter/models/image_encoder/"
    ip_ckpt = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
    device = "cuda"

    # load SDXL pipeline
    noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
    )
    pipe.enable_xformers_memory_efficient_attention()
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub("face_consistent_image2image")
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get install git-lfs",
        "git lfs install",
        "git clone https://github.com/tencent-ailab/IP-Adapter.git",
        "pip install diffusers==0.19.3 transformers accelerate safetensors torch xformers==0.0.22"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import time
        st= time.time()
        import torch, os, sys
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
        from PIL import Image
        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")
        from ip_adapter import IPAdapterPlus
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        os.system("git clone https://huggingface.co/h94/IP-Adapter")

        base_model_path = "Lykon/dreamshaper-8"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        image_encoder_path = "IP-Adapter/models/image_encoder/"
        ip_ckpt = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
        device = "cuda"

        # load SDXL pipeline
        noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
        )
        pipe.enable_xformers_memory_efficient_attention()
        self.ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor_safety = CLIPImageProcessor()  
        self.container_execution_time=time.time()-st

    def generate_image_urls(self, image_data):
        import io, base64, requests
        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls=[]
        for im in range(0, len(image_data["images"])):
            filtered_image = io.BytesIO()
            if(image_data["Has_NSFW_Content"][im]):
                pass
            else:
                image_data["images"][im].save(filtered_image, "JPEG")
                myobj = {
                        "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                    }
                rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
                im_url=rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        return image_urls


    @method()
    def run_inference(self, img,prompt,height, width ,batch, negative_prompt):
        import random, torch
        import time
        import numpy as np
        from PIL import Image
        st=time.time()
        seed=random.sample(range(1, 1000000000), 1)[0]

        image = self.ip_model.generate(pil_image=img, num_samples=batch, num_inference_steps=40,height=height, width=width,prompt=prompt, seed=seed, negative_prompt=negative_prompt)
        torch.cuda.empty_cache()

        safety_checker_input = self.feature_extractor_safety(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 

        image_data = {"images" :  image, "Has_NSFW_Content" : has_nsfw_concept}
        image_urls =self.generate_image_urls(image_data)
        
        self.runtime=time.time()-st
        return {"result":image_urls,  
                "Has_NSFW_Content":has_nsfw_concept, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
