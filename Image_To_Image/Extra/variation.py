from modal import Image, Stub, method


def download_models():
    import os, sys, torch
    from diffusers import StableDiffusionXLPipeline

    os.chdir("../IP-Adapter")
    sys.path.insert(0, "../IP-Adapter")
    os.system("git clone https://huggingface.co/h94/IP-Adapter")
    print(os.getcwd())
    from ip_adapter import IPAdapterXL
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False, use_safetensors=True, variant="fp16"
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    device = "cuda"
    image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
    ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

stub = Stub("variation_image2image")
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
        import os, sys, torch
        from diffusers import StableDiffusionXLPipeline
        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")
        from ip_adapter import IPAdapterXL
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"  
        device = "cuda"

        # load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False, use_safetensors=True, variant="fp16"
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
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
    def run_inference(self, img,prompt,guidance_scale,batch, negative_prompt):
        import time
        st=time.time()
        import random, torch
        import numpy as np
        from PIL import Image
        height=img.size[1]
        width=img.size[0]
        seed=random.sample(range(1, 1000000000), 1)[0]
        torch.cuda.empty_cache()
        image = self.ip_model.generate(pil_image=img,prompt=prompt, num_samples=batch, num_inference_steps=30, height=height, width=width, seed=seed, scale=0.5, negative_prompt=negative_prompt)
        torch.cuda.empty_cache()

        safety_checker_input = self.feature_extractor_safety(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        torch.cuda.empty_cache()
        image_data = {"images" : image, "Has_NSFW_Content" : has_nsfw_concept}
        image_urls =self.generate_image_urls(image_data)
        
        self.runtime=time.time()-st
        return {"result":image_urls,  
                "Has_NSFW_Content":  [False]*batch, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
