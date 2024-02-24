from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, StableDiffusionControlNetImg2ImgPipeline
    import torch
    BASE_MODEL = "Lykon/dreamshaper-8"

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    #init_pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)#, torch_dtype=torch.float16)
    main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to("cuda")
    image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)


stub = Stub("Illusion_Diffusion_image2image")
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers==0.0.22"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import time
        st= time.time()
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, StableDiffusionControlNetImg2ImgPipeline
        import torch
        BASE_MODEL ="Lykon/dreamshaper-8"

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            vae=vae,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to("cuda")
        # self.pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        # self.feature_extractor = CLIPFeatureExtractor()  
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
    def run_inference(self, file_url, prompt,guidance_scale,negative_prompt, batch, strength):

        import cv2, time, torch
        from PIL import Image
        import numpy as np
        st=time.time()
        OldRange = 1  
        NewRange = (2.5 - 0.8)  
        strength = (((strength) * NewRange) / OldRange) + 0.8
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            # control_image=control_image_large,        
            # image=upscaled_latents,
            guidance_scale=float(guidance_scale),
            # generator=generator,
            num_inference_steps=20,
            image = file_url,
            controlnet_conditioning_scale=strength,

            # strength=0.5,
            # control_guidance_start=float(control_guidance_start),
            # control_guidance_end=float(control_guidance_end),
            # controlnet_conditioning_scale=float(controlnet_conditioning_scale)
        ).images

        torch.cuda.empty_cache()

        image_data = {"images" :  image, "Has_NSFW_Content" : [False]*batch}
        
        image_urls =self.generate_image_urls(image_data)
        self.runtime=time.time()-st

        return {"result":image_urls,  
                "Has_NSFW_Content":[False]*batch, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
