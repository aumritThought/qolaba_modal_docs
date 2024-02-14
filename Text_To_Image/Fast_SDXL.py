from modal import Image, Stub, method
from Common_code import *

model_schema= get_schema()
model_schema["name"] = "Fast_SDXL_text2image"
model_schema["model_id"] = "../MexxL_LCM2.safetensors"

# def download_models():
#     import torch
#     from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
    
#     pipe = StableDiffusionXLPipeline.from_single_file("../turbovision.safetensors", torch_dtype=torch.float16, variant="fp16")
#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.to("cuda")
#     refiner = DiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=pipe.text_encoder_2,
#             vae=pipe.vae,
#             torch_dtype=torch.float16,
#             use_safetensors=True,
#             variant="fp16",
#         ).to("cuda")

#     from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
#     from transformers import CLIPFeatureExtractor
#     safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
#     feature_extractor = CLIPFeatureExtractor()
def download_models():
    download_models_(model_schema["model_id"])

stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "pip install diffusers --upgrade",
        "pip install invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf",
        "wget https://civitai.com/api/download/models/292185",
        "mv 292185 MexxL_LCM2.safetensors"
        ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu="a10g", memory=model_schema["memory"], container_idle_timeout=200)
class stableDiffusion:   
    def __init__(self):
        import time
        st= time.time()
        from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        

        self.pipe = StableDiffusionXLPipeline.from_single_file(model_schema["model_id"], torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

        self.pipe.to("cuda")
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()   
        self.refiner.enable_xformers_memory_efficient_attention()    
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPImageProcessor()   
        self.container_execution_time=time.time()-st

    def generate_image_urls(self, image):
        import io, base64, requests
        import numpy as np

        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=image, clip_input=safety_checker_input.pixel_values
                    )
        # image=[ Image.fromarray(np.uint8(i)) for i in image] 

        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"

        filtered_image = io.BytesIO()
        if(has_nsfw_concept[0]):
            pass
        else:
            image.save(filtered_image, "JPEG")
            myobj = {
                        "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                    }
            rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
            im_url=rps.json()["data"]["secure_url"]
        return [im_url, has_nsfw_concept[0]]
    

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):
        import torch, io, requests, base64
        import numpy as np
        from PIL import Image
        import time, threading
        
        st=time.time()
        torch.cuda.empty_cache()
        threads=[]
        results = []
        
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
                ).images[0]
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                denoising_start=0.8,
                image=image,
            ).images[0]
            torch.cuda.empty_cache()


            thread = threading.Thread(target=lambda: results.append(self.generate_image_urls(image)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        image_urls=[]
        has_nsfw_concept=[]
        for i, j in results:
            image_urls.append(i)
            has_nsfw_concept.append(j)


        
        self.runtime=time.time()-st
        return {"result":image_urls,  
                "Has_NSFW_Content":has_nsfw_concept, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}