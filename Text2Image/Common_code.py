from modal import Image, Stub, method


def get_schema():
    model_schema={
    "memory":10240,
    "container_idle_timeout":600,
    "gpu":"a10g"
    }
    return model_schema

def download_models_(model_name):
    from diffusers import StableDiffusionXLPipeline
    import torch
    pipe = StableDiffusionXLPipeline.from_single_file(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()


class stableDiffusion_:   
    def __init__(self, model_name):
        import time
        st= time.time()
        from diffusers import StableDiffusionXLPipeline
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        

        self.pipe = StableDiffusionXLPipeline.from_single_file(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()     
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPFeatureExtractor()   
        self.container_execution_time=time.time()-st

    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):
        import torch, io, requests, base64
        import numpy as np
        from PIL import Image
        import time

        st=time.time()
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        image = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images
        torch.cuda.empty_cache()

        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        
        

        url = "https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls=[]
        for im in range(0, len(image)):
            filtered_image = io.BytesIO()
            if(has_nsfw_concept[im]):
                pass
            else:
                image[im].save(filtered_image, "JPEG")
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