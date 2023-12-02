from modal import Image, Stub, method


def download_models():
    import torch

    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()


stub = Stub("depth"+"_controlnet_"+"_image2image")
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers"
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
        import torch
        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor

        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        # self.feature_extractor_safety = CLIPFeatureExtractor()  
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
        import torch
        import cv2, time
        from PIL import Image
        import numpy as np
        st=time.time()

        def get_depth_map(image):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad(), torch.autocast("cuda"):
                depth_map = self.depth_estimator(image).predicted_depth

            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(1024, 1024),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            image = torch.cat([depth_map] * 3, dim=1)

            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
            return image
        
        
        
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        
        image = get_depth_map(file_url)
        image = image.resize((file_url.size))
        image = self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=guidance_scale, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.5).images
        
        torch.cuda.empty_cache()

        # safety_checker_input = self.feature_extractor_safety(
        #         image, return_tensors="pt"
        #     ).to("cuda")
        # image, has_nsfw_concept = self.safety_checker(
        #                 images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
        #             )
        # image=[ Image.fromarray(np.uint8(i)) for i in image] 

        image_data = {"images" :  image, "Has_NSFW_Content" : [False]*batch}
        image_urls =self.generate_image_urls(image_data)
        self.runtime=time.time()-st

        return {"result":image_urls,  
                "Has_NSFW_Content":[False]*batch, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
