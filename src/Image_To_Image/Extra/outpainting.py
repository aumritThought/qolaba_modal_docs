from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
    import torch

    controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "uf/epicrealism_pureEvolutionV3", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub("outpainting"+"_image2image")
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers==0.0.22"
                   ])
    ).run_function(
            download_models,
            gpu="a10g",
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import time
        st= time.time()
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        
        controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "uf/epicrealism_pureEvolutionV3", controlnet=controlnet, torch_dtype=torch.float16
        )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
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
    def run_inference(self, file_url, batch,  right, left, top, bottom, prompt, negative_prompt): #add prompt and negative prompt
        import numpy as np
        from PIL import Image
        import torch
        import time
        st=time.time()
        def make_inpaint_condition(image, image_mask):
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image

        def maskimage(image, right, left, top, bottom):

            width, height = image.size

            new_width = width + right + left 
            new_height = height + top + bottom

            result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
            mask_image = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
            extra_image = Image.new(image.mode, (image.size[0]-64,image.size[1]-64), (0, 0, 0))
            result.paste(image, (left, top))
            mask_image.paste(extra_image, (left+32, top+32))
            return result, mask_image

        init_image, mask_image=maskimage(file_url,  right, left, top, bottom)
        init_image=init_image.resize((64 * round(init_image.size[0] / 64), 64 * round(init_image.size[1] / 64)))
        if(init_image.size[0]>1024 or init_image.size[0]<256 or init_image.size[1]>1024 or init_image.size[1]<256):
            if(init_image.size[1]>=init_image.size[0]):
                height=1024
                width=((int(init_image.size[0]*1024/init_image.size[1]))//64)*64
            else:
                width=1024
                height=((int(init_image.size[1]*1024/init_image.size[0]))//64)*64
            init_image=init_image.resize((width, height))
            mask_image=mask_image.resize((width, height))
        control_image = make_inpaint_condition(init_image, mask_image)

        # generate image
        image = self.pipe(
            prompt,
            num_inference_steps=20,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            guess_mode=True,
            num_images_per_prompt=batch, 
            negative_prompt=negative_prompt
        ).images
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
