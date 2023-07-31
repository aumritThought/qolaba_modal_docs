from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
    import torch

    controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "metobabba/epiCRealism_V3.0_inpainting", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub("controlnet_outpainting"+"_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        
        controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "metobabba/epiCRealism_V3.0_inpainting", controlnet=controlnet, torch_dtype=torch.float16
        )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()



    @method()
    def run_inference(self, img, batch,  right, left, top, bottom):
        import numpy as np
        from PIL import Image
        import torch

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
            extra_image = Image.new(image.mode, (image.size[0],image.size[1]), (0, 0, 0))
            result.paste(image, (left, top))
            mask_image.paste(extra_image, (left, top))
            return result, mask_image

        init_image, mask_image=maskimage(img,  right, left, top, bottom)
        init_image=init_image.resize((64 * round(init_image.size[0] / 64), 64 * round(init_image.size[1] / 64)))
        if(init_image.size[0]>1024 or init_image.size[0]<256 or init_image.size[1]>1024 or init_image.size[1]<256):
            init_image=init_image.resize((768, int(init_image.size[1]/init_image.size[0])*768))
            mask_image=mask_image.resize((768, int(mask_image.size[1]/mask_image.size[0])*768))
        control_image = make_inpaint_condition(init_image, mask_image)

        # generate image
        image = self.pipe(
            "",
            num_inference_steps=20,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            guess_mode=True,
            num_images_per_prompt=batch
        ).images

        
        return {"images":image,  "Has_NSFW_Content":[False]*batch}
