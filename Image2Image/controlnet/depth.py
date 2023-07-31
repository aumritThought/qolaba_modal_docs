from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import torch
    from transformers import pipeline

    depth_estimator = pipeline('depth-estimation')

    controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stablediffusionapi/rev-anim", controlnet=controlnet, torch_dtype=torch.float16
        )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub("depth"+"_controlnet_"+"_image2image")
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

@stub.cls(gpu="a10g", container_idle_timeout=100, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        from transformers import pipeline

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stablediffusionapi/rev-anim", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

        self.depth_estimator = pipeline('depth-estimation')


    @method()
    def run_inference(self, img, prompt,guidance_scale,negative_prompt, batch, strength):
        import cv2
        from PIL import Image
        import numpy as np
        
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        
        image = self.depth_estimator(img)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        image = self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=guidance_scale, negative_prompt=negative_prompt).images

        return {"images":image,  "Has_NSFW_Content":[False]*batch}
