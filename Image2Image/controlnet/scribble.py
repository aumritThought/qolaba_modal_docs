from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import torch
    from controlnet_aux import HEDdetector

    hed = HEDdetector.from_pretrained('lllyasviel/Annotators')


    controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
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

stub = Stub("scribble"+"_controlnet_"+"_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers controlnet_aux matplotlib mediapipe"
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
        from controlnet_aux import HEDdetector

        self.hed = HEDdetector.from_pretrained('lllyasviel/Annotators')

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stablediffusionapi/rev-anim", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()


    @method()
    def run_inference(self, img, prompt,guidance_scale,negative_prompt, batch, strength):

        
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        
        image = self.hed(img, scribble=True)
        image = image.resize((img.size[0], img.size[1]))
        image = self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=guidance_scale, negative_prompt=negative_prompt).images

        return {"images":image,  "Has_NSFW_Content":[False]*batch}
