from modal import Image, Stub, method

model_schema={
    "memory":10240,
    "container_idle_timeout":600,
    "gpu":"a10g"
}
model_schema["name"]="cosmic-babes_image2image"

stub = Stub(model_schema["name"])

def download_models():
    from diffusers import StableDiffusionXLImg2ImgPipeline
    import torch
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("nyxia/dynavision-xl", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()


image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "pip install diffusers --upgrade",
        "pip install invisible_watermark transformers accelerate safetensors xformers omegaconf",
        ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"])
class stableDiffusion():
    def __enter__(self) -> None:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor

        self.image_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("nyxia/dynavision-xl", torch_dtype=torch.float16)
        self.image_pipe.to("cuda")

        self.image_pipe.enable_xformers_memory_efficient_attention()     
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPFeatureExtractor()   

    @method()
    def run_inference(self,img, prompt,guidance_scale,negative_prompt,batch, strength):

        import torch
        import numpy as np
        from PIL import Image
        prompt = [prompt] * batch
        negative_prompt=[negative_prompt]*batch
        image = self.image_pipe(
                    prompt=prompt, image=img, strength=strength, guidance_scale=guidance_scale, negative_prompt=negative_prompt
                ).images
        torch.cuda.empty_cache()

        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        return {"images":image,  "Has_NSFW_Content":has_nsfw_concept}