from modal import Image, Stub, method

model_schema={
    "model_id":"stablediffusionapi/all-526",
    "memory":10240,
    "container_idle_timeout":600,
    "name":"all-526_text2image",
    "gpu":"a10g"
}
def download_models():
    from diffusers import StableDiffusionXLPipeline
    import torch
    pipe = StableDiffusionXLPipeline.from_single_file("../mbbxl.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()


stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "wget https://civitai.com/api/download/models/130312",
        "pip install diffusers --upgrade",
        "pip install invisible_watermark transformers accelerate safetensors xformers omegaconf",
        "mv 130312 mbbxl.safetensors",
        ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"])
class stableDiffusion:   
    def __enter__(self):
        from diffusers import StableDiffusionXLPipeline
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor

        self.pipe = StableDiffusionXLPipeline.from_single_file("../mbbxl.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()     
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPFeatureExtractor()   
        

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):
        import torch
        import numpy as np
        from PIL import Image

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
        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        return {"images":image,  "Has_NSFW_Content":has_nsfw_concept}
