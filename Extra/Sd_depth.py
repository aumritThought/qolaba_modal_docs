from modal import Image, Stub, method

def download_models():
    import torch
    from diffusers import StableDiffusionDepth2ImgPipeline

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
    ).to("cuda")


stub = Stub("SD_depth_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "pip install diffusers transformers accelerate opencv-python Pillow"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=60, memory=10240)
class stableDiffusion:   
    def __enter__(self):
        import torch
        from diffusers import StableDiffusionDepth2ImgPipeline

        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        torch_dtype=torch.float16,
        ).to("cuda")

    @method()
    def run_inference(self, img, prompt,guidance_scale,negative_prompt,batch, strength):
        img=[img]*batch
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        image = self.pipe(prompt=prompt, image=img, negative_prompt=negative_prompt, strength=strength, guidance_scale=guidance_scale).images

        return {"images":image,  "Has_NSFW_Content":[False]*batch}
