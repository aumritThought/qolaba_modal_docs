from modal import Image, Stub, method
from Common_code import *
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLPipeline, LCMScheduler, DiffusionPipeline
import torch

model_schema = get_schema()
model_schema["name"] = "kia_seltos_text2image"

checkpoint = "RealismEngineSDXL.safetensors"
lora_weights = "Lora_yogasanavectorart_QPTc0sa1.safetensors"
hf_lora_path = "Qolaba/Lora_Models"


def create_stable_diffusion_pipeline(
    checkpoint: str, lora_weights: str, hf_lora_path: str = "Qolaba/Lora_Models"
):
    pipe = StableDiffusionXLPipeline.from_single_file(
        f"../{checkpoint}",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.load_lora_weights(hf_lora_path, weight_name=lora_weights)
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def create_refiner(
    pipe: StableDiffusionXLPipeline,
    name: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
):
    refiner = DiffusionPipeline.from_pretrained(
        name,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    refiner.enable_xformers_memory_efficient_attention()
    return refiner


def download_models():
    from diffusers import StableDiffusionXLPipeline, LCMScheduler, DiffusionPipeline
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    import torch, os
    from huggingface_hub import login

    login("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")

    pipe = create_stable_diffusion_pipeline(checkpoint, lora_weights)

    DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )
    CLIPFeatureExtractor()


stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
            "apt-get update && apt-get install wget -y",
            "wget https://civitai.com/api/download/models/258380",
            "pip install diffusers --upgrade",
            "pip install invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf",
            f"mv 258380 {checkpoint}",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", memory=model_schema["memory"], container_idle_timeout=200)
class stableDiffusion:
    def __enter__(self):
        import time

        st = time.time()
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        from diffusers import StableDiffusionXLPipeline, LCMScheduler, DiffusionPipeline

        self.pipe = create_stable_diffusion_pipeline(checkpoint, lora_weights)
        self.refiner = create_refiner(self.pipe)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(
        self,
        prompt,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        batch,
    ):
        import torch, io, requests, base64
        import numpy as np
        from PIL import Image
        import time

        st = time.time()
        torch.cuda.empty_cache()
        prompt = "yogasana-vector-art. , " + prompt
        images = []
        print(prompt)
        for i in range(0, batch):
            image = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                denoising_end=0.8,
                guidance_scale=guidance_scale,
                output_type="latent",
                cross_attention_kwargs={"scale": 0.95},
            ).images[0]
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_start=0.8,
                image=image,
            ).images[0]
            torch.cuda.empty_cache()
            images.append(image)

        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            "cuda"
        )
        images, has_nsfw_concept = self.safety_checker(
            images=[np.array(i) for i in images],
            clip_input=safety_checker_input.pixel_values,
        )
        images = [Image.fromarray(np.uint8(i)) for i in images]

        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls = []
        for im in range(0, len(images)):
            filtered_image = io.BytesIO()
            if has_nsfw_concept[im]:
                pass
            else:
                images[im].save(filtered_image, "JPEG")
                myobj = {
                    "image": "data:image/png;base64,"
                    + (base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                }
                rps = requests.post(
                    url, json=myobj, headers={"Content-Type": "application/json"}
                )
                im_url = rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        self.runtime = time.time() - st

        return {
            "result": image_urls,
            "Has_NSFW_Content": has_nsfw_concept,
            "time": {
                "startup_time": self.container_execution_time,
                "runtime": self.runtime,
            },
        }
