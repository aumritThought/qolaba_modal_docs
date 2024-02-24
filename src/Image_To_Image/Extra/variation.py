from modal import Image, Stub, method


def download_models():
    import os, sys, torch

    os.chdir("../IP-Adapter")
    sys.path.insert(0, "../IP-Adapter")

    os.system("git clone https://huggingface.co/h94/IP-Adapter")

    from ip_adapter import IPAdapterPlusXL
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    from diffusers import StableDiffusionXLPipeline

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )
    feature_extractor = CLIPFeatureExtractor()

    base_model_path = "../Starlight.safetensors"

    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        base_model_path, torch_dtype=torch.float16, add_watermarker=False
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    device = "cuda"
    image_encoder_path = "IP-Adapter/models/image_encoder"
    ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

    ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)


stub = Stub("variation_image2image")
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6 git wget -y",
            "apt-get install git-lfs",
            "git lfs install",
            "git clone https://github.com/tencent-ailab/IP-Adapter.git",
            "pip install --upgrade diffusers transformers accelerate safetensors torch xformers onnxruntime einops insightface omegaconf",
            "wget https://civitai.com/api/download/models/182077",
            "mv 182077 Starlight.safetensors",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:
    def __enter__(self):
        import time

        st = time.time()
        import os, sys, torch

        from diffusers import StableDiffusionXLPipeline

        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")
        from ip_adapter import IPAdapterPlusXL
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        device = "cuda"
        image_encoder_path = "IP-Adapter/models/image_encoder"
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

        base_model_path = "../Starlight.safetensors"

        # load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_path, torch_dtype=torch.float16, add_watermarker=False
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.ip_model = IPAdapterPlusXL(
            pipe, image_encoder_path, ip_ckpt, device, num_tokens=16
        )
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor_safety = CLIPImageProcessor()
        self.container_execution_time = time.time() - st

    def generate_image_urls(self, image_data):
        import io, base64, requests

        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls = []
        for im in range(0, len(image_data["images"])):
            filtered_image = io.BytesIO()
            if image_data["Has_NSFW_Content"][im]:
                pass
            else:
                image_data["images"][im].save(filtered_image, "JPEG")
                myobj = {
                    "image": "data:image/png;base64,"
                    + (base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                }
                rps = requests.post(
                    url, json=myobj, headers={"Content-Type": "application/json"}
                )
                im_url = rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        return image_urls

    @method()
    def run_inference(
        self,
        file_url,
        prompt,
        guidance_scale,
        batch,
        num_inference_steps=30,
        negative_prompt="blurr",
    ):

        import time

        st = time.time()
        import random, torch
        import numpy as np
        from PIL import Image

        height = file_url.size[1]
        width = file_url.size[0]

        torch.cuda.empty_cache()
        image = []
        for i in range(0, batch):
            seed = random.sample(range(1, 1000000000), 1)[0]
            img = self.ip_model.generate(
                pil_image=file_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_samples=1,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                seed=seed,
                scale=0.5,
            )
            image.append(img[0])
        torch.cuda.empty_cache()

        safety_checker_input = self.feature_extractor_safety(
            image, return_tensors="pt"
        ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
            images=[np.array(i) for i in image],
            clip_input=safety_checker_input.pixel_values,
        )
        image = [Image.fromarray(np.uint8(i)) for i in image]
        torch.cuda.empty_cache()
        image_data = {"images": image, "Has_NSFW_Content": has_nsfw_concept}
        image_urls = self.generate_image_urls(image_data)

        self.runtime = time.time() - st
        return {
            "result": image_urls,
            "Has_NSFW_Content": [False] * batch,
            "time": {
                "startup_time": self.container_execution_time,
                "runtime": self.runtime,
            },
        }
