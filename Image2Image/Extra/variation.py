from modal import Image, Stub, method


def download_models():
    import os, sys, torch
    from diffusers import StableDiffusionXLPipeline

    os.chdir("../IP-Adapter")
    sys.path.insert(0, "../IP-Adapter")
    os.system("git clone https://huggingface.co/h94/IP-Adapter")
    print(os.getcwd())
    from ip_adapter import IPAdapterXL

    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False, use_safetensors=True, variant="fp16"
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    device = "cuda"
    image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
    ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

stub = Stub("Canny_controlnet_variation_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get install git-lfs",
        "git lfs install",
        "git clone https://github.com/tencent-ailab/IP-Adapter.git",
        "pip install diffusers==0.19.3 transformers accelerate safetensors torch xformers"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import os, sys, torch
        from diffusers import StableDiffusionXLPipeline
        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")
        from ip_adapter import IPAdapterXL

        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"  
        device = "cuda"

        # load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False, use_safetensors=True, variant="fp16"
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

    @method()
    def run_inference(self, img,guidance_scale,batch):
        import random
        height=img.size[1]
        width=img.size[0]
        seed=random.sample(range(1, 1000000000), 1)[0]

        image = self.ip_model.generate(pil_image=img, num_samples=batch, num_inference_steps=30, height=height, width=width, seed=seed)

        return {"images":image,  "Has_NSFW_Content":[False]*batch}
