from modal import Image, Stub, method


def download_models():
    import torch, os, sys
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
    from PIL import Image
    os.chdir("../IP-Adapter")
    sys.path.insert(0, "../IP-Adapter")
    from ip_adapter import IPAdapterPlus


    os.system("git clone https://huggingface.co/h94/IP-Adapter")

    base_model_path = "danbrown/RevAnimated-v1-2-2"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "IP-Adapter/models/image_encoder/"
    ip_ckpt = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
    device = "cuda"

    # load SDXL pipeline
    noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
    )
    pipe.enable_xformers_memory_efficient_attention()
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

stub = Stub("face_consistent_image2image")
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
        import torch, os, sys
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
        from PIL import Image
        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")
        from ip_adapter import IPAdapterPlus


        os.system("git clone https://huggingface.co/h94/IP-Adapter")

        base_model_path = "danbrown/RevAnimated-v1-2-2"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        image_encoder_path = "IP-Adapter/models/image_encoder/"
        ip_ckpt = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
        device = "cuda"

        # load SDXL pipeline
        noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
        )
        pipe.enable_xformers_memory_efficient_attention()
        self.ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)


    @method()
    def run_inference(self, img,prompt,height, width ,batch):
        import random

        seed=random.sample(range(1, 1000000000), 1)[0]

        images = self.ip_model.generate(pil_image=img, num_samples=batch, num_inference_steps=50,height=height, width=width,prompt=prompt, seed=seed)

        return {"images":images,  "Has_NSFW_Content":[False]*batch}
