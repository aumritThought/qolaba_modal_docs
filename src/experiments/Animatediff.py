from modal import Image, Stub, method

model_schema={
    "memory":10240,
    "container_idle_timeout":200,
    "gpu":"a100",
    }
model_schema["name"] = "AnimateDiff_video"

def download_models():
    import os
    os.system("git clone https://github.com/guoyww/AnimateDiff.git")
    os.chdir("AnimateDiff")
    os.system("git checkout sdxl")
    os.system("bash download_bashscripts/0-MotionModule.sh")

    from diffusers import DiffusionPipeline
    import torch

    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/dynavision-xl-0557", torch_dtype=torch.float16, variant="fp16").save_pretrained("models/Stable_diffusion")

    
stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.10")
    .dockerfile_commands([
        "RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git curl wget pkg-config libssl-dev openssl -y",
        "RUN pip install numpy==1.25.0 pillow==9.4.0 torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 scipy absl-py==1.4.0 accelerate==0.21.0 av==10.0.0 beautifulsoup4==4.12.2 bitsandbytes==0.41.1 colorama==0.4.4 decord==0.6.0 diffusers==0.20.2 easydict==1.10 einops==0.7.0rc1 gdown==4.7.1 imageio==2.27.0 omegaconf==2.3.0 tqdm==4.65.0 transformers==4.30.0 wandb==0.15.8 xformers==0.0.22.post4",
        "RUN pip install imageio[ffmpeg]"
        ]).run_function(
             download_models,
             gpu='t4'
        )
    )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=200, timeout=1200)
class stableDiffusion:
    def __enter__(self):
        import time
        st=time.time()
        import os,sys
        
        os.chdir("/root/AnimateDiff")
        print(os.listdir(os.getcwd()))
        sys.path.insert(0, "/root/AnimateDiff")

        import inspect
        from omegaconf import OmegaConf

        from diffusers import AutoencoderKL, EulerDiscreteScheduler

        from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

        from animatediff.models.unet import UNet3DConditionModel
        from animatediff.pipelines.pipeline_animation import AnimationPipeline
        from animatediff.utils.util import load_weights

        from diffusers.utils.import_utils import is_xformers_available

        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        config = OmegaConf.load("configs/inference/inference.yaml")

        tokenizer = CLIPTokenizer.from_pretrained("models/Stable_diffusion", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("models/Stable_diffusion", subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained("models/Stable_diffusion", subfolder="vae")
        tokenizer_two = CLIPTokenizer.from_pretrained("models/Stable_diffusion", subfolder="tokenizer_2")
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained("models/Stable_diffusion", subfolder="text_encoder_2")

        # init unet model
        unet = UNet3DConditionModel.from_pretrained_2d("models/Stable_diffusion", subfolder="unet",
                                                       unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))

        # Enable memory efficient attention
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

        scheduler = EulerDiscreteScheduler(timestep_spacing='leading', steps_offset=1, **config.noise_scheduler_kwargs)

        pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
            text_encoder_2=text_encoder_two, tokenizer_2=tokenizer_two
        ).to("cuda")

        # Load model weights
        self.pipeline = load_weights(
            pipeline=pipeline,
            motion_module_path="models/Motion_Module/mm_sdxl_v10_beta.ckpt",
            ckpt_path=config.get("ckpt_path", ""),
            lora_path=config.get("lora_path", ""),
            lora_alpha=config.get("lora_alpha", 0.8)
        )

        self.pipeline.unet = pipeline.unet.half()
        self.pipeline.text_encoder = pipeline.text_encoder.half()
        self.pipeline.text_encoder_2 = pipeline.text_encoder_2.half()
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.startup_time=time.time()-st

    @method()
    def run_inference(self, prompts, height, width, length, n_prompts, guidance_scale, steps):
        import time
        st=time.time()
        import base64, requests, torch, os, datetime
        from animatediff.utils.util import save_videos_grid

        time_str = datetime.datetime.now().strftime("%Y-%m-%d")

        savedir = f"sample/{height}_{width}-{time_str}"
        os.makedirs(savedir, exist_ok=True)

        random_seeds = -1
        seeds = []
        samples = []

        with torch.inference_mode():
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip([prompts], [n_prompts], [random_seeds])):
                # manually set random seed for reproduction
                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                seeds.append(torch.initial_seed())
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = self.pipeline(
                    prompt,
                    negative_prompt=n_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    single_model_length=length,
                ).videos
                samples.append(sample)
                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                prompt = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

                # save video
                save_videos_grid(sample, f"{savedir}/sample/{prompt}.mp4")
                print(f"save to {savedir}/sample/{prompt}.mp4")
                with open(f"{savedir}/sample/{prompt}.mp4", "rb") as f:
                    url="https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
                    byte=f.read()
                    myobj = {"image":"data:audio/mpeg;base64,"+(base64.b64encode(byte).decode("utf8"))}
                        
                    rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
                video_url=rps.json()["data"]["secure_url"]
        
        
        self.runtime=time.time()-st

        return {"result":[video_url],  
                "Has_NSFW_Content":[False], 
                "time": {"startup_time" : self.startup_time, "runtime":self.runtime}}


