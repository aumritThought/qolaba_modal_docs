from modal import Image, Stub, method

model_schema={
    "memory":10240,
    "container_idle_timeout":200,
    "gpu":"a100",
    }
model_schema["name"] = "stable_diffusion_video"

def download_models():
    import os,sys, torch
    os.chdir('/generative-models')
    sys.path.insert(0, '/generative-models')
    os.system("apt install ffmpeg libavcodec-extra -y")
    os.system("pip install imageio[ffmpeg] ffmpeg-python")
    from omegaconf import OmegaConf
    from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
    from sgm.util import instantiate_from_config
    
    num_frames = 25
    num_steps = 30
    config = "scripts/sampling/configs/svd_xt.yaml"

    device="cuda"
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
    filter = DeepFloydDataFiltering(verbose=False, device=device)
     

stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.10")
    .dockerfile_commands([
        "RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git curl wget pkg-config libssl-dev openssl -y",
        "RUN apt-get update \
        && apt-get install -y --no-install-recommends build-essential curl \
        && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        && . $HOME/.cargo/env\
        && rustup default stable",
        "RUN git clone https://github.com/Stability-AI/generative-models.git",
        "RUN cd generative-models && . $HOME/.cargo/env && pip3 install -r requirements/pt2.txt",
        "RUN cd generative-models && pip3 install .",
        "RUN cd generative-models && mkdir checkpoints",
        "RUN cd generative-models/checkpoints && wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors",
        "RUN cd generative-models/checkpoints && wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors"
        ]).run_function(
             download_models,
             gpu='t4'
        )
    )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=200, timeout=1200)
class stableDiffusion: 
    def __enter__(self):
        import os,sys, torch
        import time
        st=time.time()
        os.chdir('/generative-models')
        sys.path.insert(0, '/generative-models')
        from omegaconf import OmegaConf
        from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
        from sgm.util import instantiate_from_config
        
        num_frames = 30
        num_steps = 30
        config = "scripts/sampling/configs/svd_xt.yaml"

        device="cuda"
        config = OmegaConf.load(config)
        if device == "cuda":
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device

        config.model.params.sampler_config.params.num_steps = num_steps
        config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )
        if device == "cuda":
            with torch.device(device):
                self.model = instantiate_from_config(config.model).to(device).eval()
        else:
            self.model = instantiate_from_config(config.model).to(device).eval()
        self.filter = DeepFloydDataFiltering(verbose=False, device=device)
        self.startup_time=time.time()-st


    @method()
    def run_inference(self,file_url):
        import math
        import os,sys
        os.chdir('/generative-models')
        sys.path.insert(0, '/generative-models')
        from glob import glob
        from typing import Optional

        import numpy as np
        import torch
        from einops import rearrange, repeat
        from torchvision.transforms import ToTensor

        import random, time, base64, requests, imageio
        from PIL import Image
        from io import BytesIO


        def fetch_image_from_url(url):
            try:
                response = requests.get(url)
                response.raise_for_status()

                image = Image.open(BytesIO(response.content))

                return image
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error fetching image: {e}", "Provide URL of Image")

        def resize_image(img):
            img=img.resize((64 * round(img.size[0] / 64), 64 * round(img.size[1] / 64)))
            if(img.size[0]>1024 or img.size[0]<512 or img.size[1]>1024 or img.size[1]<512):
                if(img.size[1]>=img.size[0]):
                    height=1024
                    width=((int(img.size[0]*1024/img.size[1]))//64)*64
                else:
                    width=1024
                    height=((int(img.size[1]*1024/img.size[0]))//64)*64
                img=img.resize((width, height))
            return img

        if (not(isinstance(file_url, Image.Image))):
            file_url=fetch_image_from_url(file_url)
            file_url=resize_image(file_url)

        def sample(
            image,  # Can either be image file or folder with image files
            num_frames: Optional[int] = None,
            num_steps: Optional[int] = None,
            version: str = "svd_xt",
            fps_id: int = 6,
            motion_bucket_id: int = 127,
            cond_aug: float = 0.02,
            seed: int = random.randint(1, 100000000),
            decoding_t: int = 5,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
            device: str = "cuda",
            output_folder= "outputs/simple_video_sample/svd_xt/",
        ):

            num_frames = 30

            torch.manual_seed(seed)


            if image.mode == "RGBA":
                image = image.convert("RGB")
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

            image = image.unsqueeze(0).to(device)
            H, W = image.shape[2:]
            assert image.shape[1] == 3
            F = 8
            C = 4
            shape = (num_frames, C, H // F, W // F)
            

            value_dict = {}
            value_dict["motion_bucket_id"] = motion_bucket_id
            value_dict["fps_id"] = fps_id
            value_dict["cond_aug"] = cond_aug
            value_dict["cond_frames_without_noise"] = image
            value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
            value_dict["cond_aug"] = cond_aug

            with torch.no_grad():
                with torch.autocast(device):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                        value_dict,
                        [1, num_frames],
                        T=num_frames,
                        device=device,
                    )
                    c, uc = self.model.conditioner.get_unconditional_conditioning(
                            batch,
                            batch_uc=batch_uc,
                            force_uc_zero_embeddings=[
                                "cond_frames",
                                "cond_frames_without_noise",
                            ],
                        )

                    for k in ["crossattn", "concat"]:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                    randn = torch.randn(shape, device=device)

                    additional_model_inputs = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(
                            2, num_frames
                        ).to(device)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c):
                            return self.model.denoiser(
                                self.model.model, input, sigma, c, **additional_model_inputs
                            )

                    samples_z = self.model.sampler(denoiser, randn, cond=c, uc=uc)
                    self.model.en_and_decode_n_samples_a_time = decoding_t
                    samples_x = self.model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                    os.makedirs(output_folder, exist_ok=True)
                    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

                    samples = self.filter(samples)
                    vid = (
                            (rearrange(samples, "t c h w -> t h w c") * 255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )


                    writer = imageio.get_writer(video_path, fps=int(fps_id), macro_block_size=None)

                    for frame in vid:
                        writer.append_data(frame)

                    writer.close()

                    with open(video_path, "rb") as f:
                        url="https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
                        byte=f.read()
                        myobj = {"image":"data:audio/mpeg;base64,"+(base64.b64encode(byte).decode("utf8"))}
                        
                        rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
                    video_url=rps.json()["data"]["secure_url"]
            torch.cuda.empty_cache()
            print(torch.cuda.max_memory_allocated())
            os.remove(video_path)
            return video_url


        def get_unique_embedder_keys_from_conditioner(conditioner):
            return list(set([x.input_key for x in conditioner.embedders]))


        def get_batch(keys, value_dict, N, T, device):
            batch = {}
            batch_uc = {}

            for key in keys:
                if key == "fps_id":
                    batch[key] = (
                        torch.tensor([value_dict["fps_id"]])
                        .to(device)
                        .repeat(int(math.prod(N)))
                    )
                elif key == "motion_bucket_id":
                    batch[key] = (
                        torch.tensor([value_dict["motion_bucket_id"]])
                        .to(device)
                        .repeat(int(math.prod(N)))
                    )
                elif key == "cond_aug":
                    batch[key] = repeat(
                        torch.tensor([value_dict["cond_aug"]]).to(device),
                        "1 -> b",
                        b=math.prod(N),
                    )
                elif key == "cond_frames":
                    batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
                elif key == "cond_frames_without_noise":
                    batch[key] = repeat(
                        value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
                    )
                else:
                    batch[key] = value_dict[key]

            if T is not None:
                batch["num_video_frames"] = T

            for key in batch.keys():
                if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                    batch_uc[key] = torch.clone(batch[key])
            return batch, batch_uc


        st=time.time()
        video_url=sample(image=file_url)
        self.runtime=time.time()-st
        return {"result":[video_url],  
                "Has_NSFW_Content":[False], 
                "time": {"startup_time" : self.startup_time, "runtime":self.runtime}}