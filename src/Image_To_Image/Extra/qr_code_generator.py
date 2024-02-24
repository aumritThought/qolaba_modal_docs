from modal import Image, Stub, method


def download_models():
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )
    import torch
    import transformers
    from diffusers import UniPCMultistepScheduler
    from diffusers import DPMSolverMultistepScheduler

    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
    )

    controlnet_tile = ControlNetModel.from_pretrained(
        "ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        controlnet=[controlnet, controlnet_tile],
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas="true"
    )


stub = Stub("QR_code_image2image")
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
            "pip install diffusers transformers accelerate opencv-python Pillow xformers==0.0.22",
            "pip install sdqrcode[diffusers]",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:
    def __enter__(self):
        import time
        import torch

        st = time.time()
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
        )
        from diffusers import DPMSolverMultistepScheduler

        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
        )

        controlnet_tile = ControlNetModel.from_pretrained(
            "ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/dreamshaper-8",
            controlnet=[controlnet, controlnet_tile],
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas="true"
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
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

    def add_margin(self, pil_img, top, right, bottom, left, color):
        from PIL import Image

        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    @method()
    def run_inference(
        self, file_url, prompt, guidance_scale, negative_prompt, batch, strength, bg_img
    ):
        import time

        st = time.time()

        controlnets_weights = [0.85, 0.35]
        controlnets_startstop = [(0, 1), (0.3, 0.7)]
        guidance_starts = [val[0] for val in controlnets_startstop]
        guidance_stops = [val[1] for val in controlnets_startstop]
        file_url = file_url.resize((768, 768))
        # print(img)
        file_url = self.add_margin(file_url, 50, 50, 50, 50, (255, 255, 255))
        file_url = file_url.resize((768, 768))

        image = self.pipe(
            prompt=prompt,
            image=[file_url, file_url],
            controlnet_conditioning_scale=controlnets_weights,
            control_guidance_start=guidance_starts,
            control_guidance_end=guidance_stops,
            guidance_scale=guidance_scale,
            # controlnet_guidance=controlnets_startstop,
            num_inference_steps=30,
            width=768,
            height=768,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch,
        ).images

        image_data = {"images": image, "Has_NSFW_Content": [False] * batch}
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
