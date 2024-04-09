from modal import Image, Stub, method


def download_models():
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
    import torch
    from controlnet_aux import HEDdetector

    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stablediffusionapi/rev-anim", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )
    feature_extractor = CLIPFeatureExtractor()


stub = Stub("scribble" + "_controlnet_" + "_image2image")
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
            "pip install diffusers transformers accelerate opencv-python Pillow xformers controlnet_aux matplotlib mediapipe",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:
    def __enter__(self):
        import time

        st = time.time()
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        from controlnet_aux import HEDdetector

        self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stablediffusionapi/rev-anim",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        # self.feature_extractor_safety = CLIPFeatureExtractor()
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
        self, file_url, prompt, guidance_scale, negative_prompt, batch, strength
    ):
        import torch
        import numpy as np
        import time
        from PIL import Image

        st = time.time()

        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch

        image = self.hed(file_url, scribble=True)
        image = image.resize((file_url.size[0], file_url.size[1]))
        image = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=20,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        ).images
        torch.cuda.empty_cache()

        # safety_checker_input = self.feature_extractor_safety(
        #         image, return_tensors="pt"
        #     ).to("cuda")
        # image, has_nsfw_concept = self.safety_checker(
        #                 images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
        #             )
        # image=[ Image.fromarray(np.uint8(i)) for i in image]

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
