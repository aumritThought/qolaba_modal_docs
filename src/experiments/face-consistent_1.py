from modal import Image, Stub, method


def download_models():
    import os, sys

    os.chdir("../IP-Adapter")
    sys.path.insert(0, "../IP-Adapter")
    import cv2
    from insightface.app import FaceAnalysis
    import torch

    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    import torch
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
    from PIL import Image

    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceIDXL
    from huggingface_hub import login

    login("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")
    os.system(
        "wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin"
    )
    ip_ckpt = "ip-adapter-faceid_sdxl.bin"
    device = "cuda"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/protovision-xl-high-fidel",
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.load_lora_weights(
        "Qolaba/Lora_Models",
        weight_name="Vector cartoon illustration-000008.safetensors",
    )

    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )
    feature_extractor = CLIPFeatureExtractor()


stub = Stub("face_consistent_1_image2image")
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6 git wget -y",
            "apt-get install git-lfs",
            "git lfs install",
            "git clone https://github.com/tencent-ailab/IP-Adapter.git",
            "pip install --upgrade diffusers transformers accelerate safetensors torch xformers onnxruntime einops insightface omegaconf",
            "pip install -U peft",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:
    def __enter__(self):
        import time

        st = time.time()
        import os, sys

        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")
        import cv2
        from insightface.app import FaceAnalysis
        import torch

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        import torch
        from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
        from PIL import Image

        from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceIDXL
        from huggingface_hub import login

        login("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")

        ip_ckpt = "ip-adapter-faceid_sdxl.bin"
        device = "cuda"

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stablediffusionapi/protovision-xl-high-fidel",
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        pipe.load_lora_weights(
            "Qolaba/Lora_Models",
            weight_name="Vector cartoon illustration-000008.safetensors",
        )

        self.ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        )
        self.feature_extractor = CLIPFeatureExtractor()

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
    def run_inference(self, file_url, prompt, height, width, batch):
        import torch
        import time
        import numpy as np
        from PIL import Image
        from insightface.utils import face_align

        st = time.time()

        face_img = np.array(file_url)
        faces = self.app.get(face_img)
        face_image = face_align.norm_crop(
            face_img, landmark=faces[0].kps, image_size=224
        )

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        negative_prompt = (
            "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
        )

        image = []

        for i in range(0, batch):

            img = self.ip_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_image=face_image,
                faceid_embeds=faceid_embeds,
                num_samples=1,
                width=width,
                height=height,
                num_inference_steps=30,
            )
            image.append(img[0])
        torch.cuda.empty_cache()

        # safety_checker_input = self.feature_extractor(
        #         image, return_tensors="pt"
        #     ).to("cuda")
        # image, has_nsfw_concept = self.safety_checker(
        #                 images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
        #             )
        # image=[ Image.fromarray(np.uint8(i)) for i in image]
        has_nsfw_concept = [False] * batch
        image_data = {"images": image, "Has_NSFW_Content": has_nsfw_concept}
        image_urls = self.generate_image_urls(image_data)

        self.runtime = time.time() - st
        return {
            "result": image_urls,
            "Has_NSFW_Content": has_nsfw_concept,
            "time": {
                "startup_time": self.container_execution_time,
                "runtime": self.runtime,
            },
        }
