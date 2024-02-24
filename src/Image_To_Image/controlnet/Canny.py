from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    import torch
    controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            "../Starlight.safetensors",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to("cuda")

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()


stub = Stub("Canny"+"_controlnet_"+"_image2image")
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "wget https://civitai.com/api/download/models/182077",
        "pip install diffusers --upgrade",
        "pip install controlnet_aux invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf",
        "mv 182077 Starlight.safetensors",
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import time
        st= time.time()
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        import torch
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            "../Starlight.safetensors",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        # self.feature_extractor = CLIPFeatureExtractor()  
        self.container_execution_time=time.time()-st

    def generate_image_urls(self, image_data):
        import io, base64, requests
        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls=[]
        for im in range(0, len(image_data["images"])):
            filtered_image = io.BytesIO()
            if(image_data["Has_NSFW_Content"][im]):
                pass
            else:
                image_data["images"][im].save(filtered_image, "JPEG")
                myobj = {
                        "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                    }
                rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
                im_url=rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        return image_urls



    @method()
    def run_inference(self, file_url, prompt,guidance_scale,negative_prompt, batch, strength):
        import cv2, time, torch
        from PIL import Image
        import numpy as np
        st=time.time()

        image = np.array(file_url)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        images = []
        for i in range(0, batch):
            img=self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=guidance_scale, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.5).images
            images.append(img[0])

        torch.cuda.empty_cache()

        image_data = {"images" :  images, "Has_NSFW_Content" : [False]*batch}
        
        image_urls =self.generate_image_urls(image_data)
        self.runtime=time.time()-st


        from src.data_models.ImageToImage import Inference
        

        inference = Inference()
        inference.result = image_urls
        inference.has_nsfw_content = [False]*batch
        inference.time.startup_time = self.container_execution_time
        inference.time.runtime = self.runtime

        return inference
