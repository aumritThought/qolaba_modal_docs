from modal import Image, Stub, method


def download_models():
    import torch

    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")


stub = Stub("depth"+"_controlnet_"+"_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import torch
        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL


        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()



    @method()
    def run_inference(self, img, prompt,guidance_scale,negative_prompt, batch, strength):
        import torch
        import cv2
        from PIL import Image
        import numpy as np

        def get_depth_map(image):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad(), torch.autocast("cuda"):
                depth_map = self.depth_estimator(image).predicted_depth

            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(1024, 1024),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            image = torch.cat([depth_map] * 3, dim=1)

            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
            return image
        
        
        
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        
        image = get_depth_map(img)
        image = image.resize((img.size))
        image = self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=guidance_scale, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.5)
        return {"images":image.images,  "Has_NSFW_Content":[False]*batch}
