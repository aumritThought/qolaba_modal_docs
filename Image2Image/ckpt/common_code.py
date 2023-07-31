from modal import Image, Secret, Stub, method

def give_model_schema():
    return {
    "memory":10240,
    "container_idle_timeout":60,
    "gpu":"a10g"
}

def download_models(model_id):
    import os, sys
    sys.path.append("/AITemplate/examples/05_stable_diffusion/")
    os.chdir("/AITemplate/examples/05_stable_diffusion/")
    os.system("python3 scripts/download_pipeline.py --ckpt "+model_id)

    from scripts.compile_alt import compile_diffusers
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    compile_diffusers("./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2")
    class stableDiffusion:   
        def __init__(self):
            import sys,os
            sys.path.append("/AITemplate/examples/05_stable_diffusion/")
            os.chdir("/AITemplate/examples/05_stable_diffusion/")
            from aitemplate.utils.import_path import import_parent
            if __name__ == "__main__":
                import_parent(filepath=__file__, level=1)
            from src.pipeline_stable_diffusion_ait_alt import StableDiffusionAITPipeline
            from diffusers import  EulerDiscreteScheduler

            hf_hub_or_path="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2"
            self.pipe = StableDiffusionAITPipeline(
                hf_hub_or_path=hf_hub_or_path,
                ckpt=None,
            )
            self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
                hf_hub_or_path, subfolder="scheduler"
            )
    
    a=stableDiffusion()
    
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

class SD_common:   
    def __init__(self, prompt_extra) -> None:
        self.prompt_extra=prompt_extra
        import sys,os
        sys.path.append("/AITemplate/examples/05_stable_diffusion/")
        os.chdir("/AITemplate/examples/05_stable_diffusion/")
        from aitemplate.utils.import_path import import_parent
        if __name__ == "__main__":
            import_parent(filepath=__file__, level=1)
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        from src.pipeline_stable_diffusion_img2img_ait import StableDiffusionImg2ImgAITPipeline
        import torch
        from diffusers import PNDMScheduler

        hf_hub_or_path="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2"
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPFeatureExtractor()
        self.prompt_extra=prompt_extra
        self.image_pipe=StableDiffusionImg2ImgAITPipeline.from_pretrained(
            hf_hub_or_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
        ).to("cuda")
        self.scheduler = PNDMScheduler.from_pretrained(
            hf_hub_or_path, subfolder="scheduler"
        )
    def run_inference(self,img, prompt,guidance_scale,negative_prompt,batch, strength):

        import torch
        import numpy as np
        from PIL import Image
        prompt=prompt+ self.prompt_extra
        prompt = [prompt] * batch

        with torch.autocast("cuda"):
            image = self.image_pipe(
                    prompt=prompt, init_image=img, strength=strength, guidance_scale=guidance_scale
                ).images

        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        return {"images":image,  "Has_NSFW_Content":has_nsfw_concept}