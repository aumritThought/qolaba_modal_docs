from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import torch
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")


    controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
        )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stablediffusionapi/rev-anim", controlnet=controlnet, torch_dtype=torch.float16
        )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()

stub = Stub("seg"+"_controlnet_"+"_image2image")
image = (
    Image.debian_slim(python_version="3.11")
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
        import time
        st= time.time()
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        import torch
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        import numpy as np

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stablediffusionapi/rev-anim", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        self.palette = np.asarray([
            [0, 0, 0],
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ])
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        # self.feature_extractor_safety = CLIPFeatureExtractor()  
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
    def run_inference(self, img, prompt,guidance_scale,negative_prompt, batch, strength):
        from PIL import Image
        import numpy as np
        import torch
        import time
        st=time.time()
        
        prompt = [prompt] * batch
        negative_prompt = [negative_prompt] * batch
        
        pixel_values = self.image_processor(img, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)

        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

        for label, color in enumerate(self.palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)

        image = Image.fromarray(color_seg)

        image = self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=guidance_scale, negative_prompt=negative_prompt).images
        torch.cuda.empty_cache()

        # safety_checker_input = self.feature_extractor_safety(
        #         image, return_tensors="pt"
        #     ).to("cuda")
        # image, has_nsfw_concept = self.safety_checker(
        #                 images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
        #             )
        # image=[ Image.fromarray(np.uint8(i)) for i in image] 

        image_data = {"images" :  image, "Has_NSFW_Content" : [False]*batch}
        image_urls =self.generate_image_urls(image_data)
        self.runtime=time.time()-st

        return {"result":image_urls,  
                "Has_NSFW_Content":[False]*batch, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
