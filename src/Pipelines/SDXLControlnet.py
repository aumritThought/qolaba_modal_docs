from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, SDXLControlNetParameters, InitParameters
from src.utils.Globals import get_base_image, get_refiner, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import sdxl_model_list, VOLUME_NAME, VOLUME_PATH, SECRET_NAME, controlnet_model_list
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
import torch, time
from PIL import Image
from src.utils.Constants import CANNY, OPENPOSE, SKETCH, DEPTH
from controlnet_aux import OpenposeDetector, CannyDetector, MidasDetector, PidiNetDetector
import numpy as np
from diffusers.utils import load_image

stub_name = StubNames().sdxl_controlnet

stub = Stub(stub_name)



class ContrloNetImageGeneration:
    def __init__(self, image : Image, controlnet : str) -> None:
        self.images = []
        self.image = image
        self.controlnet = controlnet

    def canny_image_generation(self):
        if(CANNY == self.controlnet):
            canny = CannyDetector()
            image = canny(self.image, detect_resolution=384, image_resolution=1024, output_type = "pil")
            self.images.append(image.resize(self.image.size))
        return self

    def openpose_image_generation(self):
        if(OPENPOSE == self.controlnet):
            openpose =  OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
            image = openpose(self.image, detect_resolution=512, image_resolution=1024)
            image = np.array(image)[:, :, ::-1]           
            image = Image.fromarray(np.uint8(image))
            self.images.append(image.resize(self.image.size))
        return self
        
    def depth_image_generation(self):
        if(DEPTH == self.controlnet):
            depth = MidasDetector.from_pretrained("valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large").to("cuda")
            image = depth(self.image, detect_resolution=512, image_resolution=1024,  output_type="pil")
            self.images.append(image.resize(self.image.size))
        return self
    
    def sketch_image_generation(self):
        if(SKETCH == self.controlnet):
            pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
            image = pidinet(self.image, detect_resolution=1024, image_resolution=1024, apply_filter=True)
            self.images.append(image.resize(self.image.size))
        return self
    
    def prepare_images(self) -> Image:
        return self.canny_image_generation().depth_image_generation().openpose_image_generation().sketch_image_generation().images[0]
    


def download_base_sdxl():
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        torch_dtype=torch.float16,
        varient="fp16",
    )
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", adapter = adapter
        )
    url = "https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg"
    image = load_image(url)
    controlnet_class = ContrloNetImageGeneration(image, [CANNY, DEPTH, SKETCH, OPENPOSE])
    controlnet_class.prepare_images()


vol = Volume.persisted(VOLUME_NAME)

image = get_base_image().run_function(download_base_sdxl, gpu = "t4")

stub.image = image


@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)])
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        self.init_parameters : InitParameters = InitParameters(**init_parameters)
        st = time.time()

        controlnet_model = T2IAdapter.from_pretrained(
                    controlnet_model_list[self.init_parameters.controlnet_model],
                    torch_dtype=torch.float16,
                    varient="fp16")

        self.pipe = StableDiffusionXLAdapterPipeline.from_single_file(
            sdxl_model_list.get(self.init_parameters.model), adapter = controlnet_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")
        if(self.init_parameters.lora_model):
            self.pipe.load_lora_weights(self.init_parameters.lora_model)

        self.refiner = get_refiner(self.pipe)

        self.pipe.enable_xformers_memory_efficient_attention()
        self.refiner.enable_xformers_memory_efficient_attention()
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:

        parameters : SDXLControlNetParameters = SDXLControlNetParameters(**parameters)

        parameters.image = get_image_from_url(parameters.image, resize = True)

        controlnet_image_model = ContrloNetImageGeneration(parameters.image, self.init_parameters.controlnet_model)
        controlnet_image = controlnet_image_model.prepare_images()

        st = time.time()

        images = []

        for i in range(0, parameters.batch):
            image = self.pipe(
                prompt = parameters.prompt,
                negative_prompt = parameters.negative_prompt,
                image=controlnet_image,
                denoising_end = 0.8,
                guidance_scale = parameters.guidance_scale,
                output_type="latent",
                num_inference_steps = parameters.num_inference_steps,
                cross_attention_kwargs={"scale": parameters.lora_scale},
                adapter_conditioning_scale = parameters.strength
            ).images[0]
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt = parameters.prompt,
                num_inference_steps = parameters.num_inference_steps,
                guidance_scale = parameters.guidance_scale,
                denoising_start=0.8,
                image=image,
            ).images[0]

            torch.cuda.empty_cache()
            images.append(image)

        image_urls, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)