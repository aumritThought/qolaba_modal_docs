from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, SDXLImage2ImageParameters, InitParameters
from src.utils.Globals import get_base_image, get_refiner, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import sdxl_model_list, VOLUME_NAME, VOLUME_PATH, SECRET_NAME, extra_negative_prompt
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch, time

stub_name = StubNames().sdxl_image_to_image

stub = Stub(stub_name)

def download_base_sdxl():
    StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )

vol = Volume.persisted(VOLUME_NAME)

image = get_base_image().run_function(download_base_sdxl, secrets= [Secret.from_name(SECRET_NAME)])

stub.image = image


@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)])
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        init_parameters : InitParameters = InitParameters(**init_parameters)
        st = time.time()

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            sdxl_model_list.get(init_parameters.model), torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")
        if(init_parameters.lora_model):
            self.pipe.load_lora_weights(init_parameters.lora_model)

        self.refiner = get_refiner(self.pipe)

        # self.pipe.enable_xformers_memory_efficient_attention()
        # self.refiner.enable_xformers_memory_efficient_attention()
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:

        parameters : SDXLImage2ImageParameters = SDXLImage2ImageParameters(**parameters)

        parameters.negative_prompt = parameters.negative_prompt + extra_negative_prompt

        parameters.file_url = get_image_from_url(parameters.file_url)

        st = time.time()

        images = []

        for i in range(0, parameters.batch):
            image = self.pipe(
                prompt = parameters.prompt,
                negative_prompt = parameters.negative_prompt,
                strength=parameters.strength,
                image=parameters.file_url,
                denoising_end = 0.8,
                guidance_scale = parameters.guidance_scale,
                output_type="latent",
                cross_attention_kwargs={"scale": parameters.lora_scale},
            ).images[0]
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt = parameters.prompt,
                num_inference_steps = int(50*parameters.strength),
                guidance_scale = parameters.guidance_scale,
                denoising_start=0.8,
                image=image,
            ).images[0]

            torch.cuda.empty_cache()
            images.append(image)

        image_urls, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)