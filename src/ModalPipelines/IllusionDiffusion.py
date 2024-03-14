from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, IllusionDuiffusion
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, extra_negative_prompt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch, time
        
stub_name = StubNames().illusion_diffusion

stub = Stub(stub_name)

def download_base_sdxl():
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
    ) 
    StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/dreamshaper-8", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
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
        st = time.time()
        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/dreamshaper-8", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")

        # self.pipe.enable_xformers_memory_efficient_attention()
        # self.refiner.enable_xformers_memory_efficient_attention()
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:

        parameters : IllusionDuiffusion = IllusionDuiffusion(**parameters)

        parameters.file_url = get_image_from_url(parameters.file_url)

        parameters.negative_prompt = parameters.negative_prompt + extra_negative_prompt

        st = time.time()

        images = []

        for i in range(0, parameters.batch):
            image = self.pipe(
                prompt = parameters.prompt,
                negative_prompt = parameters.negative_prompt,
                controlnet_conditioning_scale = parameters.controlnet_scale,
                image=parameters.file_url,
                guidance_scale = parameters.guidance_scale,
                batch = 1,
                num_inference_steps = parameters.num_inference_steps
            ).images[0]
            torch.cuda.empty_cache()

            images.append(image)

        image_urls, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)