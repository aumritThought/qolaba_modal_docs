from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, DifferentialDiffusionInpainting, InitParameters
from src.utils.Globals import get_base_image, get_refiner, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url, invert_bw_image_color
from src.utils.Constants import sdxl_model_list, VOLUME_NAME, VOLUME_PATH, SECRET_NAME, extra_negative_prompt, OUTPUT_IMAGE_EXTENSION
from src.ModalPipelines.DDInpaintingClass import StableDiffusionXLDiffImg2ImgPipeline 
import torch, time
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
from PIL.Image import Image

stub_name = StubNames().differential_diffusion_inpainting

stub = Stub(stub_name)

def download_base_sdxl():
    StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
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
        st = time.time()
        init_parameters : InitParameters = InitParameters(**init_parameters)

        self.pipe = StableDiffusionXLDiffImg2ImgPipeline.from_single_file(
            sdxl_model_list.get(init_parameters.model), torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")

        self.refiner = get_refiner(self.pipe)

        # self.pipe.enable_xformers_memory_efficient_attention()
        # self.refiner.enable_xformers_memory_efficient_attention()
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    def preprocess_image(self, image : Image) -> torch.tensor:
        image = image.convert("RGB")
        image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
        image = transforms.ToTensor()(image)
        image = image * 2 - 1
        image = image.unsqueeze(0).to("cuda")
        return image


    def preprocess_map(self, map : Image) ->  torch.tensor:
        map = map.convert("L")
        map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
        map = transforms.ToTensor()(map)
        map = 1 - map
        map = map.to("cuda")
        return map

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()
        parameters : DifferentialDiffusionInpainting = DifferentialDiffusionInpainting(**parameters)

        images = []

        parameters.negative_prompt = parameters.negative_prompt + extra_negative_prompt

        parameters.file_url = get_image_from_url(parameters.file_url)
        parameters.mask_url = invert_bw_image_color(get_image_from_url(parameters.mask_url)).resize(parameters.file_url.size)

        image = self.preprocess_image(parameters.file_url)
        map = self.preprocess_map(VaeImageProcessor().blur(parameters.mask_url, blur_factor=33))
        print(parameters.prompt)
        for i in range(0, parameters.batch):
            modified_image = self.pipe(prompt=parameters.prompt, 
                              original_image=image, 
                              image=image, 
                              strength = parameters.strength, 
                              guidance_scale=parameters.guidance_scale,
                              num_images_per_prompt=1,
                              negative_prompt=parameters.negative_prompt,
                              map=map,
                              num_inference_steps=parameters.num_inference_steps, 
                              denoising_end=0.8, 
                              output_type="latent").images[0]

            modified_image = self.refiner(prompt=parameters.prompt, 
                                 original_image=image, 
                                 image=modified_image, 
                                 strength=parameters.strength, 
                                 guidance_scale=parameters.guidance_scale,
                                 num_images_per_prompt=1,
                                 negative_prompt=parameters.negative_prompt,
                                 map=map,
                                 num_inference_steps=parameters.num_inference_steps,
                                 denoising_start=0.8).images[0]


            torch.cuda.empty_cache()
            images.append(modified_image)

        images, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(images, has_nsfw_content, self.container_execution_time, self.runtime, OUTPUT_IMAGE_EXTENSION)
    