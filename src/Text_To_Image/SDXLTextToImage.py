from src.data_models.ModalAppSchemas import SDXLParameters
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, SDXLParameters
from src.utils.Globals import get_base_image, get_refiner, SafetyChecker, generate_image_urls, prepare_response
from src.utils.Constants import sdxl_model_list
from diffusers import StableDiffusionXLPipeline
import torch, time


class SDXLTextToImage:
    def __init__(self, input_parameters : SDXLParameters, safety_checker : SafetyChecker):
        st = time.time()
        self.parameters = input_parameters
        self.safety_checker = safety_checker
        self.pipe = StableDiffusionXLPipeline.from_single_file(
                sdxl_model_list.get(self.parameters.model), torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            )
        
        self.pipe.to("cuda")
        if(self.parameters.lora_model):
            self.pipe.load_lora_weights(self.parameters.lora_model)

        self.refiner = get_refiner(self.pipe)

        self.container_execution_time = time.time() - st

    def generate_images(self):
        st = time.time()

        images = []

        for i in range(0, self.parameters.batch):
            image = self.pipe(
                prompt = self.parameters.prompt,
                height = self.parameters.height,
                width = self.parameters.width,
                negative_prompt = self.parameters.negative_prompt,
                num_inference_steps = self.parameters.num_inference_steps,
                denoising_end = 0.8,
                guidance_scale = self.parameters.guidance_scale,
                output_type="latent",
                cross_attention_kwargs={"scale": self.parameters.lora_scale},
            ).images[0]
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt = self.parameters.prompt,
                num_inference_steps = self.parameters.num_inference_steps,
                guidance_scale = self.parameters.guidance_scale,
                denoising_start=0.8,
                image=image,
            ).images[0]

            torch.cuda.empty_cache()
            images.append(image)

        image_urls, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)