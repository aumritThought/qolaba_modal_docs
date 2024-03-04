from modal import Stub, method, enter, build
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, SDXLParameters
from src.utils.Globals import get_base_image, get_refiner, SafetyChecker, generate_image_urls, prepare_response
from src.utils.Constants import sdxl_model_list
from diffusers import StableDiffusionXLPipeline
import torch, time

stub_name = StubNames().sdxl_text_to_image

stub = Stub(stub_name)

image = get_base_image()

stub.image = image

class PipelineBuilder:
    def __init__(self, input_parameters : SDXLParameters) -> None:
        self.parameters = input_parameters

    def prepare_sdxl_t2i_pipeline(self):
        if not hasattr(self, "pipe"):
            
        return self
    
    def prepare_sdxl_i2i_pipeline(self):
        if self.parameters.image != None:
            if not hasattr(self, "pipe"):
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    sdxl_model_list.get(self.parameters.model), torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
                )
        return self

    def prepare_controlnet_pipeline(self):
        if self.parameters.controlnet_models != None:
            controlnet_list = []
            for controlnet


@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory)
class stableDiffusion:
    @build()
    @enter()
    def prepate_Safety_checker(self) -> None:
        self.safety_checker = SafetyChecker()


    def __init__(self, init_parameters : dict) -> None:
        init_parameters : InitParameters = InitParameters.model_validate(init_parameters)
        st = time.time()

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            sdxl_model_list.get(init_parameters.model), torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")
        if(init_parameters.lora_model):
            self.pipe.load_lora_weights(init_parameters.lora_model)

        self.refiner = get_refiner(self.pipe)

        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:

        parameters : SDXLText2ImageParameters = SDXLText2ImageParameters.model_validate(parameters)

        st = time.time()

        images = []

        for i in range(0, parameters.batch):
            image = self.pipe(
                prompt = parameters.prompt,
                height = parameters.height,
                width = parameters.width,
                negative_prompt = parameters.negative_prompt,
                num_inference_steps = parameters.num_inference_steps,
                denoising_end = 0.8,
                guidance_scale = parameters.guidance_scale,
                output_type="latent",
                cross_attention_kwargs={"scale": parameters.lora_scale},
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
