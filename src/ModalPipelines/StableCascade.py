from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, SDXLText2ImageParameters
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, extra_negative_prompt
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
import torch, time

stub_name = StubNames().stable_cascade_text_to_image

stub = Stub(stub_name)

def download_base_sdxl():
    StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
    StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)


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

        self.prior_pipe = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16).to("cuda")
        self.decoder_pipe = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16).to("cuda")
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()
        parameters : SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        images = []

        parameters.negative_prompt = parameters.negative_prompt + extra_negative_prompt

        for i in range(0, parameters.batch):
            prior_output = self.prior_pipe(
                prompt=parameters.prompt,
                height=parameters.height,
                width=parameters.width,
                negative_prompt=parameters.negative_prompt,
                guidance_scale=parameters.guidance_scale,
                num_images_per_prompt=1,
                num_inference_steps=parameters.num_inference_steps
            )

            image = self.decoder_pipe(
                image_embeddings=prior_output.image_embeddings.to(torch.float16),
                prompt=parameters.prompt,
                negative_prompt=parameters.negative_prompt,
                guidance_scale=0.0,
                output_type="pil",
                num_inference_steps=10
            ).images[0]

            torch.cuda.empty_cache()

            images.append(image)

        image_urls, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)