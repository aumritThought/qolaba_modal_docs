from modal import App, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, UpscaleParameters
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, OUTPUT_IMAGE_EXTENSION
import torch, time
from PIL import Image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

def download_models():
    controlnet = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        torch_dtype=torch.bfloat16
    )
    FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    )

MEAN_HEIGHT = 1024

stub_name = StubNames().ultrasharp_upscaler

app = App(stub_name)

vol = Volume.from_name(VOLUME_NAME)

image = get_base_image().run_commands("pip install accelerate").run_function(download_models, secrets = [Secret.from_name(SECRET_NAME)], gpu = "a10g")

app.image = image



@app.cls(gpu = "a100", 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)],
          concurrency_limit=stub_dictionary[stub_name].num_containers)
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()

        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    def resize_image(self, img: Image) -> Image:
        img = img.resize((64 * round(img.size[0] / 64), 64 * round(img.size[1] / 64)))
        if(img.size[0]*img.size[1] > MEAN_HEIGHT*MEAN_HEIGHT):
            # if ( img.size[0] > MAX_HEIGHT or img.size[0] < MIN_HEIGHT or img.size[1] > MAX_HEIGHT or img.size[1] < MIN_HEIGHT):
                if img.size[1] >= img.size[0]:
                    height = MEAN_HEIGHT
                    width = ((int(img.size[0]* MEAN_HEIGHT/ img.size[1]))// 64) * 64
                else:
                    width = MEAN_HEIGHT
                    height = ((int(img.size[1]*MEAN_HEIGHT/ img.size[0]))// 64) * 64

                img = img.resize((width, height))
        return img

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()
        parameters : UpscaleParameters = UpscaleParameters(**parameters)

        if(not isinstance(parameters.file_url, Image.Image)):
            parameters.file_url = get_image_from_url(parameters.file_url)

        parameters.file_url = parameters.file_url.convert("RGB")
        parameters.file_url = self.resize_image(parameters.file_url)
        w, h = parameters.file_url.size
        
        control_image = parameters.file_url.resize((w * 2, h * 2)) #it is not possible to go beyond 2x. getting CUDA error for flux

        image = self.pipe(
            prompt="", 
            control_image=control_image,
            controlnet_conditioning_scale=parameters.strength,
            num_inference_steps=20, 
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]
        image

        images = [image]
        
        images, has_nsfw_content = generate_image_urls(images, self.safety_checker, parameters.check_nsfw)

        self.runtime = time.time() - st
        return prepare_response(images, has_nsfw_content, self.container_execution_time, self.runtime, OUTPUT_IMAGE_EXTENSION)