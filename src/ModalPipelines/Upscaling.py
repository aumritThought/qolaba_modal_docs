from modal import App, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, InitParameters, UpscaleParameters
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, ULTRASHARP_MODEL, OUTPUT_IMAGE_EXTENSION
import torch, time, cv2
import numpy as np
from PIL import Image
from imaginairy.vendored.basicsr.rrdbnet_arch import RRDBNet
from imaginairy.vendored.realesrgan import RealESRGANer

stub_name = StubNames().ultrasharp_upscaler

app = App(stub_name)

vol = Volume.from_name(VOLUME_NAME)

image = get_base_image().run_commands("pip install imaginAIry")

app.image = image



@app.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)],
          concurrency_limit=stub_dictionary[stub_name].num_containers)
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )

        self.upsampler = RealESRGANer(
            scale = 4,
            model_path = ULTRASHARP_MODEL,
            model = model,
            tile = 512,
            device = "cuda",
            tile_pad = 50,
            half = True,
        )

        self.upsampler.device = torch.device("cuda")
        self.upsampler.model.to("cuda")
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()
        parameters : UpscaleParameters = UpscaleParameters(**parameters)

        if(not isinstance(parameters.file_url, Image.Image)):
            parameters.file_url = get_image_from_url(parameters.file_url)

        parameters.file_url = parameters.file_url.convert("RGB")

        np_img = np.array(parameters.file_url, dtype=np.uint8)
        upsampler_output, img_mode = self.upsampler.enhance(
                np_img[:, :, ::-1]
            )
        
        if(parameters.scale==2):
            upsampler_output = cv2.resize(upsampler_output, (int(upsampler_output.shape[1]/2), int(upsampler_output.shape[0]/2)))
        if(parameters.scale==8):
            upsampler_output = cv2.resize(upsampler_output, (int(upsampler_output.shape[1]/2), int(upsampler_output.shape[0]/2)))
            image = Image.fromarray(upsampler_output[:, :, ::-1], mode=img_mode)
            np_img = np.array(image, dtype=np.uint8)
            upsampler_output, img_mode = self.upsampler.enhance(
                    np_img[:, :, ::-1]
                )

        images = [Image.fromarray(upsampler_output[:, :, ::-1], mode=img_mode)]
        
        images, has_nsfw_content = generate_image_urls(images, self.safety_checker, parameters.check_nsfw)

        self.runtime = time.time() - st
        return prepare_response(images, has_nsfw_content, self.container_execution_time, self.runtime, OUTPUT_IMAGE_EXTENSION)