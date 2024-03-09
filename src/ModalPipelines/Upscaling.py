from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, InitParameters, UpscaleParameters
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, ULTRASHARP_MODEL
import torch, time
import numpy as np
from PIL import Image
from imaginairy.vendored.basicsr.rrdbnet_arch import RRDBNet
from imaginairy.vendored.realesrgan import RealESRGANer

stub_name = StubNames().ultrasharp_upscaler

stub = Stub(stub_name)

vol = Volume.persisted(VOLUME_NAME)

image = get_base_image()

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

        parameters.image = get_image_from_url(parameters.image, resize = True)

        parameters.image = parameters.image.convert("RGB")

        np_img = np.array(parameters.image, dtype=np.uint8)
        upsampler_output, img_mode = self.upsampler.enhance(
            np_img[:, :, ::-1]
        )

        images = [Image.fromarray(upsampler_output[:, :, ::-1], mode=img_mode)]

        image_urls, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)