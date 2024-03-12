from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, BackGroundRemoval
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME
import torch, time, tempfile, os
from transparent_background import Remover

stub_name = StubNames().background_removal

stub = Stub(stub_name)

vol = Volume.persisted(VOLUME_NAME)

image = get_base_image()

def download_removal_model():
    Remover()

stub.image = image.run_function(download_removal_model)


@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)])
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()

        self.remover = Remover()
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()

        parameters : BackGroundRemoval = BackGroundRemoval(**parameters)

        parameters.file_url = get_image_from_url(parameters.file_url, resize = True)

        try:
            parameters.bg_img = get_image_from_url(parameters.bg_img, resize = True)
            parameters.bg_img = parameters.bg_img.convert("RGB")
            parameters.bg_img = parameters.bg_img.resize(parameters.file_url.size)
        except Exception as e:
            print(e)
            parameters.bg_img = None
        
        if parameters.blur == True:
            image = self.remover.process(parameters.file_url, type="blur")

        elif not (parameters.bg_img == None):
            temp_file_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            parameters.bg_img.save(temp_file_img, format="JPEG")

            image = self.remover.process(
                parameters.file_url, type=temp_file_img.name
            )  # use another image as a background
            print(os.path.exists(temp_file_img.name))
            try:
                print(temp_file_img.name)
                os.remove(temp_file_img.name)
                print("removed file")
            except:
                print("Error in removing file")
                pass
            print(os.path.exists(temp_file_img.name))
        elif parameters.bg_color == True:
            image = self.remover.process(
                parameters.file_url, type=str([parameters.r_color, parameters.g_color, parameters.b_color])
            )
        else:
            image = self.remover.process(parameters.file_url, type="rgba")
        torch.cuda.empty_cache()


        image_urls, has_nsfw_content = generate_image_urls([image], self.safety_checker, format = "PNG")

        self.runtime = time.time() - st

        return prepare_response(image_urls, has_nsfw_content, self.container_execution_time, self.runtime)