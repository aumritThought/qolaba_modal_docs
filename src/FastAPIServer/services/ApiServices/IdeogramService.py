from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters
from src.utils.Globals import timing_decorator, make_request, prepare_response, convert_to_aspect_ratio, get_image_from_url, invert_bw_image_color
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IDEOGRAM_ASPECT_RATIO
import concurrent.futures, io
from PIL.Image import Image as Imagetype
from transparent_background import Remover

class IdeoGramText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        
    def make_api_request(self, parameters : IdeoGramText2ImageParameters) -> str:

        payload = { 
            "image_request": {
                "prompt": parameters.prompt,
                "aspect_ratio": parameters.aspect_ratio,
                "model": "V_2_TURBO",
                "magic_prompt_option": parameters.magic_prompt_option,
                "negative_prompt" : parameters.negative_prompt,
                "style_type" : parameters.style_type
            } 
        }
        headers = {
            "Api-Key": self.ideogram_api_key,
            "Content-Type": "application/json"
        }
        response = make_request(self.ideogram_url, method="POST", json=payload, headers=headers)

        response = make_request(response.json()["data"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)
        if(not (aspect_ratio in IDEOGRAM_ASPECT_RATIO)):
            raise Exception("Invalid Height and width dimension")
        
        parameters.aspect_ratio = IDEOGRAM_ASPECT_RATIO[aspect_ratio]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
class IdeogramInpainting(IService):
    def __init__(self) -> None:
        super().__init__()

    def generate_image(self, parameters : IdeoGramText2ImageParameters) -> Imagetype:
        filtered_image = io.BytesIO()
        parameters.file_url.save(filtered_image, "JPEG")

        mask_filtered_image = io.BytesIO()

        parameters.mask_url.save(mask_filtered_image, "JPEG")

        headers = {
            "Api-Key": self.ideogram_api_key
        }
        files={
                "image_file": filtered_image.getvalue(),
                "mask": mask_filtered_image.getvalue()
        }

        payload = {
            "prompt": parameters.prompt,
            "model": "V_2_TURBO",
            "magic_prompt_option": parameters.magic_prompt_option,
            "style_type": parameters.style_type,
            "magic_prompt_option": parameters.magic_prompt_option,
        }

        response = make_request(self.ideogram_edit_url, method="POST", json_data=payload, headers=headers, files=files)

        response = make_request(response.json()["data"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)
        parameters.file_url = get_image_from_url(
            parameters.file_url
        )

        parameters.mask_url = get_image_from_url(parameters.mask_url)

        parameters.mask_url = parameters.mask_url.resize((parameters.file_url.size))

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class IdeogramReplaceBackground(IService):
    def __init__(self, remover : Remover) -> None:
        super().__init__()
        self.remover = remover        

    def generate_image(self, parameters : IdeoGramText2ImageParameters) -> Imagetype:
        filtered_image = io.BytesIO()
        parameters.file_url.save(filtered_image, "JPEG")

        mask_filtered_image = io.BytesIO()

        parameters.mask_url.save(mask_filtered_image, "JPEG")

        headers = {
            "Api-Key": self.ideogram_api_key
        }
        files={
                "image_file": filtered_image.getvalue(),
                "mask": mask_filtered_image.getvalue()
        }

        payload = {
            "prompt": parameters.prompt,
            "model": "V_2_TURBO",
            "magic_prompt_option": parameters.magic_prompt_option,
            "style_type": parameters.style_type,
            "magic_prompt_option": parameters.magic_prompt_option,
        }

        response = make_request(self.ideogram_edit_url, method="POST", json_data=payload, headers=headers, files=files)

        response = make_request(response.json()["data"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)
        parameters.file_url = get_image_from_url(
            parameters.file_url
        )
        parameters.mask_url = self.remover.process(parameters.file_url, type="rgba")

        parameters.mask_url = parameters.mask_url.getchannel('A')

        parameters.mask_url = parameters.mask_url.resize((parameters.file_url.size))

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
    