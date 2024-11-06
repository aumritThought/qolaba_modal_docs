from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters
from src.utils.Globals import timing_decorator, make_request, prepare_response, convert_to_aspect_ratio
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IDEOGRAM_ASPECT_RATIO
import concurrent.futures 


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