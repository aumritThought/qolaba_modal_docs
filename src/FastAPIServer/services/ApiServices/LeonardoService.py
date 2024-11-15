from src.data_models.ModalAppSchemas import SDXLText2ImageParameters
from src.utils.Globals import timing_decorator, make_request, prepare_response
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION
import time

class LeonardoText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        
    def make_api_request(self, parameters : SDXLText2ImageParameters) -> str:

        data = {
            "modelId": "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
            "contrast": 3.5,
            "prompt": parameters.prompt,
            "num_images": parameters.batch,
            "width": parameters.height,
            "height": parameters.width,
            "alchemy": False,
            "styleUUID": "556c1ee5-ec38-42e8-955a-1e82dad0ffa1",
            "enhancePrompt": False
        }
        headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {self.leonardo_api_key}',
            'content-type': 'application/json'
        }
        response = make_request(self.leonardo_image_generation_url, method="POST", json=data, headers=headers)
        id = response.json()["sdGenerationJob"]["generationId"]
        generation_status = None

        while(generation_status !="COMPLETE"):
    
            url = f"{self.leonardo_image_status_url}{id}"

            headers = {
                'accept': 'application/json',
                'authorization': f'Bearer {self.leonardo_api_key}'
            }
            response = make_request(url, "GET", headers=headers)
            generation_status = response.json()["generations_by_pk"]["status"]
            time.sleep(1)
        
        images = []

        for img in response.json()["generations_by_pk"]["generated_images"]:
            image_response = make_request(img["url"], "GET")
            images.append(image_response.content)
        return images

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        results = self.make_api_request(parameters)

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
    