from src.data_models.ModalAppSchemas import HairStyleParameters
from src.utils.Globals import timing_decorator, make_request, prepare_response, get_image_from_url
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION
import time, io

class AILabHairStyle(IService):
    def __init__(self) -> None:
        super().__init__()
        
    def make_api_request(self, parameters : HairStyleParameters) -> str:

        payload={'hair_style' : parameters.hair_style,'color' : parameters.color, "task_type" : "async", "image_size" : parameters.batch}

        filtered_image = io.BytesIO()
        parameters.file_url.save(filtered_image, "JPEG")
        image_data = filtered_image.getvalue()

        files=[
            ('image',('file',image_data,'application/octet-stream'))
        ]
        headers = {
            'ailabapi-api-key': self.hairstyle_api_key
        }

        response = make_request(self.hairstyle_url, "POST", headers=headers, json_data=payload, files=files)

        task_id = response.json()["task_id"]
        images = None

        while(images == None or images == []):
    
            url = f"{self.hairstyle_status_url}{task_id}"
            payload={}
            headers = {
                'ailabapi-api-key': self.hairstyle_api_key
            }
            response = make_request(url, "GET", headers=headers, json_data=payload)

            if("data" in  response.json().keys()):
                images = response.json()["data"]["images"]
            time.sleep(1)
        
        image_data = []
        for img in images:
            image_data.append(make_request(img, "GET").content)
        return image_data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : HairStyleParameters = HairStyleParameters(**parameters)

        parameters.file_url = get_image_from_url(parameters.file_url)

        results = self.make_api_request(parameters)

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
    