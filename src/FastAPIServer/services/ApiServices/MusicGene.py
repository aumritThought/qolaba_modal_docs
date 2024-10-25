from src.data_models.ModalAppSchemas import MusicGenParameters
from src.utils.Globals import timing_decorator, prepare_response, upload_data_gcp, make_request
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_AUDIO_EXTENSION
import requests

class MusicGen(IService):
    def __init__(self) -> None:
        super().__init__()

    def download_and_convert_to_base64(self, file_url) -> str:
        response = requests.get(file_url)
        
        if response.status_code == 200:
            file_content = response.content
            return file_content
        else:
            response.raise_for_status()


    @timing_decorator
    def remote(self, parameters : dict) -> dict:
        parameters : MusicGenParameters = MusicGenParameters(**parameters)
        payload = {
            "prompt": parameters.prompt
        }
        headers = {"Content-Type": "application/json",  "Authorization": f"Bearer {self.music_gen_api_key}" }

        response = make_request(self.music_gen_url, "POST", json=payload, headers=headers)

        if(response.status_code == 200):
            base64_str = self.download_and_convert_to_base64(response.json()[0]["file_url"])
        else:
            response.raise_for_status()

        Has_NSFW_Content = [False] * 1

        url = upload_data_gcp(base64_str, OUTPUT_AUDIO_EXTENSION)

        return prepare_response(url, Has_NSFW_Content, 0, 0)
        
 