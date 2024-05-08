from src.data_models.ModalAppSchemas import DalleParameters
from src.utils.Globals import timing_decorator, upload_data_gcp, make_request, prepare_response
from openai import OpenAI
from src.utils.Constants import DALLE_SUPPORTED_HW
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION
import concurrent.futures 



class DalleText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key = self.openai_api_key)

    def make_dalle_api_request(self, prompt: str, Height_width: str, quality: str) -> str:
        response = self.client.images.generate(
            model="dall-e-3", prompt=prompt, size=Height_width, quality=quality, n=1
        )

        response = make_request(response.data[0].url, "GET")

        return upload_data_gcp(response.content, OUTPUT_IMAGE_EXTENSION)

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : DalleParameters = DalleParameters(**parameters)

        Height_width = f"{parameters.width}x{parameters.height}"

        if not (Height_width in DALLE_SUPPORTED_HW):
            raise Exception(
                f"Height and width should be {str(DALLE_SUPPORTED_HW)} for Dalle"
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_dalle_api_request, parameters.prompt, Height_width, parameters.quality)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0)

