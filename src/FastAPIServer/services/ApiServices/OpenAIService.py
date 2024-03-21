from src.data_models.ModalAppSchemas import DalleParameters, PromptParrotParameters
import threading
from src.utils.Globals import timing_decorator, upload_data_gcp, make_request, prepare_response
from openai import OpenAI
from src.utils.Constants import DALLE_SUPPORTED_HW, BASE_PROMPT_FOR_GENERATION
from typing import List
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION


class DalleText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key = self.openai_api_key)

    def make_dalle_api_request( self, prompt: str, Height_width: str, quality: str) -> str:
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
        threads: List[threading.Thread] = []
        results = []

        for _ in range(parameters.batch):
            thread = threading.Thread(
                target = lambda: results.append(
                    self.make_dalle_api_request(
                        parameters.prompt, Height_width, parameters.quality
                    )
                )
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0)


class PromptParrot(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key = self.openai_api_key)

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : PromptParrotParameters = PromptParrotParameters(**parameters)
        parameters.prompt = a = f"{BASE_PROMPT_FOR_GENERATION} \n{parameters.prompt}"
        modified_prompts = []
        for i in range(0, parameters.batch):
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": a}
                ],
                temperature=0.5
                )
            
            modified_prompts.append(response.choices[0].message.content)

        return prepare_response(modified_prompts, [False]*parameters.batch, 0, 0)

