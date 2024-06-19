from src.data_models.ModalAppSchemas import DalleParameters, OpenAITTSParameters, TTSOutput
from src.utils.Globals import timing_decorator, make_request, prepare_response
from openai import OpenAI
from src.utils.Constants import DALLE_SUPPORTED_HW
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, TTS_CHAR_COST, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG
import concurrent.futures 
from typing import Iterator
import json, base64


class DalleText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key = self.openai_api_key)

    def make_dalle_api_request(self, prompt: str, Height_width: str, quality: str) -> str:
        try:
            response = self.client.images.generate(
                model="dall-e-3", prompt=prompt, size=Height_width, quality=quality, n=1
            )
        except Exception as error:
            if("content_policy_violation" in str(error)):
                raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)
            else:
                raise Exception(str(error))
            

        response = make_request(response.data[0].url, "GET")

        return response.content

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

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class OpenAITexttoSpeech(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key = self.openai_api_key)

    def remote(self, parameters: OpenAITTSParameters) -> Iterator:
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=parameters.prompt,
        )

        for chunk in response.iter_bytes(chunk_size=4096):
            encoded_data = base64.b64encode(chunk).decode('utf-8')

            data = TTSOutput(output= encoded_data, cost=0)
            yield json.dumps(data.model_dump())
        data = TTSOutput(output= None, cost=len(parameters.prompt)*TTS_CHAR_COST)
        yield json.dumps(data.model_dump())
    
