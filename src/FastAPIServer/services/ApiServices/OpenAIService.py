from src.data_models.ModalAppSchemas import DalleParameters, OpenAITTSParameters, TTSOutput, NSFWSchema
from src.utils.Globals import timing_decorator, make_request, prepare_response
from openai import OpenAI
from src.utils.Constants import DALLE_SUPPORTED_HW
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, TTS_CHAR_COST, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG, COPYRIGHT_DETECTION_FUNCTION_CALLING_SCHEMA
import concurrent.futures 
from typing import Iterator
import json, base64
import google.auth
import google.auth.transport.requests
from google import genai
from src.utils.Constants import google_credentials_info
from src.utils.Globals import get_image_from_url
from src.data_models.ModalAppSchemas import NSFWSchema

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
    

class OpenAIImageCheck(IService):
    def __init__(self) -> None:
        super().__init__()
        self.function_calling_schema = COPYRIGHT_DETECTION_FUNCTION_CALLING_SCHEMA
        credentials, project_id = google.auth.load_credentials_from_dict(google_credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        self.client = genai.Client(vertexai=True, project="marine-potion-404413", credentials = credentials, location="us-central1")

    def remote(self, image_url: str) -> bool:
        response = self.client.models.generate_content(
            contents=["""Analyze if an image contains any NSFW content.

            Return true if the image contains any kind of NSFW content, including but not limited to:

            Visible private parts,
            Pornographic material,
            Explicit sexual activities,
            Nudity,
            Suggestive or provocative imagery,
            Activities like kissing or intimate physical contact with sexual undertones.

            Return false for any other category apart from NSFW content.
            
            You must return false for any content that does not fall into the NSFW category.""", get_image_from_url(image_url)],
            model="gemini-2.0-flash",
            config={
                'response_mime_type': 'application/json',
                'response_schema': NSFWSchema,
            },
        )
        return response.parsed.NSFW_content
    