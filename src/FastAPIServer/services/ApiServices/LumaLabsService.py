from src.data_models.ModalAppSchemas import LumaLabsText2VideoParameters
from src.utils.Globals import timing_decorator, prepare_response, make_request
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_VIDEO_EXTENSION
import concurrent.futures 
from src.utils.Globals import timing_decorator, make_request, prepare_response
from lumaai import LumaAI

class LumaText2Video(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = LumaAI(auth_token=self.lumalabs_api_key)

    def generate_image(self, parameters : LumaLabsText2VideoParameters) -> str:
        generation = self.client.generations.create(
            prompt=parameters.prompt,
            aspect_ratio=parameters.aspect_ratio,
            loop=parameters.loop
        )
        completed = False
        while not completed:
            generation = self.client.generations.get(id=generation.id)
            if generation.state == "completed":
                completed = True
            elif generation.state == "failed":
                raise RuntimeError(f"Generation failed: {generation.failure_reason}")
            
        
        response = make_request(generation.assets.video, "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : LumaLabsText2VideoParameters = LumaLabsText2VideoParameters(**parameters)
        

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)


class LumaImage2Video(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = LumaAI(auth_token=self.lumalabs_api_key)

    def generate_image(self, parameters : LumaLabsText2VideoParameters) -> str:
        if(len(parameters.file_url) == 1):
            if(parameters.file_url[0].position == "first"):
                frame = "frame0"
            else:
                frame = "frame1"
            keyframes = {
                frame : {
                    "type": "image",
                    "url": parameters.file_url[0].uri
                }
            }
        else:
            if(parameters.file_url[0].position == "first"):
                first_image = parameters.file_url[0].uri
                second_image = parameters.file_url[1].uri
            else:
                first_image = parameters.file_url[1].uri
                second_image = parameters.file_url[0].uri
            keyframes = {
                "frame0": {
                    "type": "image",
                    "url": first_image
                },
                "frame1": {
                    "type": "image",
                    "url": second_image
                }
            }
        generation = self.client.generations.create(
            prompt=parameters.prompt,
            aspect_ratio=parameters.aspect_ratio,
            loop=parameters.loop,
            keyframes=keyframes
        )
        completed = False
        while not completed:
            generation = self.client.generations.get(id=generation.id)
            if generation.state == "completed":
                completed = True
            elif generation.state == "failed":
                raise RuntimeError(f"Generation failed: {generation.failure_reason}")
            
        
        response = make_request(generation.assets.video, "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : LumaLabsText2VideoParameters = LumaLabsText2VideoParameters(**parameters)
        

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)
