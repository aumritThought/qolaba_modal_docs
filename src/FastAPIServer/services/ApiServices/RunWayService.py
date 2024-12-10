from src.data_models.ModalAppSchemas import RunwayImage2VideoParameters
from src.utils.Globals import timing_decorator, prepare_response, make_request
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_VIDEO_EXTENSION
import concurrent.futures 
from src.utils.Globals import timing_decorator, make_request, prepare_response
from runwayml import RunwayML

class RunwayImage2Video(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = RunwayML(api_key=self.runway_api_key)

    def generate_image(self, parameters : RunwayImage2VideoParameters) -> str:

        task = self.client.image_to_video.create(
            model='gen3a_turbo',
            prompt_image=parameters.model_dump()["file_url"],
            prompt_text=parameters.prompt,
            duration=parameters.duration,
            ratio=parameters.aspect_ratio
        )
        task = self.client.tasks.retrieve(id=task.id)
        while(task.status != "SUCCEEDED"):
            task = self.client.tasks.retrieve(id=task.id)
            if(task.status in ["FAILED", "CANCELLED"]):
                print(task.status, task.model_dump())
                raise Exception("Video generation failed")
        
        response = make_request(task.output[0], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : RunwayImage2VideoParameters = RunwayImage2VideoParameters(**parameters)
        

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)
