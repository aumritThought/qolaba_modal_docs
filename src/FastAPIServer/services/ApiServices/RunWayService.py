from src.data_models.ModalAppSchemas import RunwayImage2VideoParameters
from src.utils.Globals import timing_decorator, prepare_response, make_request, get_image_from_url
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_VIDEO_EXTENSION, RUNWAY_ERROR, RUNWAY_ERROR_MSG, RUNWAY_POSITION_ERROR_MSG
import concurrent.futures 
from src.utils.Globals import timing_decorator, make_request, prepare_response
from runwayml import RunwayML
import io, base64

class RunwayImage2Video(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = RunwayML(api_key=self.runway_api_key)

    def download_and_convert_to_base64(self, url : str) -> str:
        image = get_image_from_url(url)

        base64_image = io.BytesIO()

        image.save(base64_image, "JPEG")

        img_str = base64.b64encode(base64_image.getvalue())
        
        base64_string = img_str.decode('utf-8')
        
        return f"data:image/jpg;base64,{base64_string}"

    def generate_image(self, parameters : RunwayImage2VideoParameters) -> str:

        if(len(parameters.file_url) > 0):
            parameters.file_url[0].uri = self.download_and_convert_to_base64(parameters.file_url[0].uri)

            if(len(parameters.file_url) > 1):
                parameters.file_url[1].uri = self.download_and_convert_to_base64(parameters.file_url[1].uri)
                if(parameters.file_url[0].position == parameters.file_url[1].position):
                    raise Exception(RUNWAY_ERROR, RUNWAY_POSITION_ERROR_MSG)
        else:
            raise Exception(RUNWAY_ERROR, RUNWAY_ERROR_MSG)

        # print(parameters.file_url)
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
