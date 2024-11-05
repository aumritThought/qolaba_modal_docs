from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters
from src.utils.Globals import timing_decorator, prepare_response, convert_to_aspect_ratio
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IMAGEGEN_ASPECT_RATIOS, google_credentials_info, IMAGEGEN_ERROR, IMAGEGEN_ERROR_MSG
import concurrent.futures 
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import google.auth
import google.auth.transport.requests

class ImageGenText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        credentials, project_id = google.auth.load_credentials_from_dict(google_credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        vertexai.init(project="marine-potion-404413", credentials = credentials)
        self.generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")


    def make_api_request(self, parameters : IdeoGramText2ImageParameters) -> str:
        image = self.generation_model.generate_images(
            prompt=parameters.prompt,
            number_of_images=1,
            aspect_ratio=parameters.aspect_ratio,
            safety_filter_level="block_some"
        )
        if(len(image.images)==0):
            raise Exception(IMAGEGEN_ERROR, IMAGEGEN_ERROR_MSG)
        return image.images[0]._image_bytes

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)

        parameters.aspect_ratio = convert_to_aspect_ratio(parameters.height, parameters.width)
        if(not (parameters.aspect_ratio in IMAGEGEN_ASPECT_RATIOS)):
            raise Exception("Invalid Height and width dimension")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)