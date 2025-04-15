import io, base64
from src.data_models.ModalAppSchemas import SDXL3APITextToImageParameters, SDXLAPITextToImageParameters, SDXLAPIImageToImageParameters, SDXLAPIInpainting, SDXL3APIImageToImageParameters
from src.utils.Globals import timing_decorator, make_request, get_image_from_url, prepare_response, invert_bw_image_color, convert_to_aspect_ratio
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, extra_negative_prompt, SDXL3_RATIO_LIST
from PIL.Image import Image as Imagetype
from transparent_background import Remover
import concurrent.futures 

class SDXL3Text2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the SDXL service with appropriate API endpoints.
        
        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        self.api_key = self.stability_api_key
        self.url = self.sdxl3_url


    def generate_image(self, parameters : SDXL3APITextToImageParameters, model : str) -> str:
        """
        Processes an individual image generation request through the SDXL API.
        
        This function constructs the appropriate API payload from the parameters,
        sends the request to the specified API endpoint, and processes the response
        including NSFW content detection and result formatting.
        
        Args:
            parameters: Configuration parameters specific to the operation type
            *args: Additional arguments specific to the operation type
            
        Returns:
            list: A list containing the generated image data and NSFW flag
            
        Raises:
            Exception: If the generated content is flagged as NSFW or an API error occurs
        """
        headers={
            "authorization": f"Bearer {self.api_key}",
            "accept": "application/json"
        }
                

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)
        if(aspect_ratio == "7:3"):
            aspect_ratio = "21:9"
        if(aspect_ratio == "3:7"):
            aspect_ratio = "9:21"
        if(not (aspect_ratio in SDXL3_RATIO_LIST)):
            raise Exception("Invalid Height and width dimension")
        
        files={"none": ''}

        json_data = {
            "prompt": parameters.prompt,
            "model" : model,
            "mode" : "text-to-image",
            "aspect_ratio": aspect_ratio,
            "negative_prompt": parameters.negative_prompt + extra_negative_prompt,
        }

        response = make_request(
            self.url, "POST", json_data=json_data, headers=headers, files=files
        )

        has_NSFW = False
        if(response.json()["finish_reason"] == "CONTENT_FILTERED"):
            has_NSFW = True

        image_url = None
        if(not(has_NSFW)):
            image_url = base64.b64decode(response.json()["image"])
        return [image_url, has_NSFW]

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.
        
        This method validates the input parameters, prepares any required resources
        (like input images or masks), and creates multiple parallel generation tasks
        based on the batch size. The @timing_decorator tracks and adds execution
        time to the response.
        
        Args:
            parameters (dict): Request parameters for the specific operation
            
        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
                
        Raises:
            Exception: If parameter validation fails or the API returns errors
        """
        parameters : SDXL3APITextToImageParameters = SDXL3APITextToImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters, "sd3")
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = []
        image_urls = []
        for i in range(0, len(results)):
            Has_NSFW_Content.append(results[i][1])
            if(results[i][0] != None):
                image_urls.append(
                    results[i][0]
                )
        return prepare_response(image_urls, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
