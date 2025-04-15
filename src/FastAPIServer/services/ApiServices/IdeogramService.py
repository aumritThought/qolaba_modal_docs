from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters, IdeoGramRemixParameters
from src.utils.Globals import timing_decorator, make_request, prepare_response, convert_to_aspect_ratio, get_image_from_url
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IDEOGRAM_ASPECT_RATIO
import concurrent.futures, io, json

class IdeoGramText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the Ideogram service.
        
        Sets up the service by inheriting API credentials and endpoints
        from the parent IService class.
        """
        super().__init__()
        
    def make_api_request(self, parameters : IdeoGramText2ImageParameters) -> str:
        """
        Sends a request to the Ideogram API for image generation.
        
        This function constructs the appropriate API payload from the parameters,
        sends the request to the Ideogram API endpoint, and handles the response
        processing to extract the resulting image.
        
        Args:
            parameters: Configuration parameters for image generation including
                prompt, style settings, and format options
            
        Returns:
            str: The generated image data as bytes
        """
        payload = { 
            "image_request": {
                "prompt": parameters.prompt,
                "aspect_ratio": parameters.aspect_ratio,
                "model": "V_2_TURBO",
                "magic_prompt_option": parameters.magic_prompt_option,
                "negative_prompt" : parameters.negative_prompt,
                "style_type" : parameters.style_type
            } 
        }
        headers = {
            "Api-Key": self.ideogram_api_key,
            "Content-Type": "application/json"
        }
        response = make_request(self.ideogram_url, method="POST", json=payload, headers=headers)

        response = make_request(response.json()["data"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.
        
        This method validates and processes the input parameters, converts dimensions
        to Ideogram's supported aspect ratios, and creates multiple parallel generation
        tasks based on the batch size. The @timing_decorator tracks and adds execution
        time to the response.
        
        Args:
            parameters (dict): Request parameters for image generation
            
        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
                
        Raises:
            Exception: If the requested aspect ratio is not supported by Ideogram
        """
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)
        if(not (aspect_ratio in IDEOGRAM_ASPECT_RATIO.keys())):
            raise Exception("Invalid Height and width dimension")
        
        parameters.aspect_ratio = IDEOGRAM_ASPECT_RATIO[aspect_ratio]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
class IdeogramRemix(IService):
    def __init__(self) -> None:
        """
        Initializes the Ideogram service.
        
        Sets up the service by inheriting API credentials and endpoints
        from the parent IService class.
        """
        super().__init__()
        
    def make_api_request(self, parameters : IdeoGramRemixParameters) -> str:
        """
        Sends a request to the Ideogram API for image generation.
        
        This function constructs the appropriate API payload from the parameters,
        sends the request to the Ideogram API endpoint, and handles the response
        processing to extract the resulting image.
        
        Args:
            parameters: Configuration parameters for image generation including
                prompt, style settings, and format options
            
        Returns:
            str: The generated image data as bytes
        """
        payload = { 
            "image_request": json.dumps({
                "prompt": parameters.prompt,
                "aspect_ratio": parameters.aspect_ratio,
                "color_palette" : {
                    "name": parameters.color_palette
                },
                "image_weight" : int(parameters.strength*100), 
                "model": "V_2_TURBO",
                "magic_prompt_option": parameters.magic_prompt_option,
                "negative_prompt" : parameters.negative_prompt,
                "style_type" : parameters.style_type
            })
        }

        image = get_image_from_url(
            parameters.file_url
        )
        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")
        files = {
            'image_file': filtered_image.getvalue()
        }
        headers = {
            "Api-Key": self.ideogram_api_key,
        }

        response = make_request(self.ideogram_remix_url, method="POST", json_data=payload, headers=headers, files=files)

        response = make_request(response.json()["data"][0]["url"], "GET")
   
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.
        
        This method validates and processes the input parameters, converts dimensions
        to Ideogram's supported aspect ratios, and creates multiple parallel generation
        tasks based on the batch size. The @timing_decorator tracks and adds execution
        time to the response.
        
        Args:
            parameters (dict): Request parameters for image generation
            
        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
                
        Raises:
            Exception: If the requested aspect ratio is not supported by Ideogram
        """
        parameters : IdeoGramRemixParameters = IdeoGramRemixParameters(**parameters)

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)
        if(not (aspect_ratio in IDEOGRAM_ASPECT_RATIO.keys())):
            raise Exception("Invalid Height and width dimension")
        
        parameters.aspect_ratio = IDEOGRAM_ASPECT_RATIO[aspect_ratio]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)