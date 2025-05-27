import concurrent.futures
import io
import json

from src.data_models.ModalAppSchemas import (
    IdeoGramRemixParameters,
    IdeoGramText2ImageParameters,
)
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import IDEOGRAM_ASPECT_RATIO, OUTPUT_IMAGE_EXTENSION
from src.utils.Globals import (
    convert_to_aspect_ratio,
    get_image_from_url,
    make_request,
    prepare_response,
    timing_decorator,
)


class IdeoGramText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the Ideogram service.

        Sets up the service by inheriting API credentials and endpoints
        from the parent IService class.
        """
        super().__init__()

    def make_api_request(self, parameters: IdeoGramText2ImageParameters) -> str:
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
            
                "prompt": parameters.prompt,
                "aspect_ratio": parameters.aspect_ratio,
                "model": "V_3",
                "magic_prompt_option": parameters.magic_prompt_option,
                "negative_prompt": parameters.negative_prompt,
                "style_type": parameters.style_type,
            
        }
        headers = {"Api-Key": self.ideogram_api_key, "Content-Type": "application/json"}
        response = make_request(
            self.ideogram_url, method="POST", json=payload, headers=headers
        )

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
        parameters: IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(
            **parameters
        )

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)
        if aspect_ratio not in IDEOGRAM_ASPECT_RATIO.keys():
            raise Exception("Invalid Height and width dimension")

        parameters.aspect_ratio = IDEOGRAM_ASPECT_RATIO[aspect_ratio]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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

    def make_api_request(self, parameters: IdeoGramRemixParameters) -> str:
        payload = {
            "prompt": parameters.prompt,
            "aspect_ratio": parameters.aspect_ratio,
            "image_weight": int(parameters.strength * 100),
        }
        
        if hasattr(parameters, 'magic_prompt_option') and parameters.magic_prompt_option:
             payload["magic_prompt"] = parameters.magic_prompt_option
        if hasattr(parameters, 'negative_prompt') and parameters.negative_prompt:
            payload["negative_prompt"] = parameters.negative_prompt
        if hasattr(parameters, 'style_type') and parameters.style_type:
            payload["style_type"] = parameters.style_type
        if hasattr(parameters, 'color_palette') and parameters.color_palette:
            payload["color_palette"] = json.dumps({"name": parameters.color_palette})
        
        image_data = get_image_from_url(parameters.file_url)
        
        save_format = "JPEG"
        if image_data and image_data.format:
            image_format_upper = image_data.format.upper()
            if image_format_upper in ["JPEG", "PNG", "WEBP"]:
                save_format = image_format_upper
        
        filtered_image = io.BytesIO()
        image_data.save(filtered_image, format=save_format)
        
        files = {"image": ("input_image." + save_format.lower(), filtered_image.getvalue())}
        
        headers = {"Api-Key": self.ideogram_api_key}

        response = make_request(
            self.ideogram_remix_url,
            method="POST",
            json_data=payload, 
            headers=headers,
            files=files,
        )

        response = make_request(response.json()['data'][0]['url'], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters_model: IdeoGramRemixParameters = IdeoGramRemixParameters(**parameters)

        aspect_ratio = convert_to_aspect_ratio(parameters_model.width, parameters_model.height)
        if aspect_ratio not in IDEOGRAM_ASPECT_RATIO.keys():
            raise Exception("Invalid Height and width dimension")

        parameters_model.aspect_ratio = IDEOGRAM_ASPECT_RATIO[aspect_ratio]
        
        results_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _ in range(parameters_model.batch): # Use parameters_model.batch
                future = executor.submit(self.make_api_request, parameters_model) # Pass parameters_model
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                results_list.append(future.result())


        Has_NSFW_Content = [False] * parameters_model.batch # Use parameters_model.batch

        return prepare_response(results_list, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
