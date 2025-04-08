import io, base64
from src.data_models.ModalAppSchemas import SDXL3APITextToImageParameters, SDXLAPITextToImageParameters, SDXLAPIImageToImageParameters, SDXLAPIInpainting, SDXL3APIImageToImageParameters
from src.utils.Globals import timing_decorator, make_request, get_image_from_url, prepare_response, invert_bw_image_color, convert_to_aspect_ratio
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, extra_negative_prompt, SDXL3_RATIO_LIST
from PIL.Image import Image as Imagetype
from transparent_background import Remover
import concurrent.futures 


class SDXLText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the SDXL service with appropriate API endpoints.
        
        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        self.api_host = self.stability_api
        self.api_key = self.stability_api_key
        self.engine_id = self.stability_engine_id
        self.url = f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image"

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
        parameters : SDXLAPITextToImageParameters = SDXLAPITextToImageParameters(**parameters)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        json_data = {
            "text_prompts": [
                                {
                                "text": parameters.prompt,
                                "weight": 1
                                },
                                {
                                "text": f"{parameters.negative_prompt}, {extra_negative_prompt}",
                                "weight": -1
                                }
                            ],
            "cfg_scale": parameters.guidance_scale,
            "clip_guidance_preset": "FAST_BLUE",
            "height": parameters.height,
            "width": parameters.width,
            "samples": parameters.batch,
            "steps": parameters.num_inference_steps,
            "style_preset": parameters.style_preset,
        }
        response = make_request(
            self.url, "POST", json=json_data, headers=headers
        )
        
        Has_NSFW_Content = [False] * parameters.batch

        data = response.json()
        image_urls = []
        for image in data["artifacts"]:
            image_urls.append(
                base64.b64decode(image["base64"])
            )
        return prepare_response(image_urls, Has_NSFW_Content, 0, 0,  OUTPUT_IMAGE_EXTENSION)


class SDXLImage2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the SDXL service with appropriate API endpoints.
        
        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        self.api_host = self.stability_api
        self.api_key = self.stability_api_key
        self.engine_id = self.stability_engine_id
        self.url = f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image"

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
        parameters : SDXLAPIImageToImageParameters = SDXLAPIImageToImageParameters(**parameters)
        image = get_image_from_url(
            parameters.file_url
        )

        image = image.resize((parameters.width, parameters.height)).convert("RGB")

        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        files = {"init_image": filtered_image.getvalue()}

        json_data = {
            "image_strength": 1 - parameters.strength,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": parameters.prompt,
            "text_prompts[0][weight]": 1,
            "text_prompts[1][text]": f"{parameters.negative_prompt}, {extra_negative_prompt}",
            "text_prompts[1][weight]": -1,
            "cfg_scale": parameters.guidance_scale,
            "samples": parameters.batch,
            # "steps": 30,
            "style_preset": parameters.style_preset,
        }

        response = make_request(
            self.url, "POST", json_data=json_data, headers=headers, files=files
        )

        Has_NSFW_Content = [False] * parameters.batch

        data = response.json()
        image_urls = []
        for image in data["artifacts"]:
            image_urls.append(
                base64.b64decode(image["base64"])
            )

        return prepare_response(image_urls, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)

class SDXLInpainting(IService):
    def __init__(self) -> None:
        """
        Initializes the SDXL service with appropriate API endpoints.
        
        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        self.api_key = self.stability_api_key
        self.url = self.stability_inpaint_url

    def generate_image(self, mask_image : Imagetype, image : Imagetype, prompt : str, negative_prompt : str) -> Imagetype:
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
        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        mask_filtered_image = io.BytesIO()

        mask_image.save(mask_filtered_image, "JPEG")

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        files={
                "image": filtered_image.getvalue(),
                "mask": mask_filtered_image.getvalue()
        }

        json_data = {
            "prompt": prompt,
            "output_format": "jpeg",
            "negative_prompt" : negative_prompt + extra_negative_prompt,
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
        parameters : SDXLAPIInpainting = SDXLAPIInpainting(**parameters)
        image = get_image_from_url(
            parameters.file_url
        )

        mask : Imagetype = get_image_from_url(parameters.mask_url)

        mask = invert_bw_image_color(mask)

        mask = mask.resize((image.size))

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, mask, image, parameters.prompt, parameters.negative_prompt)
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


class SDXLReplaceBackground(IService):
    def __init__(self, remover : Remover) -> None:
        """
        Initializes the SDXL service with appropriate API endpoints.
        
        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        self.api_key = self.stability_api_key
        self.url = self.stability_inpaint_url
        self.remover = remover
    
    def generate_image(self, mask_image : Imagetype, image : Imagetype, prompt : str, negative_prompt : str) -> Imagetype:
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
        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        mask_filtered_image = io.BytesIO()

        mask_image.save(mask_filtered_image, "JPEG")

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        files={
                "image": filtered_image.getvalue(),
                "mask": mask_filtered_image.getvalue()
        }

        json_data = {
            "prompt": prompt,
            "output_format": "jpeg",
            "negative_prompt" : negative_prompt + extra_negative_prompt,
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
        parameters : SDXLAPIInpainting = SDXLAPIInpainting(**parameters)
        image = get_image_from_url(
            parameters.file_url
        )

        mask = self.remover.process(image, type="rgba")

        mask = mask.getchannel('A')

        mask = invert_bw_image_color(mask)

        mask = mask.resize((image.size))

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, mask, image, parameters.prompt, parameters.negative_prompt)
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
    

class SDXL3Image2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the SDXL service with appropriate API endpoints.
        
        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        self.api_key = self.stability_api_key
        self.url = self.sdxl3_url


    def generate_image(self, parameters : SDXL3APIImageToImageParameters, model : str) -> str:
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
        image = get_image_from_url(
            parameters.file_url
        )

        image = image.convert("RGB")

        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        files = {"image": filtered_image.getvalue()}


        headers={
            "authorization": f"Bearer {self.api_key}",
            "accept": "application/json"
        }
                        
        json_data = {
            "prompt": parameters.prompt,
            "model" : model,
            "mode" : "image-to-image",
            "negative_prompt": parameters.negative_prompt + extra_negative_prompt,
            "strength" : parameters.strength
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
        parameters : SDXL3APIImageToImageParameters = SDXL3APIImageToImageParameters(**parameters)

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
    