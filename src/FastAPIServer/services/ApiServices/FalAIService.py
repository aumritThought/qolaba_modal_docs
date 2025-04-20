import base64
import concurrent.futures

import fal_client
import requests  # Ensure requests is imported
from loguru import logger  # Ensure logger is imported
from PIL.Image import Image as Imagetype
from pydantic import ValidationError  # Ensure ValidationError is imported
from transparent_background import Remover

from src.data_models.ModalAppSchemas import (
    FluxImage2ImageParameters,
    FluxText2ImageParameters,
    IdeoGramText2ImageParameters,
    Kling2MasterParameters,
    OmnigenParameters,
    RecraftV3Text2ImageParameters,
    SDXLText2ImageParameters,
    Veo2Parameters,
)  # Import the specific schema for Veo2
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import (
    IMAGE_GENERATION_ERROR,
    NSFW_CONTENT_DETECT_ERROR_MSG,
    OUTPUT_IMAGE_EXTENSION,
    OUTPUT_VIDEO_EXTENSION,
)
from src.utils.Globals import (
    get_image_from_url,
    invert_bw_image_color,
    make_request,
    prepare_response,
    simple_boundary_blur,
    timing_decorator,
    upload_data_gcp,
)


class FalAIFluxProText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxText2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "safety_tolerance": "2",
            "sync_mode": True,
            "enable_safety_checker": True,
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1.1",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)

        data = base64.b64decode(encoded)

        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxDevText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxText2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "safety_tolerance": "2",
            "sync_mode": True,
            "enable_safety_checker": True,
        }
        result = fal_client.subscribe(
            "fal-ai/flux/dev",
            arguments=input,
            with_logs=False,
        )

        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)

        data = base64.b64decode(encoded)

        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxschnellText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxText2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "num_inference_steps": 12,
            "num_images": 1,
            "enable_safety_checker": True,
        }
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxDevImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxImage2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "image_url": parameters.file_url,
            "prompt": parameters.prompt,
            "safety_tolerance": "2",
            "sync_mode": True,
            "enable_safety_checker": True,
            "strength": parameters.strength,
        }
        result = fal_client.subscribe(
            "fal-ai/flux/dev/image-to-image",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)

        data = base64.b64decode(encoded)

        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIRefactorV3Text2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: RecraftV3Text2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "style": parameters.style,
        }
        result = fal_client.subscribe(
            "fal-ai/recraft-v3",
            arguments=input,
            with_logs=False,
        )

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: RecraftV3Text2ImageParameters = RecraftV3Text2ImageParameters(
            **parameters
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAISD35LargeText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: SDXLText2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": parameters.guidance_scale,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True,
        }
        result = fal_client.subscribe(
            "fal-ai/stable-diffusion-v35-large",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)

        data = base64.b64decode(encoded)

        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAISD35LargeTurboText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: SDXLText2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "num_inference_steps": 10,
            "guidance_scale": 3,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True,
        }
        result = fal_client.subscribe(
            "fal-ai/stable-diffusion-v35-large/turbo",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)

        data = base64.b64decode(encoded)

        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAISD35MediumText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: SDXLText2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": parameters.guidance_scale,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True,
        }
        result = fal_client.subscribe(
            "fal-ai/stable-diffusion-v35-medium",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)

        data = base64.b64decode(encoded)

        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFlux3Inpainting(IService):
    def __init__(self) -> None:
        super().__init__()

    def generate_image(self, parameters: IdeoGramText2ImageParameters) -> Imagetype:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_url": parameters.file_url,
            "mask_url": parameters.mask_url,
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/fill",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(
            **parameters
        )
        original_image = get_image_from_url(parameters.file_url, rs_image=False)

        parameters.mask_url = get_image_from_url(parameters.mask_url)

        parameters.mask_url = invert_bw_image_color(parameters.mask_url)

        parameters.mask_url = simple_boundary_blur(parameters.mask_url)

        parameters.mask_url = parameters.mask_url.resize((original_image.size))

        parameters.mask_url = upload_data_gcp(parameters.mask_url, "webp")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFlux3ReplaceBackground(IService):
    def __init__(self, remover: Remover) -> None:
        super().__init__()
        self.remover = remover

    def generate_image(self, parameters: IdeoGramText2ImageParameters) -> Imagetype:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "prompt": parameters.prompt,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_url": parameters.file_url,
            "mask_url": parameters.mask_url,
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/fill",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(
            **parameters
        )
        original_image = get_image_from_url(parameters.file_url, rs_image=False)
        parameters.mask_url = self.remover.process(original_image, type="rgba")

        parameters.mask_url = parameters.mask_url.getchannel("A")

        parameters.mask_url = invert_bw_image_color(parameters.mask_url)

        parameters.mask_url = simple_boundary_blur(parameters.mask_url)

        parameters.mask_url = parameters.mask_url.resize((original_image.size))

        parameters.mask_url = upload_data_gcp(parameters.mask_url, "webp")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxProRedux(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxImage2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "image_url": parameters.file_url,
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_prompt_strength": parameters.strength,
            "prompt": parameters.prompt,
            "image_size": {"width": parameters.height, "height": parameters.width},
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1.1/redux",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxProCanny(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxImage2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "control_image_url": parameters.file_url,
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": parameters.prompt,
        }

        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/canny",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxProDepth(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxImage2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "control_image_url": parameters.file_url,
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": parameters.prompt,
        }

        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/depth",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class OmnigenV1(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: OmnigenParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        if type(parameters.file_url) == str:
            parameters.file_url = [parameters.file_url]
        input = {
            "prompt": parameters.prompt,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 2.5,
            "img_guidance_scale": 1.6,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "input_image_urls": parameters.file_url,
        }

        result = fal_client.subscribe(
            "fal-ai/omnigen-v1",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: OmnigenParameters = OmnigenParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class FalAIFluxPulID(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters: FluxImage2ImageParameters) -> str:
        """
        Sends a request to the Fal.AI.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the Flux Pro API endpoint, and handles the response
        processing including base64 decoding of the resulting image.

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the generated content is flagged as NSFW
        """
        input = {
            "reference_image_url": parameters.file_url,
            "image_size": {"width": parameters.width, "height": parameters.height},
            "num_inference_steps": 20,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "enable_safety_checker": True,
            "id_weight": 1,
        }

        result = fal_client.subscribe(
            "fal-ai/flux-pulid",
            arguments=input,
            with_logs=False,
        )
        if sum(result["has_nsfw_concepts"]) == 1:
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format
        """
        parameters: FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class Veo2(IService):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Veo2 Service Initialized")

    def make_api_request(self, parameters: Veo2Parameters) -> bytes:
        """Sends a request to the Fal AI Veo2 API and retrieves the result."""
        logger.info(f"Veo2 Parameters received: {parameters}")

        # Determine model_id based on whether an image URL is provided
        if parameters.file_url:
            model_id = "fal-ai/veo2"
            arguments = {
                "prompt": parameters.prompt,
                "image_url": parameters.file_url,
                "duration": parameters.duration,
                "aspect_ratio": parameters.aspect_ratio,
                # Add other image-to-video specific args here
            }
        else:
            # Assuming a text-to-video endpoint exists if no image URL
            model_id = "fal-ai/veo2"  # Check Fal docs for exact ID
            arguments = {
                "prompt": parameters.prompt,
                "duration": parameters.duration,
                "aspect_ratio": parameters.aspect_ratio,
                # Add other text-to-video specific args here
            }
            logger.warning("Veo2 called without file_url, attempting text-to-video.")

        logger.info(f"Calling Fal API [{model_id}] with payload: {arguments}")

        try:
            # Submit the request and get the handle
            request_handle = fal_client.submit(model_id, arguments=arguments)
            logger.info(
                f"Fal Veo2 API request submitted. Handle type: {type(request_handle)}"
            )

            # --- Fetch the final result from the handle ---
            final_response = request_handle.get()

            logger.info(
                f"Fal Veo2 API final result fetched. Type: {type(final_response)}"
            )
            logger.debug(
                f"Fal Veo2 API final result content: {final_response}"
            )  # Log the actual result

            # --- Process the final_response dictionary ---
            if not isinstance(final_response, dict):
                logger.error(
                    f"Expected dict result from Fal Veo2 API handle, but got: {type(final_response)}"
                )
                raise Exception(
                    "Video generation failed", "Unexpected result type from API handle"
                )

            video_info = final_response.get(
                "video"
            )  # Access the key from the result dict
            if video_info and isinstance(video_info, dict) and "url" in video_info:
                video_url = video_info["url"]
                logger.info(f"Found video URL in result['video']['url']: {video_url}")

                # Download the video content
                logger.info(f"Attempting to download video from: {video_url}")
                # Increased timeout for potentially large video files
                video_response = requests.get(video_url, timeout=180)
                logger.info(f"Video download status code: {video_response.status_code}")
                video_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                video_content = video_response.content
                logger.info(
                    f"Video content type: {type(video_content)}, length: {len(video_content)}"
                )
                return video_content
            else:
                # Log the structure if it's unexpected
                logger.error(
                    f"Unexpected response structure or missing video URL from Fal Veo2 API: {final_response}"
                )
                raise Exception(
                    "Video generation failed",
                    "Unexpected API response structure or missing video URL",
                )

        except Exception as e:
            # Log the full exception trace for better debugging
            logger.exception(
                f"Fal AI Veo2 API call, result fetching, or download failed: {e}"
            )
            # Re-raise a user-friendly exception, check if FalClientError provides details
            error_msg = f"Fal API interaction error: {e}"
            # Attempt to extract more specific error message from FalClientError if possible
            # This depends on the structure of the exception raised by fal_client
            # if hasattr(e, 'response') and hasattr(e.response, 'text'):
            #     try:
            #         error_detail = e.response.json().get('detail')
            #         if error_detail:
            #             error_msg = f"Fal API Error: {error_detail}"
            #     except: # Fallback if response is not JSON or detail key missing
            #         pass
            raise Exception("Video generation failed", error_msg)

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the Veo2 service. Handles parameter processing and API call.
        """
        logger.info(f"Veo2 Raw parameters received in remote: {parameters}")

        # --- Pre-process parameters before validation ---
        processed_params = parameters.copy()  # Work on a copy

        # Extract file_url string if it's passed as a list/dict
        file_url_input = processed_params.get("file_url")
        if (
            isinstance(file_url_input, list)
            and len(file_url_input) > 0
            and isinstance(file_url_input[0], dict)
        ):
            processed_params["file_url"] = file_url_input[0].get("uri")
            logger.debug(
                f"Extracted file_url from list: {processed_params['file_url']}"
            )
        elif isinstance(file_url_input, dict):  # Handle if just a dict is passed
            processed_params["file_url"] = file_url_input.get("uri")
            logger.debug(
                f"Extracted file_url from dict: {processed_params['file_url']}"
            )

        # Ensure it's None if not a valid string URL after processing OR if it wasn't provided
        if not isinstance(
            processed_params.get("file_url"), str
        ) or not processed_params.get("file_url"):
            processed_params["file_url"] = None  # Set explicitly to None
            logger.debug(
                "Processed file_url set to None as input was missing or not a valid string."
            )

        # --- FIX: Format duration to the required "Xs" string format ---
        duration_input = processed_params.get("duration")
        if isinstance(duration_input, (int, float)):  # Handle numeric input (5, 5.0)
            processed_params["duration"] = (
                f"{int(duration_input)}s"  # Convert to int and format as string "Xs"
            )
            logger.debug(
                f"Formatted numeric duration to string: {processed_params['duration']}"
            )
        elif (
            isinstance(duration_input, str) and duration_input.isdigit()
        ):  # Handle string digit input ("5")
            processed_params["duration"] = f"{duration_input}s"  # Format as string "Xs"
            logger.debug(
                f"Formatted string-digit duration to string: {processed_params['duration']}"
            )
        elif isinstance(duration_input, str) and duration_input.endswith(
            "s"
        ):  # Already correct format ("5s")
            logger.debug(f"Duration already correctly formatted: {duration_input}")
        else:
            # Let Pydantic handle validation for other invalid types or missing values
            logger.warning(
                f"Unexpected duration type/format: {duration_input}. Passing to validation."
            )
            # Keep original value for Pydantic validation to catch it if it's not one of the Literals

        # Validate the *processed* parameters
        try:
            # Pydantic now validates against the "Xs" string format
            params: Veo2Parameters = Veo2Parameters(**processed_params)
            logger.info(f"Veo2 Parsed parameters: {params}")
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for Veo2 parameters after processing: {e}"
            )
            # Re-raise to be caught by the calling task's error handler
            raise e  # Raise the validation error

        # Make the API request using the validated Pydantic object
        video_bytes = self.make_api_request(params)  # Now passes the validated object

        # --- Prepare response ---
        logger.info(
            f"Type received from make_api_request in remote: {type(video_bytes)}"
        )

        result_list = [video_bytes] if isinstance(video_bytes, bytes) else []
        if not result_list and isinstance(
            video_bytes, list
        ):  # Handle if make_api_request returned a list already
            result_list = video_bytes

        prepared_dict = prepare_response(
            result=result_list,
            Has_NSFW_content=[False]
            * len(
                result_list
            ),  # Assume false for now, Fal might not report this for video
            time_data=0,  # Placeholder, decorator handles this
            runtime=0,  # Placeholder, decorator handles this
            extension=OUTPUT_VIDEO_EXTENSION,
        )

        return prepared_dict


class Kling2Master(IService):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Kling2Master Service Initialized")

    def make_api_request(self, parameters: Kling2MasterParameters) -> bytes:
        try:
            duration_float = float(parameters.duration)
        except ValueError:
            # If duration isn't strictly '5' or '10', the API will likely fail anyway.
            # Let the API call handle the error for invalid duration strings.
            logger.error(
                f"Invalid duration format received: {parameters.duration}. Expected '5' or '10'."
            )
            raise ValueError(f"Invalid duration format: {parameters.duration}")

        input_args = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "duration": parameters.duration,  # Pass the validated string '5' or '10'
            "aspect_ratio": parameters.aspect_ratio,
            "cfg_scale": parameters.cfg_scale,
        }
        if parameters.file_url:
            input_args["image_url"] = parameters.file_url
            model_id = "fal-ai/kling-video/v2/master/image-to-video"
        else:
            model_id = "fal-ai/kling-video/v2/master/text-to-video"

        logger.debug(f"Calling Fal API [{model_id}]")
        try:
            request_handle = fal_client.submit(model_id, arguments=input_args)
            final_response = request_handle.get()
            logger.debug("Kling2Master API final result fetched.")

            if not isinstance(final_response, dict):
                logger.error(
                    f"Expected dict result from Kling2Master API handle, but got: {type(final_response)}"
                )
                raise Exception(
                    "Video generation failed", "Unexpected result type from API handle"
                )

            video_info = final_response.get("video")
            video_url = None

            if video_info and isinstance(video_info, dict) and "url" in video_info:
                video_url = video_info["url"]
            else:
                for key, value in final_response.items():
                    if (
                        isinstance(value, dict)
                        and "url" in value
                        and isinstance(value["url"], str)
                    ):
                        video_url = value["url"]
                        logger.info(f"Found potential video URL in key '{key}'")
                        break
                if not video_url:
                    logger.error(
                        f"Could not find any video URL in API response. Keys: {final_response.keys()}"
                    )
                    raise KeyError("Video URL not found in API response")

            video_response = requests.get(video_url, timeout=180)
            video_response.raise_for_status()
            return video_response.content

        except Exception as e:
            logger.exception(f"Fal AI Kling2Master API interaction failed: {e}")
            error_msg = f"Fal API interaction error: {e}"
            if isinstance(e, fal_client.client.FalClientError):
                try:
                    details = e.args[0]
                    if (
                        isinstance(details, list)
                        and len(details) > 0
                        and isinstance(details[0], dict)
                    ):
                        error_msg = (
                            f"Fal API Error: {details[0].get('msg', 'Unknown error')}"
                        )
                except Exception:
                    pass
            raise Exception("Video generation failed", error_msg)

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        logger.info(
            f"Kling2Master Raw parameters received: {parameters.get('prompt', 'No prompt found')[:50]}..."
        )  # Log start with partial prompt
        processed_params = parameters.copy()

        file_url_input = processed_params.get("file_url")
        if (
            isinstance(file_url_input, list)
            and len(file_url_input) > 0
            and isinstance(file_url_input[0], dict)
        ):
            processed_params["file_url"] = file_url_input[0].get("uri")
        elif isinstance(file_url_input, dict):
            processed_params["file_url"] = file_url_input.get("uri")
        if not isinstance(
            processed_params.get("file_url"), str
        ) or not processed_params.get("file_url"):
            processed_params["file_url"] = None

        duration_input = processed_params.get("duration")
        if isinstance(duration_input, (int, float)):
            if int(duration_input) in [5, 10]:
                processed_params["duration"] = str(int(duration_input))
            else:
                # Let Pydantic validation handle the error
                pass
        elif isinstance(duration_input, str) and duration_input.isdigit():
            if duration_input not in ["5", "10"]:
                # Let Pydantic validation handle the error
                pass
        # If it's not a number or a valid string digit, let Pydantic handle it

        try:
            # Ensure duration is validated correctly ('5' or '10' as string)
            params: Kling2MasterParameters = Kling2MasterParameters(**processed_params)
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for Kling2Master parameters: {e}")
            raise e  # Re-raise the validation error

        video_bytes = self.make_api_request(params)

        result_list = [video_bytes] if isinstance(video_bytes, bytes) else []
        if not result_list and isinstance(video_bytes, list):
            result_list = video_bytes

        return prepare_response(
            result=result_list,
            Has_NSFW_content=[False] * len(result_list),
            time_data=0,
            runtime=0,
            extension=OUTPUT_VIDEO_EXTENSION,
        )
