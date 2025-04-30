import concurrent.futures

import google.auth
import google.auth.transport.requests
from google import genai
from openai import OpenAI
from loguru import logger
from openai.types.images_response import ImagesResponse
from src.data_models.ModalAppSchemas import (
    DalleParameters,
    NSFWSchema,
    GPTImageParameters,
)
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import (
    COPYRIGHT_DETECTION_FUNCTION_CALLING_SCHEMA,
    DALLE_SUPPORTED_HW,
    IMAGE_GENERATION_ERROR,
    NSFW_CONTENT_DETECT_ERROR_MSG,
    OUTPUT_IMAGE_EXTENSION,
    google_credentials_info,
    GPT_IMAGE_SUPPORTED_HW,
)
from src.utils.Globals import (
    get_image_from_url,
    make_request,
    prepare_response,
    timing_decorator,
)


class DalleText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the DALL-E 3 image generation service.

        Sets up the service by inheriting API credentials from the parent IService
        class and initializes the OpenAI client with the authentication token.
        """
        super().__init__()
        self.client = OpenAI(api_key=self.openai_api_key)

    def make_dalle_api_request(
        self, prompt: str, Height_width: str, quality: str
    ) -> str:
        """
        Sends a request to the OpenAI DALL-E 3 API for image generation.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the DALL-E API endpoint, and handles the response
        processing to extract the resulting image.

        Args:
            prompt (str): The text prompt describing the desired image
            Height_width (str): Dimensions of the output image (e.g., "1024x1024")
            quality (str): Image quality setting ("standard" or "hd")

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the prompt violates content policy or other API errors occur
        """
        try:
            response: ImagesResponse = self.client.images.generate(
                model="dall-e-3", prompt=prompt, size=Height_width, quality=quality, n=1
            )
        except Exception as error:
            if "content_policy_violation" in str(error):
                raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)
            else:
                raise Exception(str(error))

        response = make_request(response.data[0].url, "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method validates the input parameters, ensures the requested dimensions
        are supported by DALL-E, and creates multiple parallel generation tasks
        based on the batch size. The @timing_decorator tracks and adds execution
        time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format

        Raises:
            Exception: If the requested dimensions are not supported by DALL-E
        """
        parameters: DalleParameters = DalleParameters(**parameters)

        Height_width = f"{parameters.width}x{parameters.height}"

        if Height_width not in DALLE_SUPPORTED_HW:
            raise Exception(
                f"Height and width should be {str(DALLE_SUPPORTED_HW)} for Dalle"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(
                    self.make_dalle_api_request,
                    parameters.prompt,
                    Height_width,
                    parameters.quality,
                )
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)


class GPTText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the GPT image generation service.

        Sets up the service by inheriting API credentials from the parent IService
        class and initializes the OpenAI client with the authentication token.
        """
        super().__init__()
        self.client = OpenAI(api_key=self.openai_api_key)

    def make_gpt_api_request(self, prompt: str, Height_width: str, quality: str) -> str:
        """
        Sends a request to the OpenAI GPT Image API for image generation.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the DALL-E API endpoint, and handles the response
        processing to extract the resulting image.

        Args:
            prompt (str): The text prompt describing the desired image
            Height_width (str): Dimensions of the output image (e.g., "1024x1024")
            quality (str): Image quality setting ("standard" or "hd")

        Returns:
            str: The generated image data as bytes

        Raises:
            Exception: If the prompt violates content policy or other API errors occur
        """
        try:
            response: ImagesResponse = self.client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=Height_width,
                quality=quality,
                n=1,
            )
        except Exception as error:
            if "content_policy_violation" in str(error):
                raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)
            else:
                raise Exception(str(error))

        if response.data and len(response.data) > 0 and response.data[0].b64_json:
            base64_image = response.data[0].b64_json
            # Optionally add prefix if needed by prepare_response later
            # if not base64_image.startswith('data:image/png;base64,'):
            #     base64_image = f'data:image/png;base64,{base64_image}'
            return base64_image
        response = make_request(response.data[0].url, "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method validates the input parameters, ensures the requested dimensions
        are supported by DALL-E, and creates multiple parallel generation tasks
        based on the batch size. The @timing_decorator tracks and adds execution
        time to the response.

        Args:
            parameters (dict): Request parameters for image generation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format

        Raises:
            Exception: If the requested dimensions are not supported by DALL-E
        """

        try:
            parameters: GPTImageParameters = GPTImageParameters(**parameters)

            Height_width = f"{parameters.width}x{parameters.height}"

            if Height_width not in GPT_IMAGE_SUPPORTED_HW:
                raise Exception(
                    f"Height and width should be {str(GPT_IMAGE_SUPPORTED_HW)} for Dalle"
                )
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        self.make_gpt_api_request,
                        parameters.prompt,
                        Height_width,
                        parameters.quality,
                    )
                    for _ in range(parameters.batch)
                ]
                results = [future.result() for future in futures]

            Has_NSFW_Content = [False] * parameters.batch
            return prepare_response(
                results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION
            )

        except Exception as e:
            try:
                logger.error(
                    f"Caught exception in GPT Image processing: Type={type(e).__name__}, Args={e.args!r}",
                    exc_info=True,
                )
            except Exception as log_err:
                logger.error(
                    f"Error during exception logging: {log_err}"
                )  # Fallback log

            error_message = str(e)
            is_safety_block = "safety system" in error_message and (
                "moderation_blocked" in error_message
                or "content that is not allowed" in error_message
            )

            if is_safety_block:
                raise Exception(NSFW_CONTENT_DETECT_ERROR_MSG) from e
            else:
                raise Exception(IMAGE_GENERATION_ERROR) from e


class GeminiAIImageCheck(IService):
    def __init__(self) -> None:
        """
        Initializes the Gemini AI image checking service.

        Sets up the service by inheriting API credentials from the parent IService
        class, configures the function calling schema for NSFW detection, and
        initializes the Gemini AI client with the appropriate authentication.
        """
        super().__init__()
        self.function_calling_schema = COPYRIGHT_DETECTION_FUNCTION_CALLING_SCHEMA
        credentials, project_id = google.auth.load_credentials_from_dict(
            google_credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        self.client = genai.Client(
            vertexai=True,
            project="marine-potion-404413",
            credentials=credentials,
            location="us-central1",
        )

    def remote(self, image_url: str) -> bool:
        """
        Analyzes an image for NSFW content using Gemini AI.

        This function sends the image to Gemini's content analysis API with specific
        instructions to detect various categories of NSFW content. The model returns
        a structured response indicating whether inappropriate content was detected.

        Args:
            image_url (str): URL of the image to analyze

        Returns:
            bool: True if NSFW content is detected, False otherwise
        """
        response = self.client.models.generate_content(
            contents=[
                """Analyze if an image contains any NSFW content.

            Return true if the image contains any kind of NSFW content, including but not limited to:

            Visible private parts,
            Pornographic material,
            Explicit sexual activities,
            Nudity,
            Suggestive or provocative imagery,
            Activities like kissing or intimate physical contact with sexual undertones.

            Return false for any other category apart from NSFW content.
            
            You must return false for any content that does not fall into the NSFW category.""",
                get_image_from_url(image_url),
            ],
            # model="gemini-2.0-flash",
            model="gemini-2.0-flash-lite-001",
            config={
                "response_mime_type": "application/json",
                "response_schema": NSFWSchema,
            },
        )
        return response.parsed.NSFW_content
