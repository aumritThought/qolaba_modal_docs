import concurrent.futures
import openai
import google.auth
import google.auth.transport.requests
from google import genai
from openai import OpenAI
from loguru import logger
import requests
import io
import os
from urllib.parse import urlparse
from typing import List, Tuple, BinaryIO
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
import tempfile # <--- Add this

from PIL import Image  # Already imported in other files

# Add this helper function before the GPTText2Image class
def prepare_image_for_openai(img_bytes, ext):
    """
    Prepares an image for OpenAI API with explicit MIME type handling.
    
    Args:
        img_bytes (bytes): Raw image bytes
        ext (str): File extension with dot (.png, .jpg, etc)
        
    Returns:
        bytes-like object ready for OpenAI API
    """
    # Set up MIME type mapping
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp'
    }
    
    # Get MIME type or default to image/png
    mime_type = mime_map.get(ext.lower(), 'image/png')
    
    # Write to a temporary file with the correct extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_file_path = temp_file.name
    
    try:
        temp_file.write(img_bytes)
        temp_file.close()
        
        # Convert the temp file into an OpenAI-compatible format with correct MIME type
        with open(temp_file_path, 'rb') as f:
            return f, mime_type, temp_file_path  # Return the file object, MIME type, and path for cleanup
    except Exception as e:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise e

def get_openai_image_param_from_urls(image_url_list: List[str]) -> Tuple[str, BinaryIO]:
    """
    Downloads image from first URL, prepares it as (filename, io.BytesIO object).

    Intended to be called directly as the value for the 'image' parameter in
    OpenAI API calls like images.edit.

    Args:
        image_url_list: List containing the image URL string as the first element.

    Returns:
        Tuple[str, io.BytesIO]: (filename_with_extension, open_bytesio_object)

    Raises:
        ValueError: If input is invalid or extension cannot be determined.
        requests.exceptions.RequestException: If download fails.
        IOError: If BytesIO creation fails.

    REMINDER: The caller MUST close the returned io.BytesIO object afterwards.
    """
    if not image_url_list or not isinstance(image_url_list[0], str) or not image_url_list[0]:
        logger.error("Invalid input: Requires a non-empty list with a valid URL string.")
        raise ValueError("Requires a non-empty list with a valid URL string as the first element.")

    url = image_url_list[0]
    logger.debug(f"Processing URL for OpenAI param: {url}")
    try:
        # Basic filename/extension extraction
        path = urlparse(url).path
        filename = os.path.basename(path) if path else "image.png" # Default filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.png', '.jpeg', '.jpg', '.webp']:
             logger.warning(f"Could not determine valid extension from '{filename}', defaulting to .png")
             filename = f"{os.path.splitext(filename)[0] or 'image'}.png" # Default extension
        elif not os.path.splitext(filename)[0]: # Handle case like "/.png"
             filename = f"image{ext}"
        logger.debug(f"Using filename '{filename}' for API.")


        # Download bytes
        logger.info(f"Downloading image from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if not response.content:
             logger.error(f"Downloaded empty content from {url}")
             raise ValueError(f"Downloaded empty content from {url}")
        logger.info(f"Downloaded {len(response.content)} bytes from {url}")


        # Return the required tuple
        bytes_io_obj = io.BytesIO(response.content)
        logger.debug(f"Returning tuple: ({filename}, {type(bytes_io_obj)})")
        return (filename, bytes_io_obj)

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}", exc_info=True)
        raise # Re-raise original error
    except Exception as e:
        logger.error(f"Failed to prepare image parameter from {url}: {e}", exc_info=True)
        raise # Re-raise original e


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
    
    

    def make_gpt_api_request(self, prompt: str, Height_width: str, quality: str, images: list = None) -> str:
        """
        Sends a request to OpenAI API for image generation or editing.
        
        Args:
            prompt: The text prompt.
            Height_width: Target dimensions (e.g., "1024x1024").
            quality: Quality setting.
            images: Optional list containing the URL of the image to edit. Only the first is used.
            
        Returns:
            str or bytes: Resulting image data (base64 string or bytes).
        """
        if images and isinstance(images, list) and len(images) > 0 and isinstance(images[0], str) and images[0]:
            # Image edit mode
            url = images[0]
            logger.info(f"Performing image edit using URL: {url}")
            
            # Download the image
            logger.info(f"Downloading image from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            img_bytes = response.content
            logger.info(f"Downloaded {len(img_bytes)} bytes from {url}")
            
            # Get the filename and extension
            path = urlparse(url).path
            filename = os.path.basename(path) if path else "image.png"
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.png', '.jpeg', '.jpg', '.webp']:
                ext = ".png"
            
            # Process the image with PIL to PNG format
            img = Image.open(io.BytesIO(img_bytes))
            output_buffer = io.BytesIO()
            img.save(output_buffer, format="PNG")
            output_buffer.seek(0)
            
            # Write to temporary file and use for API call
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(output_buffer.getvalue())
                tmp.close()
            
            # Make API call with the temporary file
            logger.info(f"Calling OpenAI images.edit API with model gpt-image-1...")
            with open(tmp_path, "rb") as image_file:
                response = self.client.images.edit(
                    model="gpt-image-1",
                    image=image_file,
                    prompt=prompt,
                    n=1
                )
            
            # Clean up temporary file
            os.unlink(tmp_path)
            logger.info("OpenAI images.edit call successful")
            
        else:
            # Generate image from text
            generate_model = "gpt-image-1"
            logger.info(f"Performing text-to-image generation using {generate_model} model")
            response = self.client.images.generate(
                model=generate_model,
                prompt=prompt,
                size=Height_width,
                quality=quality,
                n=1,
            )
            logger.info("OpenAI images.generate call successful")

        # Process the response
        output_data = response.data[0]
        
        # Return the appropriate data format
        if output_data.b64_json:
            logger.debug("Received b64_json response from OpenAI")
            return output_data.b64_json
        else:
            logger.debug(f"Received URL response from OpenAI: {output_data.url}")
            logger.info(f"Fetching final image from result URL: {output_data.url}")
            final_response = requests.get(output_data.url, timeout=60)
            final_response.raise_for_status()
            logger.info(f"Fetched {len(final_response.content)} bytes from result URL")
            return final_response.content
        
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
                        parameters.images
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
