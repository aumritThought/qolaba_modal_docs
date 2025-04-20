import concurrent.futures

import google.auth
import google.auth.transport.requests
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import (
    IMAGEGEN_ASPECT_RATIOS,
    IMAGEGEN_ERROR,
    IMAGEGEN_ERROR_MSG,
    OUTPUT_IMAGE_EXTENSION,
    google_credentials_info,
)
from src.utils.Globals import (
    convert_to_aspect_ratio,
    prepare_response,
    timing_decorator,
)


class ImageGenText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the Gemini service with appropriate API endpoints.

        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        credentials, project_id = google.auth.load_credentials_from_dict(
            google_credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        vertexai.init(project="marine-potion-404413", credentials=credentials)
        self.generation_model = ImageGenerationModel.from_pretrained(
            "imagen-3.0-generate-001"
        )

    def make_api_request(self, parameters: IdeoGramText2ImageParameters) -> str:
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
        image = self.generation_model.generate_images(
            prompt=parameters.prompt,
            number_of_images=1,
            aspect_ratio=parameters.aspect_ratio,
            safety_filter_level="block_some",
        )
        if len(image.images) == 0:
            raise Exception(IMAGEGEN_ERROR, IMAGEGEN_ERROR_MSG)
        return image.images[0]._image_bytes

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
        parameters: IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(
            **parameters
        )

        parameters.aspect_ratio = convert_to_aspect_ratio(
            parameters.width, parameters.height
        )
        if parameters.aspect_ratio not in IMAGEGEN_ASPECT_RATIOS:
            raise Exception("Invalid Height and width dimension")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
