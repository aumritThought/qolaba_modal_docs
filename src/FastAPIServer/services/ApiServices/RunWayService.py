from src.data_models.ModalAppSchemas import RunwayImage2VideoParameters
from src.utils.Globals import (
    timing_decorator,
    prepare_response,
    make_request,
    get_image_from_url,
)
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import (
    OUTPUT_VIDEO_EXTENSION,
    RUNWAY_ERROR,
    RUNWAY_ERROR_MSG,
    RUNWAY_POSITION_ERROR_MSG,
)
import concurrent.futures
from runwayml import RunwayML
import io
import base64


class RunwayImage2Video(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = RunwayML(api_key=self.runway_api_key)

    def download_and_convert_to_base64(self, url: str) -> str:
        """
        Downloads an image and converts it to base64 format for Runway's API.

        This function retrieves an image from a URL, processes it to ensure
        compatibility with Runway's requirements, and converts it to a base64
        encoded string with the proper data URI format.

        Args:
            url (str): URL of the image to download and convert

        Returns:
            str: Base64 encoded image data in data URI format
        """
        image = get_image_from_url(url)

        # width, height = image.size
        # aspect_ratio = width / height

        # if aspect_ratio > 2:
        #     new_width = height * 2
        #     left = (width - new_width) // 2
        #     image = image.crop((left, 0, left + new_width, height))

        # elif aspect_ratio < 0.5:
        #     new_height = width * 2
        #     top = (height - new_height) // 2
        #     image = image.crop((0, top, width, top + new_height))

        base64_image = io.BytesIO()

        image.save(base64_image, "JPEG")

        img_str = base64.b64encode(base64_image.getvalue())

        base64_string = img_str.decode("utf-8")

        return f"data:image/jpg;base64,{base64_string}"

    def generate_image(self, parameters: RunwayImage2VideoParameters) -> str:
        """
        Generates a video from reference images and text prompts.

        This function processes the input parameters, converts reference images
        to the required format, and sends a request to RunwayML's image-to-video
        API. It monitors the generation process until completion, handling the
        asynchronous nature of video generation.

        Args:
            parameters: Configuration parameters including prompt, aspect ratio,
                duration, and reference images

        Returns:
            str: The generated video data as bytes

        Raises:
            Exception: If no images are provided, if image positions conflict,
                or if the generation process fails
        """
        if len(parameters.file_url) > 0:
            parameters.file_url[0].uri = self.download_and_convert_to_base64(
                parameters.file_url[0].uri
            )

            if len(parameters.file_url) > 1:
                parameters.file_url[1].uri = self.download_and_convert_to_base64(
                    parameters.file_url[1].uri
                )
                if parameters.file_url[0].position == parameters.file_url[1].position:
                    raise Exception(RUNWAY_ERROR, RUNWAY_POSITION_ERROR_MSG)
        else:
            raise Exception(RUNWAY_ERROR, RUNWAY_ERROR_MSG)

        # print(parameters.file_url)
        task = self.client.image_to_video.create(
            model="gen3a_turbo",
            prompt_image=parameters.model_dump()["file_url"],
            prompt_text=parameters.prompt,
            duration=parameters.duration,
            ratio=parameters.aspect_ratio,
        )
        task = self.client.tasks.retrieve(id=task.id)
        while task.status != "SUCCEEDED":
            task = self.client.tasks.retrieve(id=task.id)
            if task.status in ["FAILED", "CANCELLED"]:
                print(task.status, task.model_dump())
                raise Exception("Video generation failed")

        response = make_request(task.output[0], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method processes the input parameters, creates multiple parallel
        generation tasks based on the batch size, and aggregates the results.
        The @timing_decorator tracks and adds execution time to the response.

        Args:
            parameters (dict): Request parameters for video generation

        Returns:
            dict: Standardized response containing generated videos, NSFW flags,
                timing information, and file format
        """
        parameters: RunwayImage2VideoParameters = RunwayImage2VideoParameters(
            **parameters
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)
