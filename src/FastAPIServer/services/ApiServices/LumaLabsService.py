from src.data_models.ModalAppSchemas import LumaLabsVideoParameters
from src.utils.Globals import timing_decorator, prepare_response, make_request
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_VIDEO_EXTENSION
import concurrent.futures
from lumaai import LumaAI


class LumaVideo(IService):
    def __init__(self) -> None:
        """
        Initializes the Luma video generation service.

        Sets up the service by inheriting API credentials from the parent IService
        class and initializes the LumaAI client with the authentication token.
        """
        super().__init__()
        self.client = LumaAI(auth_token=self.lumalabs_api_key)

    def generate_image(self, parameters: LumaLabsVideoParameters) -> str:
        """
        Generates video content using Luma's AI system.

        This function supports multiple generation modes:
        - Text-only generation: Using just a prompt to create a video
        - Single-image guided: Using one reference image (at start or end)
        - Two-image guided: Using two reference images as start and end frames

        The function polls the Luma API until the generation is complete or
        fails, handling the asynchronous nature of video generation.

        Args:
            parameters: Configuration parameters including prompt, aspect ratio,
                loop settings, and optional reference images

        Returns:
            str: The generated video data as bytes

        Raises:
            RuntimeError: If the video generation process fails
        """
        if parameters.file_url == None:
            parameters.file_url = []
        if len(parameters.file_url) == 0:
            generation = self.client.generations.create(
                prompt=parameters.prompt,
                aspect_ratio=parameters.aspect_ratio,
                loop=parameters.loop,
            )
            completed = False
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    raise RuntimeError(
                        f"Generation failed: {generation.failure_reason}"
                    )
        else:
            if len(parameters.file_url) == 1:
                if parameters.file_url[0].position == "first":
                    frame = "frame0"
                else:
                    frame = "frame1"
                keyframes = {
                    frame: {"type": "image", "url": parameters.file_url[0].uri}
                }
            else:
                if parameters.file_url[0].position == "first":
                    first_image = parameters.file_url[0].uri
                    second_image = parameters.file_url[1].uri
                else:
                    first_image = parameters.file_url[1].uri
                    second_image = parameters.file_url[0].uri
                keyframes = {
                    "frame0": {"type": "image", "url": first_image},
                    "frame1": {"type": "image", "url": second_image},
                }
            generation = self.client.generations.create(
                prompt=parameters.prompt,
                aspect_ratio=parameters.aspect_ratio,
                loop=parameters.loop,
                keyframes=keyframes,
            )
            completed = False
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    raise RuntimeError(
                        f"Generation failed: {generation.failure_reason}"
                    )

        response = make_request(generation.assets.video, "GET")

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
        """
        parameters: LumaLabsVideoParameters = LumaLabsVideoParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)
