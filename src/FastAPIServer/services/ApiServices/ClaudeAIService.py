from src.data_models.ModalAppSchemas import PromptParrotParameters
from src.utils.Globals import timing_decorator, prepare_response
from src.utils.Constants import (
    BASE_PROMPT_FOR_GENERATION,
    google_credentials_info,
    BASE_PROMPT_FOR_VIDEO_GENERATION,
)
from src.FastAPIServer.services.IService import IService
import concurrent.futures
from anthropic import AnthropicVertex
import google.auth
import google.auth.transport.requests


class PromptParrot(IService):
    def __init__(self) -> None:
        """
        Initializes the PromptParrot service with Claude AI client.

        Sets up the AnthropicVertex client with appropriate authentication credentials
        from Google Cloud. This establishes the connection to Claude AI that will be
        used for prompt enhancement.
        """
        super().__init__()
        self.client = AnthropicVertex(
            region="us-east5", project_id="marine-potion-404413"
        )
        self.credentials, project_id = google.auth.load_credentials_from_dict(
            google_credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        self.credentials.refresh(request)
        self.client.credentials = self.credentials

    def generate_prompt(self, query: str) -> str:
        """
        Enhances a user prompt using Claude AI.

        This function takes a simple user query and sends it to Claude AI to create
        a more detailed, creative prompt that can produce better image generation results.
        It uses the claude-3-haiku model with a moderate temperature to balance creativity
        and coherence.

        Args:
            query (str): The user's input prompt to enhance, combined with base prompt

        Returns:
            str: The enhanced, detailed prompt for image generation
        """
        response = self.client.messages.create(
            model="claude-3-haiku@20240307",
            messages=[{"role": "user", "content": query}],
            temperature=0.5,
            stream=False,
            max_tokens=1000,
        )

        return response.content[0].text

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Main entry point for prompt enhancement service.

        This method handles the request for prompt enhancement, supporting batch processing
        to generate multiple variations. It combines the user's input with a base prompt
        template to provide proper context to the AI, then processes the requests in
        parallel using a thread pool for efficiency.

        Args:
            parameters (dict): Request parameters including the original prompt and batch size

        Returns:
            dict: Standardized response containing the enhanced prompts
        """
        parameters: PromptParrotParameters = PromptParrotParameters(**parameters)
        parameters.prompt = f"{BASE_PROMPT_FOR_GENERATION} \n{parameters.prompt}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_prompt, parameters.prompt)
                futures.append(future)

            results = [future.result() for future in futures]

        return prepare_response(results, [False] * parameters.batch, 0, 0)


class VideoPromptParrot(IService):
    def __init__(self) -> None:
        """
        Initializes the VideoPromptParrot service with Claude AI client.

        Sets up the AnthropicVertex client with appropriate authentication credentials
        from Google Cloud. This establishes the connection to Claude AI that will be
        used for video prompt enhancement.
        """
        super().__init__()
        self.client = AnthropicVertex(
            region="us-east5", project_id="marine-potion-404413"
        )
        self.credentials, project_id = google.auth.load_credentials_from_dict(
            google_credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        self.credentials.refresh(request)
        self.client.credentials = self.credentials

    def generate_prompt(self, query: str) -> str:
        """
        Enhances a user prompt for video generation using Claude AI.

        This function takes a simple user query and sends it to Claude AI to create
        a more detailed, creative prompt specifically designed for video generation.
        The enhanced prompt will include elements like scene flow, camera movements,
        and temporal details that are important for video generation.

        Args:
            query (str): The user's input prompt to enhance, combined with base prompt

        Returns:
            str: The enhanced, detailed prompt for video generation
        """
        response = self.client.messages.create(
            model="claude-3-haiku@20240307",
            messages=[{"role": "user", "content": query}],
            temperature=0.5,
            stream=False,
            max_tokens=1000,
        )

        return response.content[0].text

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Main entry point for video prompt enhancement service.

        This method handles the request for video prompt enhancement, supporting batch processing
        to generate multiple variations. It combines the user's input with a video-specific base
        prompt template to provide proper context to the AI, then processes the requests in
        parallel using a thread pool for efficiency.

        Args:
            parameters (dict): Request parameters including the original prompt and batch size

        Returns:
            dict: Standardized response containing the enhanced video prompts
        """
        parameters: PromptParrotParameters = PromptParrotParameters(**parameters)
        parameters.prompt = f"{BASE_PROMPT_FOR_VIDEO_GENERATION} \n{parameters.prompt}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_prompt, parameters.prompt)
                futures.append(future)

            results = [future.result() for future in futures]

        return prepare_response(results, [False] * parameters.batch, 0, 0)
