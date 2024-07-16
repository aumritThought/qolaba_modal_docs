from src.data_models.ModalAppSchemas import PromptParrotParameters
from src.utils.Globals import timing_decorator, prepare_response
from src.utils.Constants import BASE_PROMPT_FOR_GENERATION, google_credentials_info
from src.FastAPIServer.services.IService import IService
import concurrent.futures 
from anthropic import AnthropicVertex
import google.auth
import google.auth.transport.requests

class PromptParrot(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = AnthropicVertex(region="us-east5", project_id="marine-potion-404413")
        self.credentials, project_id = google.auth.load_credentials_from_dict(google_credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        request = google.auth.transport.requests.Request()
        self.credentials.refresh(request)
        self.client.credentials = self.credentials

    def generate_prompt(self, query : str) -> str:
        response = self.client.messages.create(
                model="claude-3-haiku@20240307",
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=0.5,
                stream=False,
                max_tokens = 1000
                )

        return response.content[0].text

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : PromptParrotParameters = PromptParrotParameters(**parameters)
        parameters.prompt = f"{BASE_PROMPT_FOR_GENERATION} \n{parameters.prompt}"

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_prompt, parameters.prompt)
                futures.append(future)
            
            results = [future.result() for future in futures]

        return prepare_response(results, [False]*parameters.batch, 0, 0)

