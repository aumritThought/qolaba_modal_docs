from abc import ABC, abstractmethod
import os
from src.utils.Constants import (
    STABILITY_API, 
    SDXL_INPAINT_URL,
    DID_TALK_API,  
    SDXL_ENGINE_ID, 
    SDX3_URL)


class IService(ABC):
    def __init__(self) -> None:
        #SDXL definations
        self.stability_api = STABILITY_API
        self.stability_engine_id = SDXL_ENGINE_ID
        self.stability_api_key = os.environ["SDXL_API_KEY"]
        self.stability_inpaint_url = SDXL_INPAINT_URL
        self.sdxl3_url = SDX3_URL

        #elevenlabs definations
        self.elevenlabs_api_key = os.environ["ELEVENLABS_API_KEY"] 

        #openai definations
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

        #claude definations
        self.claude_api_key = os.environ["CLAUDE_KEY"]

        #did definations
        self.did_api_key = os.environ["DID_KEY"]
        self.did_url = DID_TALK_API


    @abstractmethod
    def remote(self, data):
        pass
