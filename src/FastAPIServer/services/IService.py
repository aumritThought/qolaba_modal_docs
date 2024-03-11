from abc import ABC, abstractmethod
import os
from src.utils.Constants import (
    STABILITY_API, 
    DID_TALK_API,  
    SDXL_ENGINE_ID, 
    CLIPDROP_UNCROP_URL,
    CLIPDROP_CLEANUP_URL,
    CLIPDROP_REMOVE_TEXT_URL, 
    CLIPDROP_REPLACE_BACKGROUND_URL)

class IService(ABC):
    def __init__(self) -> None:
        #SDXL definations
        self.stability_api = STABILITY_API
        self.stability_engine_id = SDXL_ENGINE_ID
        self.stability_api_key = os.environ["SDXL_API_KEY"]

        #elevenlabs definations
        self.elevenlabs_api_key = os.environ["ELEVENLABS_API_KEY"] 

        #openai definations
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

        #did definations
        self.did_api_key = os.environ["DID_KEY"]
        self.did_url = DID_TALK_API

        #clipdrop definations
        self.clipdrop_api_key = os.environ["CLIPDROP_APIKEY"]
        self.clipdrop_cleanup_url = CLIPDROP_CLEANUP_URL
        self.clipdrop_remove_text_url = CLIPDROP_REMOVE_TEXT_URL
        self.clipdrop_replace_background_url = CLIPDROP_REPLACE_BACKGROUND_URL
        self.clipdrop_uncrop_url = CLIPDROP_UNCROP_URL

    @abstractmethod
    def remote(self, data):
        pass
