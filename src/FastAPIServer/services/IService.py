from abc import ABC, abstractmethod
import os
from src.utils.Constants import (
    STABILITY_API, 
    SDXL_INPAINT_URL,
    DID_TALK_API,  
    SDXL_ENGINE_ID,
    MUSIC_GEN_API, 
    IDEOGRAM_GENERATE_URL,
    IDEOGRAM_EDIT_URL,
    IDEOGRAM_REMIX_URL,
    LEONARDO_IMAGE_GEN_URL,
    LEONARDO_IMAGE_STATUS_URL,
    AILAB_HAIRSTYLE_URL,
    AILAB_STATUS_URL,
    SDX3_URL)


class IService(ABC):
    def __init__(self) -> None:
        """
        Initializes API endpoints and authentication credentials from environment variables.
        
        Sets up connections to various external services by configuring their endpoints
        and loading corresponding API keys from environment variables. This centralized
        approach ensures that all services have access to the required credentials
        without 
        """
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
        self.claude_api_key = os.environ["CLAUDE_API_KEY"]

        #did definations
        self.did_api_key = os.environ["DID_KEY"]
        self.did_url = DID_TALK_API

        #MusicGen 
        self.music_gen_api_key = os.environ["MUSIC_GEN_API_KEY"]
        self.music_gen_url = MUSIC_GEN_API

        #IdeoGram
        self.ideogram_url = IDEOGRAM_GENERATE_URL
        self.ideogram_api_key = os.environ["IDEOGRAM_API_KEY"]
        self.ideogram_edit_url = IDEOGRAM_EDIT_URL
        self.ideogram_remix_url = IDEOGRAM_REMIX_URL

        #Leonardo
        self.leonardo_image_generation_url = LEONARDO_IMAGE_GEN_URL
        self.leonardo_image_status_url = LEONARDO_IMAGE_STATUS_URL
        self.leonardo_api_key = os.environ["LEONARDO_API_KEY"]

        #AILAB
        self.hairstyle_url = AILAB_HAIRSTYLE_URL
        self.hairstyle_status_url = AILAB_STATUS_URL
        self.hairstyle_api_key = os.environ["AILAB_API_KEY"]

        #Runway
        self.runway_api_key = os.environ["RUNWAY_API_KEY"]

        #Luma
        self.lumalabs_api_key = os.environ["LUMAAI_API_KEY"]

    @abstractmethod
    def remote(self, data):
        """
        Executes the service's main functionality with the provided data.
        
        This abstract method must be implemented by all concrete service classes.
        It defines the primary interface for interacting with the service, processing
        input data and returning results from the external API or processing logic.
        
        Args:
            data: Input data specific to the service implementation
            
        Returns:
            The processed results from the service
        """
        pass
