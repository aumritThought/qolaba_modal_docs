import random
import string
import threading
from elevenlabs import Voice, VoiceSettings
from pydub import AudioSegment
from src.data_models.ModalAppSchemas import ElevenLabsParameters
from io import BytesIO
from src.utils.Globals import timing_decorator, upload_data_gcp, prepare_response
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_AUDIO_EXTENSION
from elevenlabs import ElevenLabs

voice_genration_lock = threading.Lock()


class ElvenLabsAudio(IService):
    def __init__(self) -> None:
        """
        Initializes the ElevenLabs audio service with API credentials.

        Sets up the ElevenLabs client using the API key inherited from the
        IService parent class, establishing the connection for subsequent
        audio generation requests.
        """
        super().__init__()
        self.client = ElevenLabs(api_key=self.elevenlabs_api_key)

    def generate_random_string(self, length=6):
        """
        Creates a random alphanumeric string of specified length.

        Used for generating unique identifiers for voices when adding
        them to the ElevenLabs voice library, ensuring no naming conflicts.

        Args:
            length (int): Length of the random string to generate

        Returns:
            str: Random alphanumeric string
        """
        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choice(characters) for _ in range(length))
        return random_string

    def get_audio_length(self, file_name: str) -> int:
        """
        Calculates the duration of an audio file in seconds.

        This function loads an audio file using pydub and determines
        its length, which is useful for reporting to clients or for
        downstream processing.

        Args:
            file_name (str): Path to the audio file

        Returns:
            int: Duration of the audio in seconds
        """
        audio = AudioSegment.from_file(file_name)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000
        return duration_seconds

    def generate_audio(self, parameters: ElevenLabsParameters) -> dict:
        """
        Generates audio from text using ElevenLabs voices.

        This function manages voice library access and handles text-to-speech
        conversion. If the requested voice is not available in the user's
        voice library, it attempts to add it by first removing an existing
        professional voice to stay within ElevenLabs' voice limit.

        The function uses a lock to ensure thread safety during voice management
        operations, as voice addition/deletion must be performed sequentially.

        Args:
            parameters (ElevenLabsParameters): Text and voice configuration parameters

        Returns:
            dict: Standardized response with audio URL and duration
        """
        available_voice_ids = []
        voices = self.client.voices.get_all().voices
        for voice in voices:
            available_voice_ids.append(voice.voice_id)

        with voice_genration_lock:
            if not (
                parameters.audio_parameters.public_id == None
                or parameters.audio_parameters.public_id == ""
            ):
                if parameters.audio_parameters.voice_id not in available_voice_ids:
                    for voice in self.client.voices.get_all().voices:
                        if voice.category == "professional":
                            self.client.voices.delete(voice.voice_id)
                            self.client.voices.add_sharing_voice(
                                public_user_id=parameters.audio_parameters.public_id,
                                voice_id=parameters.audio_parameters.voice_id,
                                new_name=self.generate_random_string(),
                            )
                            break

            voice = Voice(
                voice_id=parameters.audio_parameters.voice_id,
                settings=VoiceSettings(
                    stability=parameters.audio_parameters.stability,
                    similarity_boost=parameters.audio_parameters.similarity_boost,
                    style=parameters.audio_parameters.style,
                    use_speaker_boost=parameters.audio_parameters.use_speaker_boost,
                ),
            )

            audio = self.client.generate(
                text=parameters.prompt, voice=voice, model="eleven_multilingual_v2"
            )

        audio = b"".join(audio)
        audio_length = self.get_audio_length(BytesIO(audio))

        url = upload_data_gcp(audio, OUTPUT_AUDIO_EXTENSION)

        return prepare_response(
            {"Audio url": url, "Audio length": audio_length}, [False], 0, 0
        )

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for audio generation service requests.

        Routes requests to appropriate handlers based on the requested
        operation type (standard generation, voice cloning, or voice design).
        Currently only standard audio generation is implemented.

        Args:
            parameters (dict): Request parameters for audio generation

        Returns:
            dict: Standardized response with generated audio information

        Raises:
            NotImplementedError: For unimplemented features (clone, voice_design)
        """
        parameters: ElevenLabsParameters = ElevenLabsParameters(**parameters)
        if parameters.clone == True:
            raise NotImplementedError()
        elif parameters.voice_design == True:
            raise NotImplementedError()
        else:
            return self.generate_audio(parameters)
