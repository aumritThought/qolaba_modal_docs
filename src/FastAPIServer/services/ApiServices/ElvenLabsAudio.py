import tempfile, random, string, threading
from elevenlabs import Voice, VoiceSettings, play
from pydub import AudioSegment
from src.data_models.ModalAppSchemas import ElevenLabsParameters
from pydub import AudioSegment
from io import BytesIO
from src.utils.Globals import timing_decorator, upload_data_gcp, make_request, prepare_response
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_AUDIO_EXTENSION, ELEVENLABS_ERROR, CLONE_AUDIO_LEN_ERROR_MSG, CLONE_AUDIO_OPEN_ERROR_MSG
from elevenlabs import ElevenLabs

voice_genration_lock = threading.Lock()

class ElvenLabsAudio(IService):
    def __init__(self) -> None:
        super().__init__()
        self.client = ElevenLabs(
            api_key=self.elevenlabs_api_key
        )

    def generate_random_string(self, length=6):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

    def get_audio_length(self, file_name: str) -> int:
        audio = AudioSegment.from_file(file_name)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000
        return duration_seconds
    
    def get_audio_file(self, url : str) -> str:
        temp_file_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

        response = make_request(url, "GET")

        with open(temp_file_audio.name, "wb") as f:
            f.write(response.content)
        return temp_file_audio.name


    def generate_audio(self, parameters: ElevenLabsParameters) -> dict:
        available_voice_ids = []
        voices = self.client.voices.get_all().voices
        for voice in voices:
            available_voice_ids.append(voice.voice_id)

        with voice_genration_lock:
            if(not(parameters.audio_parameters.public_id == None or parameters.audio_parameters.public_id == "")):
                if(not(parameters.audio_parameters.voice_id in available_voice_ids)):
                    for voice in self.client.voices.get_all().voices:
                        if(voice.category == "professional"):
                            self.client.voices.delete(voice.voice_id)
                            self.client.voices.add_sharing_voice(public_user_id=parameters.audio_parameters.public_id, voice_id=parameters.audio_parameters.voice_id, new_name=self.generate_random_string())
                            break

            voice = Voice(
                voice_id = parameters.audio_parameters.voice_id,
                settings=VoiceSettings(
                    stability = parameters.audio_parameters.stability,
                    similarity_boost = parameters.audio_parameters.similarity_boost,
                    style = parameters.audio_parameters.style,
                    use_speaker_boost = parameters.audio_parameters.use_speaker_boost,
                ),
            )
            
            audio = self.client.generate(
                text = parameters.prompt, voice = voice, model = "eleven_multilingual_v2"
            )

        audio = b"".join(audio)
        audio_length = self.get_audio_length(BytesIO(audio))

        url = upload_data_gcp(audio, OUTPUT_AUDIO_EXTENSION)

        return prepare_response({"Audio url": url, "Audio length": audio_length}, [False], 0, 0)


    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : ElevenLabsParameters = ElevenLabsParameters(**parameters)
        if parameters.clone == True:
            raise NotImplementedError()
        elif parameters.voice_design == True:
            raise NotImplementedError()
        else:
            return self.generate_audio(parameters)
