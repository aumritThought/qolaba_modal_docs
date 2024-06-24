import os, tempfile
from elevenlabs import VoiceClone, generate, Voice, VoiceSettings, generate, set_api_key, VoiceDesign
from pydub import AudioSegment
from src.data_models.ModalAppSchemas import ElevenLabsParameters
from pydub import AudioSegment
from io import BytesIO
from src.utils.Globals import timing_decorator, upload_data_gcp, make_request, prepare_response
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_AUDIO_EXTENSION, ELEVENLABS_ERROR, CLONE_AUDIO_LEN_ERROR_MSG, CLONE_AUDIO_OPEN_ERROR_MSG

class ElvenLabsAudio(IService):
    def __init__(self) -> None:
        super().__init__()
        set_api_key(self.elevenlabs_api_key)

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

        voice = Voice(
            voice_id = parameters.audio_parameters.voice_id,
            settings=VoiceSettings(
                stability = parameters.audio_parameters.stability,
                similarity_boost = parameters.audio_parameters.similarity_boost,
                style = parameters.audio_parameters.style,
                use_speaker_boost = parameters.audio_parameters.use_speaker_boost,
            ),
        )

        audio = generate(
            text = parameters.prompt, voice = voice, model = "eleven_multilingual_v2"
        )

        audio_length = self.get_audio_length(BytesIO(audio))

        url = upload_data_gcp(audio, OUTPUT_AUDIO_EXTENSION)

        return prepare_response({"Audio url": url, "Audio length": audio_length}, [False], 0, 0)


    def voice_clone(self, parameters: ElevenLabsParameters) -> dict:

        list_of_saved_audios = []

        for url in parameters.clone_parameters.list_of_files:
            file_name = self.get_audio_file(url)
            
            audio_length = self.get_audio_length(file_name)

            if audio_length > 600 or audio_length < 60:
                raise Exception(
                    ELEVENLABS_ERROR, CLONE_AUDIO_LEN_ERROR_MSG
                )

            list_of_saved_audios.append(file_name)

        if len(list_of_saved_audios) == 0:
            raise Exception(ELEVENLABS_ERROR, CLONE_AUDIO_OPEN_ERROR_MSG)

        clone_settings = VoiceClone(
            name = parameters.clone_parameters.name,
            description = parameters.clone_parameters.description,
            files = list_of_saved_audios
        )

        voice = Voice.from_clone(clone_settings)

        for file in list_of_saved_audios:
            try:
                os.remove(file)
            except:
                pass

        return {
            "result": {"voice_id": voice.voice_id, "category": "cloned"},
            "Has_NSFW_Content": [False],
        }

    def voice_design(self, parameters: ElevenLabsParameters) -> dict:

        design = VoiceDesign(
            name=parameters.design_parameters.name,
            text="x" * 100,  # some random string is given as it is required
            voice_description=parameters.design_parameters.description,
            gender=parameters.design_parameters.gender,
            age=parameters.design_parameters.age,
            accent=parameters.design_parameters.accent,
            accent_strength=parameters.design_parameters.accent_strength,
        )
        voice = Voice.from_design(design)

        return {
            "result": {"voice_id": voice.voice_id, "category": "generated"},
            "Has_NSFW_Content": [False],
        }

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : ElevenLabsParameters = ElevenLabsParameters(**parameters)
        if parameters.clone == True:
            return self.voice_clone(parameters)
        elif parameters.voice_design == True:
            return self.voice_design(parameters)
        else:
            return self.generate_audio(parameters)
