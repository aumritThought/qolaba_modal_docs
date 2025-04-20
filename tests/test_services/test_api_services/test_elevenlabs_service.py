from unittest.mock import MagicMock
from src.FastAPIServer.services.ApiServices.ElvenLabsAudio import ElvenLabsAudio
from src.data_models.ModalAppSchemas import ElevenLabsParameters, AudioParameters


def test_init(mocker):
    # Mock the ElevenLabs class
    mocker.patch("src.FastAPIServer.services.ApiServices.ElvenLabsAudio.ElevenLabs")

    # Mock the IService parent class to provide API key
    mocker.patch(
        "src.FastAPIServer.services.IService.IService.__init__", return_value=None
    )
    mocker.patch.object(
        ElvenLabsAudio, "elevenlabs_api_key", "mock_api_key", create=True
    )

    # Create instance and verify client initialization
    service = ElvenLabsAudio()

    from src.FastAPIServer.services.ApiServices.ElvenLabsAudio import ElevenLabs

    ElevenLabs.assert_called_once_with(api_key="mock_api_key")


def test_generate_random_string():
    service = ElvenLabsAudio()

    # Test default length
    random_str = service.generate_random_string()
    assert len(random_str) == 6
    assert all(c.isalnum() for c in random_str)

    # Test custom length
    random_str = service.generate_random_string(10)
    assert len(random_str) == 10
    assert all(c.isalnum() for c in random_str)


def test_get_audio_length(mocker):
    service = ElvenLabsAudio()

    # Mock AudioSegment
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 5000  # 5000 ms = 5 seconds

    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ElvenLabsAudio.AudioSegment.from_file",
        return_value=mock_audio,
    )

    # Test audio length calculation
    duration = service.get_audio_length("test_file.mp3")
    assert duration == 5.0


def test_generate_audio(mocker):
    service = ElvenLabsAudio()

    # Mock client and voices
    mock_voice = MagicMock()
    mock_voice.voice_id = "voice123"
    mock_voice.category = "professional"

    mock_voices = MagicMock()
    mock_voices.voices = [mock_voice]

    service.client = MagicMock()
    service.client.voices.get_all.return_value = mock_voices
    service.client.generate.return_value = [b"audio", b"data"]

    # Mock get_audio_length
    mocker.patch.object(service, "get_audio_length", return_value=10.5)

    # Mock GCP upload
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ElvenLabsAudio.upload_data_gcp",
        return_value="https://storage.url/audio.mp3",
    )

    # Mock prepare_response
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ElvenLabsAudio.prepare_response",
        return_value={"result": "success"},
    )

    # Create test parameters
    audio_params = AudioParameters(
        voice_id="voice123",
        stability=0.5,
        similarity_boost=0.7,
        style=0.3,
        use_speaker_boost=True,
        public_id=None,
    )

    params = ElevenLabsParameters(
        prompt="Test text",
        audio_parameters=audio_params,
        clone=False,
        voice_design=False,
    )

    # Test standard voice generation
    result = service.generate_audio(params)

    # Verify function calls
    service.client.generate.assert_called_once()
    from src.FastAPIServer.services.ApiServices.ElvenLabsAudio import (
        upload_data_gcp,
        prepare_response,
    )

    upload_data_gcp.assert_called_once()
    prepare_response.assert_called_once()
    assert result == {"result": "success"}


def test_generate_audio_with_public_id(mocker):
    service = ElvenLabsAudio()

    # Mock client and voices
    mock_voice = MagicMock()
    mock_voice.voice_id = "existingVoice"
    mock_voice.category = "professional"

    mock_voices = MagicMock()
    mock_voices.voices = [mock_voice]

    service.client = MagicMock()
    service.client.voices.get_all.return_value = mock_voices
    service.client.generate.return_value = [b"audio", b"data"]

    # Mock get_audio_length
    mocker.patch.object(service, "get_audio_length", return_value=10.5)

    # Mock generate_random_string
    mocker.patch.object(service, "generate_random_string", return_value="random123")

    # Mock GCP upload
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ElvenLabsAudio.upload_data_gcp",
        return_value="https://storage.url/audio.mp3",
    )

    # Mock prepare_response
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ElvenLabsAudio.prepare_response",
        return_value={"result": "success"},
    )

    # Create test parameters with public_id and a non-existing voice
    audio_params = AudioParameters(
        voice_id="newVoice",
        stability=0.5,
        similarity_boost=0.7,
        style=0.3,
        use_speaker_boost=True,
        public_id="user123",
    )

    params = ElevenLabsParameters(
        prompt="Test text",
        audio_parameters=audio_params,
        clone=False,
        voice_design=False,
    )

    # Test voice addition flow
    result = service.generate_audio(params)

    # Verify voice deletion and addition
    service.client.voices.delete.assert_called_once_with("existingVoice")
    service.client.voices.add_sharing_voice.assert_called_once_with(
        public_user_id="user123", voice_id="newVoice", new_name="random123"
    )
    assert result == {"result": "success"}


# def test_remote_standard(mocker):
#     service = ElvenLabsAudio()

#     # Mock generate_audio
#     mocker.patch.object(service, 'generate_audio', return_value={"result": "success"})

#     # Test standard audio generation
#     params = {
#         "prompt": "Test text",
#         "audio_parameters": {
#             "voice_id": "voice123",
#             "stability": 0.5,
#             "similarity_boost": 0.7,
#             "style": 0.3,
#             "use_speaker_boost": True
#         },
#         "clone": False,
#         "voice_design": False
#     }

#     result = service.remote(params)

#     # Verify function calls
#     service.generate_audio.assert_called_once()
#     assert result == {"result": "success"}

# def test_remote_clone(mocker):
#     service = ElvenLabsAudio()

#     # Test voice cloning (should raise NotImplementedError)
#     params = {
#         "prompt": "Test text",
#         "audio_parameters": {
#             "voice_id": "voice123"
#         },
#         "clone": True,
#         "voice_design": False
#     }

#     with pytest.raises(NotImplementedError):
#         service.remote(params)

# def test_remote_voice_design(mocker):
#     service = ElvenLabsAudio()

#     # Test voice design (should raise NotImplementedError)
#     params = {
#         "prompt": "Test text",
#         "audio_parameters": {
#             "voice_id": "voice123"
#         },
#         "clone": False,
#         "voice_design": True
#     }

#     with pytest.raises(NotImplementedError):
#         service.remote(params)
