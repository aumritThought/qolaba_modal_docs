import pytest
from unittest.mock import MagicMock
import google.auth
from src.FastAPIServer.services.ApiServices.ClaudeAIService import (
    PromptParrot,
    VideoPromptParrot,
)


@pytest.fixture
def mock_anthropic_response():
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Enhanced prompt text"
    return mock_response


@pytest.fixture
def mock_credentials():
    credentials = MagicMock()
    return credentials, "mock-project-id"


def test_prompt_parrot_init(mocker):
    # Arrange
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "mock-project-id"),
    )
    mocker.patch.object(google.auth.credentials.Credentials, "refresh")
    mocker.patch("anthropic.AnthropicVertex")

    # Act
    service = PromptParrot()

    # Assert
    assert service.client is not None
    google.auth.load_credentials_from_dict.assert_called_once()
    service.credentials.refresh.assert_called_once()


# def test_prompt_parrot_generate_prompt(mocker, mock_anthropic_response):
#     # Arrange
#     mocker.patch('google.auth.load_credentials_from_dict', return_value=(MagicMock(), "mock-project-id"))
#     mocker.patch.object(google.auth.credentials.Credentials, 'refresh')
#     mock_client = MagicMock()
#     mock_client.messages.create.return_value = mock_anthropic_response
#     mocker.patch('anthropic.AnthropicVertex', return_value=mock_client)

#     service = PromptParrot()
#     query = "Generate a prompt about mountains"

#     # Act
#     result = service.generate_prompt(query)

#     # Assert
#     assert result == "Enhanced prompt text"
#     mock_client.messages.create.assert_called_with(
#         model="claude-3-haiku@20240307",
#         messages=[
#             {"role": "user", "content": query}
#         ],
#         temperature=0.5,
#         stream=False,
#         max_tokens=1000
#     )


def test_prompt_parrot_remote(mocker):
    # Arrange
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "mock-project-id"),
    )
    mocker.patch.object(google.auth.credentials.Credentials, "refresh")
    mocker.patch("anthropic.AnthropicVertex")

    service = PromptParrot()
    mocker.patch.object(service, "generate_prompt", return_value="Enhanced prompt")
    mocker.patch(
        "src.utils.Globals.prepare_response",
        return_value={"result": ["Enhanced prompt"]},
    )
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = "Enhanced prompt"
    mock_executor.__enter__.return_value.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ThreadPoolExecutor", return_value=mock_executor)

    parameters = {"prompt": "Test prompt", "batch": 2}

    # Mock the imported constants
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ClaudeAIService.BASE_PROMPT_FOR_GENERATION",
        "Base prompt template: ",
    )

    # Act
    result = service.remote(parameters)

    # Assert
    # prepare_response.assert_called_once()
    assert mock_executor.__enter__.return_value.submit.call_count == 2


def test_video_prompt_parrot_init(mocker):
    # Arrange
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "mock-project-id"),
    )
    mocker.patch.object(google.auth.credentials.Credentials, "refresh")
    mocker.patch("anthropic.AnthropicVertex")

    # Act
    service = VideoPromptParrot()

    # Assert
    assert service.client is not None
    google.auth.load_credentials_from_dict.assert_called_once()
    service.credentials.refresh.assert_called_once()


# def test_video_prompt_parrot_generate_prompt(mocker, mock_anthropic_response):
#     # Arrange
#     mocker.patch('google.auth.load_credentials_from_dict', return_value=(MagicMock(), "mock-project-id"))
#     mocker.patch.object(google.auth.credentials.Credentials, 'refresh')
#     mock_client = MagicMock()
#     mock_client.messages.create.return_value = mock_anthropic_response
#     mocker.patch('anthropic.AnthropicVertex', return_value=mock_client)

#     service = VideoPromptParrot()
#     query = "Generate a video prompt about mountains"

#     # Act
#     result = service.generate_prompt(query)

#     # Assert
#     assert result == "Enhanced prompt text"
#     mock_client.messages.create.assert_called_with(
#         model="claude-3-haiku@20240307",
#         messages=[
#             {"role": "user", "content": query}
#         ],
#         temperature=0.5,
#         stream=False,
#         max_tokens=1000
#     )


def test_video_prompt_parrot_remote(mocker):
    # Arrange
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "mock-project-id"),
    )
    mocker.patch.object(google.auth.credentials.Credentials, "refresh")
    mocker.patch("anthropic.AnthropicVertex")

    service = VideoPromptParrot()
    mocker.patch.object(
        service, "generate_prompt", return_value="Enhanced video prompt"
    )
    mocker.patch(
        "src.utils.Globals.prepare_response",
        return_value={"result": ["Enhanced video prompt"]},
    )
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = "Enhanced video prompt"
    mock_executor.__enter__.return_value.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ThreadPoolExecutor", return_value=mock_executor)

    parameters = {"prompt": "Test video prompt", "batch": 2}

    # Mock the imported constants
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.ClaudeAIService.BASE_PROMPT_FOR_VIDEO_GENERATION",
        "Base video prompt template: ",
    )

    # Act
    result = service.remote(parameters)

    # Assert
    # prepare_response.assert_called_once()
    assert mock_executor.__enter__.return_value.submit.call_count == 2
