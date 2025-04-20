import pytest
from unittest.mock import MagicMock
from src.FastAPIServer.services.ApiServices.OpenAIService import (
    DalleText2Image,
    GeminiAIImageCheck,
)
from src.utils.Constants import NSFW_CONTENT_DETECT_ERROR_MSG, IMAGE_GENERATION_ERROR


def test_make_dalle_api_request_success(mocker):
    # Mock the OpenAI client response
    mock_image_data = MagicMock()
    mock_image_data.url = "http://fake-image-url.com"

    mock_response = MagicMock()
    mock_response.data = [mock_image_data]

    mock_client = MagicMock()
    mock_client.images.generate.return_value = mock_response

    # Mock the HTTP request response
    mock_request_response = MagicMock()
    mock_request_response.content = b"fake-image-content"

    mocker.patch(
        "src.FastAPIServer.services.ApiServices.OpenAIService.make_request",
        return_value=mock_request_response,
    )

    # Create service instance with mocked client
    service = DalleText2Image()
    service.client = mock_client

    # Call the function
    result = service.make_dalle_api_request("test prompt", "1024x1024", "standard")

    # Verify
    mock_client.images.generate.assert_called_once_with(
        model="dall-e-3",
        prompt="test prompt",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    assert result == b"fake-image-content"


def test_make_dalle_api_request_nsfw_error(mocker):
    # Mock the OpenAI client to raise content policy violation
    mock_client = MagicMock()
    mock_client.images.generate.side_effect = Exception("content_policy_violation")

    # Create service instance with mocked client
    service = DalleText2Image()
    service.client = mock_client

    # Call the function and check exception
    with pytest.raises(Exception) as exc_info:
        service.make_dalle_api_request("nsfw prompt", "1024x1024", "standard")

    # Verify correct error message
    assert exc_info.value.args[0] == IMAGE_GENERATION_ERROR
    assert exc_info.value.args[1] == NSFW_CONTENT_DETECT_ERROR_MSG


def test_make_dalle_api_request_other_error(mocker):
    # Mock the OpenAI client to raise other error
    mock_client = MagicMock()
    mock_client.images.generate.side_effect = Exception("API Error")

    # Create service instance with mocked client
    service = DalleText2Image()
    service.client = mock_client

    # Call the function and check exception
    with pytest.raises(Exception) as exc_info:
        service.make_dalle_api_request("prompt", "1024x1024", "standard")

    # Verify
    assert exc_info.value.args[0] == "API Error"


def test_remote_success(mocker):
    # Mock the make_dalle_api_request method
    mocker.patch.object(
        DalleText2Image, "make_dalle_api_request", return_value=b"fake-image-content"
    )

    # Mock the prepare_response function
    prepared_response = {
        "result": [b"fake-image-content"],
        "Has_NSFW_Content": [False],
        "execution_time": 0,
        "cost": 0,
        "extension": "webp",
        "time": {"startup_time": 1, "runtime": 2},
    }

    mocker.patch(
        "src.FastAPIServer.services.ApiServices.OpenAIService.prepare_response",
        return_value=prepared_response,
    )

    # Create service instance
    service = DalleText2Image()

    # Create parameters
    parameters = {
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "quality": "standard",
        "batch": 1,
    }

    # Call the function
    result = service.remote(parameters)

    # Verify
    service.make_dalle_api_request.assert_called_once_with(
        "test prompt", "1024x1024", "standard"
    )

    from src.FastAPIServer.services.ApiServices.OpenAIService import prepare_response

    prepare_response.assert_called_once_with(
        [b"fake-image-content"], [False], 0, 0, "webp"
    )

    assert "result" in prepared_response


def test_remote_batch_processing(mocker):
    # Mock the make_dalle_api_request method
    mocker.patch.object(
        DalleText2Image, "make_dalle_api_request", return_value=b"fake-image-content"
    )

    # Mock the prepare_response function
    prepared_response = {
        "result": [b"fake-image-content", b"fake-image-content"],
        "Has_NSFW_Content": [False, False],
        "execution_time": 0,
        "cost": 0,
        "extension": "webp",
        "time": {"startup_time": 1, "runtime": 2},
    }

    mocker.patch(
        "src.FastAPIServer.services.ApiServices.OpenAIService.prepare_response",
        return_value=prepared_response,
    )

    # Create service instance
    service = DalleText2Image()

    # Create parameters with batch=2
    parameters = {
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "quality": "standard",
        "batch": 2,
    }

    # Call the function
    result = service.remote(parameters)

    # Verify
    assert service.make_dalle_api_request.call_count == 2

    from src.FastAPIServer.services.ApiServices.OpenAIService import prepare_response

    prepare_response.assert_called_once_with(
        [b"fake-image-content", b"fake-image-content"], [False, False], 0, 0, "webp"
    )

    assert "result" in prepared_response


# GeminiAIImageCheck tests
def test_gemini_init(mocker):
    # Mock dependencies
    mocker.patch(
        "src.FastAPIServer.services.IService.IService.__init__", return_value=None
    )
    mock_credentials = MagicMock()
    mock_project_id = "fake-project-id"
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(mock_credentials, mock_project_id),
    )
    mocker.patch("google.auth.transport.requests.Request")
    mocker.patch("src.FastAPIServer.services.ApiServices.OpenAIService.genai.Client")

    # Create service instance
    service = GeminiAIImageCheck()

    # Verify
    from src.FastAPIServer.services.ApiServices.OpenAIService import (
        COPYRIGHT_DETECTION_FUNCTION_CALLING_SCHEMA,
    )

    assert (
        service.function_calling_schema == COPYRIGHT_DETECTION_FUNCTION_CALLING_SCHEMA
    )
    mock_credentials.refresh.assert_called_once()
    from src.FastAPIServer.services.ApiServices.OpenAIService import genai

    genai.Client.assert_called_once_with(
        vertexai=True,
        project="marine-potion-404413",
        credentials=mock_credentials,
        location="us-central1",
    )
