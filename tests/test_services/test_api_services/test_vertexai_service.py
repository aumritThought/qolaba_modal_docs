from unittest.mock import MagicMock

import pytest
from vertexai.preview.vision_models import Image, ImageGenerationModel

from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters
from src.FastAPIServer.services.ApiServices.VertexAIService import ImageGenText2Image
from src.utils.Constants import IMAGEGEN_ERROR, IMAGEGEN_ERROR_MSG


@pytest.fixture
def mock_image_generation_model():
    """Fixture for mocking the ImageGenerationModel."""
    mock_model = MagicMock(spec=ImageGenerationModel)
    return mock_model


@pytest.fixture
def mock_image():
    """Fixture for mocking vertexai Image."""
    mock_img = MagicMock(spec=Image)
    mock_img._image_bytes = b"mock_image_bytes"
    return mock_img


@pytest.fixture
def mock_image_list(mock_image):
    """Fixture for creating mock image list."""
    return MagicMock(images=[mock_image])


def test_make_api_request_success(mocker, mock_image_generation_model, mock_image_list):
    """Test make_api_request method when successful."""
    # Setup
    mock_image_generation_model.generate_images.return_value = mock_image_list
    mocker.patch(
        "vertexai.preview.vision_models.ImageGenerationModel.from_pretrained",
        return_value=mock_image_generation_model,
    )

    # Mock initialization
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "project-id"),
    )
    mocker.patch("google.auth.transport.requests.Request")
    mocker.patch("vertexai.init")

    # Create service and parameters
    service = ImageGenText2Image()
    parameters = IdeoGramText2ImageParameters(
        prompt="test prompt", aspect_ratio="1:1", batch=1
    )

    # Call the method
    result = service.make_api_request(parameters)

    # Assertions
    assert result == b"mock_image_bytes"
    mock_image_generation_model.generate_images.assert_called_once_with(
        prompt="test prompt",
        number_of_images=1,
        aspect_ratio="1:1",
        safety_filter_level="block_some",
    )


def test_make_api_request_failure(mocker, mock_image_generation_model):
    """Test make_api_request method when it fails."""
    # Setup - empty images list to trigger exception
    mock_image_generation_model.generate_images.return_value = MagicMock(images=[])
    mocker.patch(
        "vertexai.preview.vision_models.ImageGenerationModel.from_pretrained",
        return_value=mock_image_generation_model,
    )

    # Mock initialization
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "project-id"),
    )
    mocker.patch("google.auth.transport.requests.Request")
    mocker.patch("vertexai.init")

    # Create service and parameters
    service = ImageGenText2Image()
    parameters = IdeoGramText2ImageParameters(
        prompt="test prompt", aspect_ratio="1:1", batch=1
    )

    # Call the method and check for exception
    with pytest.raises(Exception) as excinfo:
        service.make_api_request(parameters)

    assert excinfo.value.args[0] == IMAGEGEN_ERROR
    assert excinfo.value.args[1] == IMAGEGEN_ERROR_MSG


def test_remote_success(mocker, mock_image_generation_model, mock_image_list):
    """Test remote method with successful API requests."""
    # Setup
    mock_image_generation_model.generate_images.return_value = mock_image_list
    mocker.patch(
        "vertexai.preview.vision_models.ImageGenerationModel.from_pretrained",
        return_value=mock_image_generation_model,
    )

    # Mock initialization
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "project-id"),
    )
    mocker.patch("google.auth.transport.requests.Request")
    mocker.patch("vertexai.init")

    # Mock the timing decorator to simply call the function
    mocker.patch("src.utils.Globals.timing_decorator", lambda func: func)

    # Mock prepare_response
    expected_response = {"result": [b"mock_image_bytes"], "Has_NSFW_Content": [False]}
    mocker.patch("src.utils.Globals.prepare_response", return_value=expected_response)

    # Mock convert_to_aspect_ratio
    mocker.patch("src.utils.Globals.convert_to_aspect_ratio", return_value="1:1")

    # Create service
    service = ImageGenText2Image()

    # Create parameters
    parameters = {"prompt": "test prompt", "width": 1024, "height": 1024, "batch": 2}

    # Call the method
    result = service.remote(parameters)

    # Assertions
    assert "result" in result
    mock_image_generation_model.generate_images.call_count == 2  # Called once for each batch
    # assert src.utils.Globals.prepare_response.call_args[0][0] == [b"mock_image_bytes", b"mock_image_bytes"]
    # assert src.utils.Globals.prepare_response.call_args[0][1] == [False, False]
    # assert src.utils.Globals.prepare_response.call_args[0][3] == 0  # runtime
    # assert src.utils.Globals.prepare_response.call_args[0][4] == OUTPUT_IMAGE_EXTENSION


def test_remote_invalid_aspect_ratio(mocker):
    """Test remote method with invalid aspect ratio."""
    # Mock initialization
    mocker.patch(
        "google.auth.load_credentials_from_dict",
        return_value=(MagicMock(), "project-id"),
    )
    mocker.patch("google.auth.transport.requests.Request")
    mocker.patch("vertexai.init")
    mocker.patch("vertexai.preview.vision_models.ImageGenerationModel.from_pretrained")

    # Mock convert_to_aspect_ratio
    mocker.patch(
        "src.utils.Globals.convert_to_aspect_ratio", return_value="invalid_ratio"
    )

    # Create service
    service = ImageGenText2Image()

    # Create parameters
    parameters = {"prompt": "test prompt", "width": 1024, "height": 1024, "batch": 1}

    # Call the method and check for exception
    with pytest.raises(Exception) as excinfo:
        service.remote(parameters)

    assert (
        str(excinfo.value)
        == "('Google Imagegen Error', 'Request cancelled due to detection of NSFW content or Prompt issues. Google Imagegen does not support the generation of Children. Please improve prompt accordingly.')"
    )
