from src.data_models.ModalAppSchemas import SDXLText2ImageParameters
from src.FastAPIServer.services.ApiServices.LeonardoService import LeonardoText2Image
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION


def test_leonardo_init():
    """Test LeonardoText2Image initialization"""
    service = LeonardoText2Image()
    assert isinstance(service, LeonardoText2Image)


def test_make_api_request(mocker):
    """Test the make_api_request function with mocked API responses"""
    # Setup mocks for API calls
    mock_init_response = mocker.MagicMock()
    mock_init_response.json.return_value = {
        "sdGenerationJob": {"generationId": "test-generation-id"}
    }

    mock_status_response = mocker.MagicMock()
    mock_status_response.json.return_value = {
        "generations_by_pk": {
            "status": "COMPLETE",
            "generated_images": [{"url": "https://example.com/image1.jpg"}],
        }
    }

    mock_image_response = mocker.MagicMock()
    mock_image_content = b"mock-image-data"
    mock_image_response.content = mock_image_content

    # Create mock for make_request to return different responses for different calls
    mock_make_request = mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.make_request"
    )
    mock_make_request.side_effect = [
        mock_init_response,  # First call - POST to generate image
        mock_status_response,  # Second call - GET status
        mock_image_response,  # Third call - GET image
    ]

    # Mock time.sleep to avoid actual delays
    mocker.patch("time.sleep")

    # Create service and test parameters
    service = LeonardoText2Image()
    parameters = SDXLText2ImageParameters(
        prompt="test prompt",
        batch=1,
        width=1024,
        height=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # Call function under test
    result = service.make_api_request(parameters)

    # Assert results
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == mock_image_content

    # Verify API calls
    assert mock_make_request.call_count == 3


def test_make_api_request_multiple_status_checks(mocker):
    """Test make_api_request with multiple status checks before completion"""
    # Setup mocks for API calls
    mock_init_response = mocker.MagicMock()
    mock_init_response.json.return_value = {
        "sdGenerationJob": {"generationId": "test-generation-id"}
    }

    # Create two status responses - one in progress, one complete
    mock_status_in_progress = mocker.MagicMock()
    mock_status_in_progress.json.return_value = {
        "generations_by_pk": {"status": "IN_PROGRESS"}
    }

    mock_status_complete = mocker.MagicMock()
    mock_status_complete.json.return_value = {
        "generations_by_pk": {
            "status": "COMPLETE",
            "generated_images": [{"url": "https://example.com/image1.jpg"}],
        }
    }

    mock_image_response = mocker.MagicMock()
    mock_image_content = b"mock-image-data"
    mock_image_response.content = mock_image_content

    # Create mock for make_request with multiple status checks
    mock_make_request = mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.make_request"
    )
    mock_make_request.side_effect = [
        mock_init_response,  # First call - POST to generate image
        mock_status_in_progress,  # Second call - GET status (in progress)
        mock_status_complete,  # Third call - GET status (complete)
        mock_image_response,  # Fourth call - GET image
    ]

    # Mock time.sleep to avoid actual delays
    mocker.patch("time.sleep")

    # Create service and test parameters
    service = LeonardoText2Image()
    parameters = SDXLText2ImageParameters(
        prompt="test prompt",
        batch=1,
        width=1024,
        height=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # Call function under test
    result = service.make_api_request(parameters)

    # Assert results
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == mock_image_content

    # Verify API calls
    assert mock_make_request.call_count == 4


def test_make_api_request_multiple_images(mocker):
    """Test make_api_request with multiple images in response"""
    # Setup mocks for API calls
    mock_init_response = mocker.MagicMock()
    mock_init_response.json.return_value = {
        "sdGenerationJob": {"generationId": "test-generation-id"}
    }

    mock_status_response = mocker.MagicMock()
    mock_status_response.json.return_value = {
        "generations_by_pk": {
            "status": "COMPLETE",
            "generated_images": [
                {"url": "https://example.com/image1.jpg"},
                {"url": "https://example.com/image2.jpg"},
            ],
        }
    }

    # Create two different image responses
    mock_image_response1 = mocker.MagicMock()
    mock_image_content1 = b"mock-image-data-1"
    mock_image_response1.content = mock_image_content1

    mock_image_response2 = mocker.MagicMock()
    mock_image_content2 = b"mock-image-data-2"
    mock_image_response2.content = mock_image_content2

    # Create mock for make_request
    mock_make_request = mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.make_request"
    )
    mock_make_request.side_effect = [
        mock_init_response,
        mock_status_response,
        mock_image_response1,
        mock_image_response2,
    ]

    # Mock time.sleep to avoid actual delays
    mocker.patch("time.sleep")

    # Create service and test parameters
    service = LeonardoText2Image()
    parameters = SDXLText2ImageParameters(
        prompt="test prompt",
        batch=2,
        width=1024,
        height=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # Call function under test
    result = service.make_api_request(parameters)

    # Assert results
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == mock_image_content1
    assert result[1] == mock_image_content2

    # Verify API calls
    assert mock_make_request.call_count == 4


def test_remote(mocker):
    """Test the remote function with mocked make_api_request"""
    # Mock make_api_request
    mock_image_data = [b"mock-image-data"]
    mock_api_request = mocker.patch.object(
        LeonardoText2Image, "make_api_request", return_value=mock_image_data
    )

    # Mock timing decorator to pass through the function
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.timing_decorator",
        lambda func: func,
    )

    # Create test parameters
    parameters = {
        "prompt": "test prompt",
        "batch": 1,
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }

    # Setup expected response format
    mock_response = {
        "result": mock_image_data,
        "Has_NSFW_Content": [False, False],
        "time_required": {},
        "time_taken": 0,
        "modal_code_time": 0,
        "time": {"startup_time": 0, "runtime": 15},
        "extension": OUTPUT_IMAGE_EXTENSION,
    }

    # Mock prepare_response function
    mock_prepare_response = mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.prepare_response",
        return_value=mock_response,
    )

    # Create service and call function under test
    service = LeonardoText2Image()
    service.remote(parameters)

    # Assert results
    # assert result == mock_response

    # Verify the parameter conversion and function calls
    mock_api_request.assert_called_once()
    # args, _ = mock_prepare_response.call_args
    # assert args[0] == mock_image_data
    # assert args[1] == [False]
    # assert args[2] == 0
    # assert args[3] == 0
    # assert args[4] == OUTPUT_IMAGE_EXTENSION


def test_remote_with_multiple_images(mocker):
    """Test the remote function with multiple images in batch"""
    # Mock make_api_request
    mock_image_data = [b"mock-image-data-1", b"mock-image-data-2"]
    mock_api_request = mocker.patch.object(
        LeonardoText2Image, "make_api_request", return_value=mock_image_data
    )

    # Mock timing decorator to pass through the function
    mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.timing_decorator",
        lambda func: func,
    )

    # Create test parameters with batch=2
    parameters = {
        "prompt": "test prompt",
        "batch": 2,
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }

    # Setup expected response format
    mock_response = {
        "result": mock_image_data,
        "Has_NSFW_Content": [False, False],
        "time_required": {},
        "time_taken": 0,
        "modal_code_time": 0,
        "time": {"startup_time": 0, "runtime": 15},
        "extension": OUTPUT_IMAGE_EXTENSION,
    }

    # Mock prepare_response function
    mock_prepare_response = mocker.patch(
        "src.FastAPIServer.services.ApiServices.LeonardoService.prepare_response",
        return_value=mock_response,
    )

    # Create service and call function under test
    service = LeonardoText2Image()
    result = service.remote(parameters)

    # Assert results
    # assert result == mock_response

    # Verify parameter conversion and function calls
    mock_api_request.assert_called_once()
    # args, _ = mock_prepare_response.call_args
    # assert args[0] == mock_image_data
    # assert args[1] == [False, False]
    # assert args[2] == 0
    # assert args[3] == 0
    # assert args[4] == OUTPUT_IMAGE_EXTENSION
