import pytest
from unittest.mock import MagicMock
import io, base64
from PIL import Image
import numpy as np
from src.data_models.ModalAppSchemas import RunwayImage2VideoParameters, PromptImage
from src.FastAPIServer.services.ApiServices.RunWayService import RunwayImage2Video
from src.utils.Constants import RUNWAY_ERROR, RUNWAY_ERROR_MSG, RUNWAY_POSITION_ERROR_MSG, OUTPUT_VIDEO_EXTENSION

def test_download_and_convert_to_base64(mocker):
    # Mock image
    mock_image = Image.new('RGB', (100, 100), color='red')
    
    # Mock the get_image_from_url function
    mocker.patch('src.FastAPIServer.services.ApiServices.RunWayService.get_image_from_url', 
                 return_value=mock_image)
    
    # Create service
    service = RunwayImage2Video()
    
    # Call the function
    result = service.download_and_convert_to_base64("http://example.com/image.jpg")
    
    # Assert
    assert isinstance(result, str)
    assert result.startswith("data:image/jpg;base64,")
    assert len(result) > 20  # Base64 string should have some length

def test_generate_image_success(mocker):
    # Mock the RunwayML client and task
    mock_client = MagicMock()
    mock_task = MagicMock()
    mock_task.id = "test_task_id"
    mock_task.status = "SUCCEEDED"
    mock_task.output = ["output_url"]
    mock_task.model_dump.return_value = {"status": "SUCCEEDED"}
    
    # Set up client mock responses
    mock_client.image_to_video.create.return_value = mock_task
    mock_client.tasks.retrieve.return_value = mock_task
    
    # Mock response from make_request
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.RunWayService.make_request', 
                 return_value=mock_response)
    
    # Mock the download_and_convert_to_base64 method
    mocker.patch.object(RunwayImage2Video, 'download_and_convert_to_base64', 
                        return_value="data:image/jpg;base64,abc123")
    
    # Create service with mocked client
    service = RunwayImage2Video()
    service.client = mock_client
    
    # Create parameters with single image
    params = RunwayImage2VideoParameters(
        prompt="test prompt",
        aspect_ratio="1280:768",
        duration=5,
        batch=1,
        file_url=[PromptImage(uri="http://example.com/image.jpg", position="first")]
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assertions
    assert result == b"video_content"
    mock_client.image_to_video.create.assert_called_once_with(
        model='gen3a_turbo',
        prompt_image=params.model_dump()["file_url"],
        prompt_text="test prompt",
        duration=5,
        ratio="1280:768"
    )

def test_generate_image_with_two_images(mocker):
    # Mock the RunwayML client and task
    mock_client = MagicMock()
    mock_task = MagicMock()
    mock_task.id = "test_task_id"
    mock_task.status = "SUCCEEDED"
    mock_task.output = ["output_url"]
    mock_task.model_dump.return_value = {"status": "SUCCEEDED"}
    
    # Set up client mock responses
    mock_client.image_to_video.create.return_value = mock_task
    mock_client.tasks.retrieve.return_value = mock_task
    
    # Mock response from make_request
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.RunWayService.make_request', 
                 return_value=mock_response)
    
    # Mock the download_and_convert_to_base64 method
    download_convert_mock = mocker.patch.object(
        RunwayImage2Video, 'download_and_convert_to_base64',
        return_value="data:image/jpg;base64,abc123"
    )
    
    # Create service with mocked client
    service = RunwayImage2Video()
    service.client = mock_client
    
    # Create parameters with two images
    params = RunwayImage2VideoParameters(
        prompt="test prompt",
        aspect_ratio="1280:768",
        duration=5,
        batch=1,
        file_url=[
            PromptImage(uri="http://example.com/start.jpg", position="first"),
            PromptImage(uri="http://example.com/end.jpg", position="last")
        ]
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assertions
    assert result == b"video_content"
    assert download_convert_mock.call_count == 2
    mock_client.image_to_video.create.assert_called_once()

def test_generate_image_with_same_position_error(mocker):
    # Mock the download_and_convert_to_base64 method
    mocker.patch.object(RunwayImage2Video, 'download_and_convert_to_base64', 
                        return_value="data:image/jpg;base64,abc123")
    
    # Create service
    service = RunwayImage2Video()
    
    # Create parameters with two images at the same position
    params = RunwayImage2VideoParameters(
        prompt="test prompt",
        aspect_ratio="1280:768",
        duration=5,
        batch=1,
        file_url=[
            PromptImage(uri="http://example.com/image1.jpg", position="first"),
            PromptImage(uri="http://example.com/image2.jpg", position="first")
        ]
    )
    
    # Call the function and expect exception
    with pytest.raises(Exception) as exc_info:
        service.generate_image(params)
    
    # Verify the error message
    assert exc_info.value.args[0] == RUNWAY_ERROR
    assert exc_info.value.args[1] == RUNWAY_POSITION_ERROR_MSG

def test_generate_image_no_images_error(mocker):
    # Create service
    service = RunwayImage2Video()
    
    # Create parameters with no images
    params = RunwayImage2VideoParameters(
        prompt="test prompt",
        aspect_ratio="1280:768",
        duration=5,
        batch=1,
        file_url=[]
    )
    
    # Call the function and expect exception
    with pytest.raises(Exception) as exc_info:
        service.generate_image(params)
    
    # Verify the error message
    assert exc_info.value.args[0] == RUNWAY_ERROR
    assert exc_info.value.args[1] == RUNWAY_ERROR_MSG

def test_generate_image_processing_failure(mocker):
    # Mock the RunwayML client and task
    mock_client = MagicMock()
    mock_task = MagicMock()
    mock_task.id = "test_task_id"
    
    # First return PENDING, then FAILED
    mock_task_pending = MagicMock()
    mock_task_pending.status = "PENDING"
    
    mock_task_failed = MagicMock()
    mock_task_failed.status = "FAILED"
    mock_task_failed.model_dump.return_value = {"status": "FAILED", "error": "Something went wrong"}
    
    # Set up client mock responses
    mock_client.image_to_video.create.return_value = mock_task
    mock_client.tasks.retrieve.side_effect = [mock_task_pending, mock_task_failed]
    
    # Mock the download_and_convert_to_base64 method
    mocker.patch.object(RunwayImage2Video, 'download_and_convert_to_base64', 
                        return_value="data:image/jpg;base64,abc123")
    
    # Create service with mocked client
    service = RunwayImage2Video()
    service.client = mock_client
    
    # Create parameters with single image
    params = RunwayImage2VideoParameters(
        prompt="test prompt",
        aspect_ratio="1280:768",
        duration=5,
        batch=1,
        file_url=[PromptImage(uri="http://example.com/image.jpg", position="first")]
    )
    
    # Call the function and expect exception
    with pytest.raises(Exception) as exc_info:
        service.generate_image(params)
    
    # Verify the error message
    assert str(exc_info.value) == "Video generation failed"

def test_remote_with_concurrent_execution(mocker):
    # Mock the generate_image method
    mocker.patch.object(RunwayImage2Video, 'generate_image', 
                        return_value=b"video_content")
    
    # Mock the prepare_response function
    mock_prepare_response = mocker.patch(
        'src.FastAPIServer.services.ApiServices.RunWayService.prepare_response',
        return_value={"result": ["video_url"], "Has_NSFW_Content": [False], "time" : {"startup_time" : 1, "runtime" : 1}, "extension" : "webp"}
    )
    
    # Create service
    service = RunwayImage2Video()
    
    # Create parameters
    params = {
        "prompt": "test prompt",
        "aspect_ratio": "1280:768",
        "duration": 5,
        "batch": 2,
        "file_url": [{"uri": "http://example.com/image.jpg", "position": "first"}]
    }
    
    # Call the remote method
    result = service.remote(params)
    
    # Assertions
    assert service.generate_image.call_count == 2
    mock_prepare_response.assert_called_once_with(
        [b"video_content", b"video_content"], 
        [False, False], 
        0, 0, 
        OUTPUT_VIDEO_EXTENSION
    )
    assert "result" in str(result)