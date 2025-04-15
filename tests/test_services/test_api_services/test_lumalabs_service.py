import pytest
from src.data_models.ModalAppSchemas import LumaLabsVideoParameters, PromptImage
from src.FastAPIServer.services.ApiServices.LumaLabsService import LumaVideo
from unittest.mock import MagicMock, patch

def test_generate_image_text_only_success(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    mock_generation = MagicMock()
    mock_generation.id = "test_id"
    mock_generation.state = "completed"
    mock_generation.assets.video = "video_url"
    
    mock_client.generations.create.return_value = mock_generation
    mock_client.generations.get.return_value = mock_generation
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    mock_client.generations.create.assert_called_once_with(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True
    )


def test_generate_image_single_image_start(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    mock_generation = MagicMock()
    mock_generation.id = "test_id"
    mock_generation.state = "completed"
    mock_generation.assets.video = "video_url"
    
    mock_client.generations.create.return_value = mock_generation
    mock_client.generations.get.return_value = mock_generation
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters with single image at start
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1,
        file_url=[PromptImage(uri="http://example.com/image.jpg", position="first")]
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    mock_client.generations.create.assert_called_once_with(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        keyframes={
            "frame0": {
                "type": "image",
                "url": "http://example.com/image.jpg"
            }
        }
    )


def test_generate_image_single_image_end(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    mock_generation = MagicMock()
    mock_generation.id = "test_id"
    mock_generation.state = "completed"
    mock_generation.assets.video = "video_url"
    
    mock_client.generations.create.return_value = mock_generation
    mock_client.generations.get.return_value = mock_generation
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters with single image at end
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1,
        file_url=[PromptImage(uri="http://example.com/image.jpg", position="last")]
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    mock_client.generations.create.assert_called_once_with(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        keyframes={
            "frame1": {
                "type": "image",
                "url": "http://example.com/image.jpg"
            }
        }
    )


def test_generate_image_two_images(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    mock_generation = MagicMock()
    mock_generation.id = "test_id"
    mock_generation.state = "completed"
    mock_generation.assets.video = "video_url"
    
    mock_client.generations.create.return_value = mock_generation
    mock_client.generations.get.return_value = mock_generation
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters with two images
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1,
        file_url=[
            PromptImage(uri="http://example.com/start.jpg", position="first"),
            PromptImage(uri="http://example.com/end.jpg", position="last")
        ]
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    mock_client.generations.create.assert_called_once_with(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        keyframes={
            "frame0": {
                "type": "image",
                "url": "http://example.com/start.jpg"
            },
            "frame1": {
                "type": "image",
                "url": "http://example.com/end.jpg"
            }
        }
    )


def test_generate_image_two_images_reversed_order(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    mock_generation = MagicMock()
    mock_generation.id = "test_id"
    mock_generation.state = "completed"
    mock_generation.assets.video = "video_url"
    
    mock_client.generations.create.return_value = mock_generation
    mock_client.generations.get.return_value = mock_generation
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters with two images in reversed order
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1,
        file_url=[
            PromptImage(uri="http://example.com/end.jpg", position="last"),
            PromptImage(uri="http://example.com/start.jpg", position="first")
        ]
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    mock_client.generations.create.assert_called_once_with(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        keyframes={
            "frame0": {
                "type": "image",
                "url": "http://example.com/start.jpg"
            },
            "frame1": {
                "type": "image",
                "url": "http://example.com/end.jpg"
            }
        }
    )


def test_generate_image_polling_success(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    
    # Create generations with different states
    pending_generation = MagicMock()
    pending_generation.id = "test_id"
    pending_generation.state = "pending"
    
    completed_generation = MagicMock()
    completed_generation.id = "test_id"
    completed_generation.state = "completed"
    completed_generation.assets.video = "video_url"
    
    # Setup the mock to return different values on successive calls
    mock_client.generations.create.return_value = pending_generation
    mock_client.generations.get.side_effect = [pending_generation, completed_generation]
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    assert mock_client.generations.get.call_count == 2


def test_remote_with_concurrent_execution(mocker):
    # Mock dependencies
    mock_generate_image = mocker.patch.object(
        LumaVideo, 
        'generate_image', 
        side_effect=[b"video1", b"video2", b"video3"]
    )
    
    # Mock the prepare_response function
    mock_videos = ["video_url1", "video_url2", "video_url3"]
    mock_nsfw = [False, False, False]
    mock_prepare_response = mocker.patch(
        'src.FastAPIServer.services.ApiServices.LumaLabsService.prepare_response',
        return_value={"result": mock_videos, "Has_NSFW_Content": mock_nsfw, "extension" : "webp", "time" : {"startup_time" : "1.2", "runtime" : "2.5"}}
    )
    
    # Mock the timing_decorator
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.timing_decorator', lambda func: func)
    
    # Create service
    service = LumaVideo()
    
    # Create parameters
    params = {
        "prompt": "test prompt",
        "aspect_ratio": "16:9",
        "loop": True,
        "batch": 3
    }
    
    # Call the function
    result = service.remote(params)
    
    # Assert
    assert mock_generate_image.call_count == 3
    mock_prepare_response.assert_called_once_with(
        [b"video1", b"video2", b"video3"], 
        [False, False, False], 
        0, 0, 
        'mp4'
    )
    assert "result" in str({"result": mock_videos, "Has_NSFW_Content": mock_nsfw, "extension" : "webp", "time" : {"startup_time" : "1.2", "runtime" : "2.5"}})


def test_generate_image_with_none_file_url(mocker):
    # Mock dependencies
    mock_client = MagicMock()
    mock_generation = MagicMock()
    mock_generation.id = "test_id"
    mock_generation.state = "completed"
    mock_generation.assets.video = "video_url"
    
    mock_client.generations.create.return_value = mock_generation
    mock_client.generations.get.return_value = mock_generation
    
    # Mock the make_request function
    mock_response = MagicMock()
    mock_response.content = b"video_content"
    mocker.patch('src.FastAPIServer.services.ApiServices.LumaLabsService.make_request', return_value=mock_response)
    
    # Create service with mocked client
    service = LumaVideo()
    service.client = mock_client
    
    # Create parameters with None file_url
    params = LumaLabsVideoParameters(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True,
        batch=1,
        file_url=None
    )
    
    # Call the function
    result = service.generate_image(params)
    
    # Assert
    assert result == b"video_content"
    mock_client.generations.create.assert_called_once_with(
        prompt="test prompt",
        aspect_ratio="16:9",
        loop=True
    )