import pytest
import base64
import concurrent.futures
from src.FastAPIServer.services.ApiServices.SDXLService import SDXL3Text2Image
from src.data_models.ModalAppSchemas import SDXL3APITextToImageParameters
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, SDXL3_RATIO_LIST

def test_sdxl3_text2image_init(mocker):
    # Mock necessary environment variables instead of constructor
    mocker.patch.dict('os.environ', {"SDXL_API_KEY": "test_key"})
    
    # Create the service instance
    service = SDXL3Text2Image()
    
    # Assert the service was initialized correctly
    assert service.api_key == service.stability_api_key
    assert service.url == service.sdxl3_url

def test_sdxl3_generate_image_success(mocker):
    # Mock environment variables
    mocker.patch.dict('os.environ', {"SDXL_API_KEY": "test_key"})
    mocker.patch('src.utils.Globals.convert_to_aspect_ratio', return_value="1:1")
    
    # Create mock response
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {
        "finish_reason": "SUCCESS",
        "image": base64.b64encode(b"test_image_data").decode('utf-8')
    }
    
    # Mock the make_request function
    mocker.patch('src.FastAPIServer.services.ApiServices.SDXLService.make_request', return_value=mock_response)
    
    # Create service instance
    service = SDXL3Text2Image()
    
    # Create parameters
    parameters = SDXL3APITextToImageParameters(
        prompt="test prompt",
        width=1024,
        height=1024,
        batch=1,
        negative_prompt=""
    )
    
    # Call the method
    result = service.generate_image(parameters, "sd3")
    
    # Assert the results
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == b"test_image_data"  # Image data
    assert result[1] is False  # NSFW flag

def test_sdxl3_generate_image_nsfw(mocker):
    # Mock environment variables
    mocker.patch.dict('os.environ', {"SDXL_API_KEY": "test_key"})
    mocker.patch('src.utils.Globals.convert_to_aspect_ratio', return_value="1:1")
    
    # Create mock response with NSFW content
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {
        "finish_reason": "CONTENT_FILTERED"
    }
    
    # Mock the make_request function
    mocker.patch('src.FastAPIServer.services.ApiServices.SDXLService.make_request', return_value=mock_response)
    
    # Create service instance
    service = SDXL3Text2Image()
    
    # Create parameters
    parameters = SDXL3APITextToImageParameters(
        prompt="test prompt",
        width=1024,
        height=1024,
        batch=1,
        negative_prompt=""
    )
    
    # Call the method
    result = service.generate_image(parameters, "sd3")
    
    # Assert the results
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is None  # No image data
    assert result[1] is True  # NSFW flag

def test_sdxl3_generate_image_invalid_aspect_ratio(mocker):
    # Mock environment variables
    mocker.patch.dict('os.environ', {"SDXL_API_KEY": "test_key"})
    mocker.patch('src.utils.Globals.convert_to_aspect_ratio', return_value="5:7")  # Invalid ratio
    
    # Create service instance
    service = SDXL3Text2Image()
    
    # Create parameters
    parameters = SDXL3APITextToImageParameters(
        prompt="test prompt",
        width=500,
        height=700,
        batch=1,
        negative_prompt=""
    )
    
    # Test invalid aspect ratio
    with pytest.raises(Exception) as e:
        service.generate_image(parameters, "sd3")
    
    # Assert the exception message
    assert "Invalid Height and width dimension" in str(e.value)

def test_sdxl3_remote_success(mocker):
    # Mock environment variables
    mocker.patch.dict('os.environ', {"SDXL_API_KEY": "test_key"})
    
    # Mock the generate_image method
    mock_generate_image = mocker.patch.object(
        SDXL3Text2Image, 'generate_image', 
        return_value=[b"test_image_data", False]
    )
    
    # Mock the prepare_response function
    expected_response = {"result": [b"test_image_data", b"test_image_data"], "Has_NSFW_Content": [False, False], "extension" : "webp", "time" : {"startup_time" : 0, "runtime" : 1}}
    mock_prepare_response = mocker.patch(
        'src.FastAPIServer.services.ApiServices.SDXLService.prepare_response',
        return_value=expected_response
    )
    
    # Create service instance
    service = SDXL3Text2Image()
    
    # Create parameters
    parameters = {
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "batch": 2,
        "negative_prompt": ""
    }
    
    # Call the method
    result = service.remote(parameters)
    
    # Verify generate_image was called correctly
    assert mock_generate_image.call_count == 2
    
    # Verify prepare_response was called correctly
    mock_prepare_response.assert_called_once()
    
    # Verify the result
    assert "result" in result

def test_sdxl3_remote_with_nsfw_content(mocker):
    # Mock environment variables
    mocker.patch.dict('os.environ', {"SDXL_API_KEY": "test_key"})
    
    # Mock generate_image method to return NSFW flags
    mock_generate_image = mocker.patch.object(
        SDXL3Text2Image, 'generate_image', 
        return_value=[None, True]
    )
    
    # Mock the prepare_response function
    expected_response = {"result": [b"test_image_data", b"test_image_data"], "Has_NSFW_Content": [False, False], "extension" : "webp", "time" : {"startup_time" : 0, "runtime" : 1}}
    mock_prepare_response = mocker.patch(
        'src.FastAPIServer.services.ApiServices.SDXLService.prepare_response',
        return_value=expected_response
    )
    
    # Create service instance
    service = SDXL3Text2Image()
    
    # Create parameters
    parameters = {
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "batch": 2,
        "negative_prompt": ""
    }
    
    # Call the method
    result = service.remote(parameters)
    
    # Verify generate_image was called correctly
    assert mock_generate_image.call_count == 2
    
    # Verify prepare_response was called correctly
    mock_prepare_response.assert_called_once()
    
    # Verify the result
    assert "result" in result