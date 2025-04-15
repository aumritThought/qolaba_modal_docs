import pytest
import base64
from unittest.mock import MagicMock, patch, Mock
import concurrent.futures
from PIL import Image
import io
import fal_client
import requests
from src.data_models.ModalAppSchemas import (
    FluxText2ImageParameters, UpscaleParameters, IdeoGramText2ImageParameters,
    OmnigenParameters, FluxImage2ImageParameters, RecraftV3Text2ImageParameters,
    SDXLText2ImageParameters
)
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG
from src.FastAPIServer.services.ApiServices.FalAIService import (
    FalAIFluxProText2Image, FalAIFluxDevText2Image, FalAIFluxschnellText2Image,
    FalAIFluxDevImage2Image, FalAIRefactorV3Text2Image, FalAISD35LargeText2Image,
    FalAISD35LargeTurboText2Image, FalAISD35MediumText2Image, FalAIFlux3Inpainting,
    FalAIFlux3ReplaceBackground, FalAIFluxProRedux, FalAIFluxProCanny,
    FalAIFluxProDepth, OmnigenV1, FalAIFluxPulID
)

# Common fixtures

@pytest.fixture
def mock_image():
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.fixture
def mock_pil_image():
    return Image.new('RGB', (100, 100), color='red')

@pytest.fixture
def mock_response(mock_image):
    mock_resp = MagicMock()
    mock_resp.content = mock_image
    mock_resp.status_code = 200
    mock_resp.text = ""
    return mock_resp

@pytest.fixture
def mock_fal_response():
    return {
        "images": [{"url": "data:image/jpeg;base64,SGVsbG8gV29ybGQ="}],
        "has_nsfw_concepts": [0]
    }

@pytest.fixture
def mock_fal_response_with_url():
    return {
        "images": [{"url": "https://example.com/image.jpg"}],
        "has_nsfw_concepts": [0]
    }

@pytest.fixture
def mock_nsfw_response():
    return {
        "images": [{"url": "data:image/jpeg;base64,SGVsbG8gV29ybGQ="}],
        "has_nsfw_concepts": [1]
    }

# Test cases for common patterns

def test_flux_pro_text2image_make_api_request(mocker, mock_fal_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response)
    service = FalAIFluxProText2Image()
    parameters = FluxText2ImageParameters(
        prompt="test prompt", 
        width=512, 
        height=512, 
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == base64.b64decode("SGVsbG8gV29ybGQ=")
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux-pro/v1.1",
        arguments={
            "prompt": "test prompt",
            "image_size": {"width": 512, "height": 512},
            "safety_tolerance": "2",
            "sync_mode": True,
            "enable_safety_checker": True,
        },
        with_logs=False,
    )

def test_flux_pro_text2image_nsfw_detection(mocker, mock_nsfw_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_nsfw_response)
    service = FalAIFluxProText2Image()
    parameters = FluxText2ImageParameters(
        prompt="test prompt", 
        width=512, 
        height=512, 
        batch=1
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        service.make_api_request(parameters)
    
    assert exc_info.value.args[0] == IMAGE_GENERATION_ERROR
    assert exc_info.value.args[1] == NSFW_CONTENT_DETECT_ERROR_MSG

# def test_flux_pro_text2image_remote(mocker, mock_fal_response):
#     # Arrange
#     mock_make_api_request = mocker.patch.object(
#         FalAIFluxProText2Image, 'make_api_request', 
#         return_value=base64.b64decode("SGVsbG8gV29ybGQ=")
#     )
#     mocker.patch('src.utils.Globals.prepare_response', return_value={"success": True})
    
#     service = FalAIFluxProText2Image()
#     parameters = {
#         "prompt": "test prompt", 
#         "width": 512, 
#         "height": 512, 
#         "batch": 2
#     }
    
#     # Act
#     result = service.remote(parameters)
    
#     # Assert
#     assert result == {"success": True}
#     assert mock_make_api_request.call_count == 2

def test_flux_dev_text2image_make_api_request(mocker, mock_fal_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response)
    service = FalAIFluxDevText2Image()
    parameters = FluxText2ImageParameters(
        prompt="test prompt", 
        width=512, 
        height=512, 
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == base64.b64decode("SGVsbG8gV29ybGQ=")
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux/dev",
        arguments={
            "prompt": "test prompt",
            "image_size": {"width": 512, "height": 512},
            "safety_tolerance": "2",
            "sync_mode": True,
            "enable_safety_checker": True,
        },
        with_logs=False,
    )

# def test_flux_schnell_text2image_make_api_request(mocker, mock_fal_response_with_url, mock_response):
#     # Arrange
#     mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
#     mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
#     service = FalAIFluxschnellText2Image()
#     parameters = FluxText2ImageParameters(
#         prompt="test prompt", 
#         width=512, 
#         height=512, 
#         batch=1
#     )
    
#     # Act
#     result = service.make_api_request(parameters)
    
#     # Assert
#     assert result == mock_response.content
#     fal_client.subscribe.assert_called_once_with(
#         "fal-ai/flux/schnell",
#         arguments={
#             "prompt": "test prompt",
#             "image_size": {"width": 512, "height": 512},
#             "num_inference_steps": 12,
#             "num_images": 1,
#             "enable_safety_checker": True,
#         },
#         with_logs=False,
#     )

def test_flux_dev_image2image_make_api_request(mocker, mock_fal_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response)
    service = FalAIFluxDevImage2Image()
    parameters = FluxImage2ImageParameters(
        prompt="test prompt", 
        file_url="https://example.com/image.jpg",
        strength=0.7,
        width=512, 
        height=512, 
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == base64.b64decode("SGVsbG8gV29ybGQ=")
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux/dev/image-to-image",
        arguments={
            "image_url": "https://example.com/image.jpg",
            "prompt": "test prompt",
            "safety_tolerance": "2",
            "sync_mode": True,
            "enable_safety_checker": True,
            "strength": 0.7,
        },
        with_logs=False,
    )

# def test_recraft_v3_text2image_make_api_request(mocker, mock_fal_response_with_url, mock_response):
#     # Arrange
#     mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
#     mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
#     mocker.patch('requests.get', return_value=mock_response)
#     service = FalAIRefactorV3Text2Image()
#     parameters = RecraftV3Text2ImageParameters(
#         prompt="test prompt", 
#         width=512, 
#         height=512,
#         style="cinematic",
#         batch=1
#     )
    
#     # Act
#     result = service.make_api_request(parameters)
    
#     # Assert
#     assert result == mock_response.content
#     fal_client.subscribe.assert_called_once_with(
#         "fal-ai/recraft-v3",
#         arguments={
#             "prompt": "test prompt",
#             "image_size": {"width": 512, "height": 512},
#             "style": "cinematic",
#         },
#         with_logs=False,
#     )

def test_sdxl_large_text2image_make_api_request(mocker, mock_fal_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response)
    service = FalAISD35LargeText2Image()
    parameters = SDXLText2ImageParameters(
        prompt="test prompt",
        negative_prompt="bad quality",
        width=512, 
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == base64.b64decode("SGVsbG8gV29ybGQ=")
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/stable-diffusion-v35-large",
        arguments={
            "prompt": "test prompt",
            "negative_prompt": "bad quality",
            "image_size": {"width": 512, "height": 512},
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True,
        },
        with_logs=False,
    )

def test_sdxl_large_turbo_text2image_make_api_request(mocker, mock_fal_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response)
    service = FalAISD35LargeTurboText2Image()
    parameters = SDXLText2ImageParameters(
        prompt="test prompt",
        negative_prompt="bad quality",
        width=512, 
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == base64.b64decode("SGVsbG8gV29ybGQ=")
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/stable-diffusion-v35-large/turbo",
        arguments={
            "prompt": "test prompt",
            "negative_prompt": "bad quality",
            "image_size": {"width": 512, "height": 512},
            "num_inference_steps": 10,
            "guidance_scale": 3,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True,
        },
        with_logs=False,
    )

def test_flux3_inpainting_generate_image(mocker, mock_fal_response_with_url, mock_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
    mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
    mocker.patch('requests.get', return_value=mock_response)
    service = FalAIFlux3Inpainting()
    parameters = IdeoGramText2ImageParameters(
        prompt="test prompt",
        file_url="https://example.com/image.jpg",
        mask_url="https://example.com/mask.jpg",
        batch=1
    )
    
    # Act
    result = service.generate_image(parameters)
    
    # Assert
    assert result == mock_response.content
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux-pro/v1/fill",
        arguments={
            "prompt": "test prompt",
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_url": "https://example.com/image.jpg",
            "mask_url": "https://example.com/mask.jpg",
        },
        with_logs=False,
    )

# def test_flux3_inpainting_remote(mocker, mock_pil_image, mock_response):
#     # Arrange
#     mock_response_obj = Mock()
#     mock_response_obj.content = mock_response.content
#     mock_response_obj.status_code = 200
    
#     mocker.patch('src.utils.Globals.get_image_from_url', return_value=mock_pil_image)
#     mocker.patch('src.utils.Globals.make_request', return_value=mock_response_obj)
#     mocker.patch('src.utils.Globals.invert_bw_image_color', return_value=mock_pil_image)
#     mocker.patch('src.utils.Globals.simple_boundary_blur', return_value=mock_pil_image)
#     mocker.patch('src.utils.Globals.upload_data_gcp', return_value=mock_pil_image)
#     mocker.patch.object(FalAIFlux3Inpainting, 'generate_image', return_value=mock_response.content)
#     mocker.patch('src.utils.Globals.prepare_response', return_value={"success": True})
    
#     service = FalAIFlux3Inpainting()
#     parameters = {
#         "prompt": "test prompt",
#         "file_url": "https://example.com/image.jpg",
#         "mask_url": "https://example.com/mask.jpg",
#         "batch": 1
#     }
    
#     # Act
#     result = service.remote(parameters)
    
#     # Assert
#     assert result == {"success": True}
#     assert service.generate_image.call_count == 1

# def test_flux3_replace_background_remote(mocker, mock_pil_image, mock_response):
#     # Arrange
#     mock_remover = MagicMock()
#     mock_remover.process.return_value = mock_pil_image
#     mock_pil_image.getchannel = MagicMock(return_value=mock_pil_image)
    
#     mock_response_obj = Mock()
#     mock_response_obj.content = mock_response.content
#     mock_response_obj.status_code = 200
    
#     mocker.patch('src.utils.Globals.get_image_from_url', return_value=mock_pil_image)
#     mocker.patch('src.utils.Globals.make_request', return_value=mock_response_obj)
#     mocker.patch('src.utils.Globals.invert_bw_image_color', return_value=mock_pil_image)
#     mocker.patch('src.utils.Globals.simple_boundary_blur', return_value=mock_pil_image)
#     mocker.patch('src.utils.Globals.upload_data_gcp', return_value="https://mock-gcp-url.com/image.jpg")
#     mocker.patch.object(FalAIFlux3ReplaceBackground, 'generate_image', return_value=mock_response.content)
#     mocker.patch('src.utils.Globals.prepare_response', return_value={"success": True})
    
#     service = FalAIFlux3ReplaceBackground(remover=mock_remover)
#     parameters = {
#         "prompt": "test prompt",
#         "file_url": "https://example.com/image.jpg",
#         "batch": 1
#     }
    
#     # Act
#     result = service.remote(parameters)
    
#     # Assert
#     assert result == {"success": True}
#     mock_remover.process.assert_called_once_with(mock_pil_image, type="rgba")
#     assert service.generate_image.call_count == 1

def test_flux_pro_redux_make_api_request(mocker, mock_fal_response_with_url, mock_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
    mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
    mocker.patch('requests.get', return_value=mock_response)
    service = FalAIFluxProRedux()
    parameters = FluxImage2ImageParameters(
        prompt="test prompt",
        file_url="https://example.com/image.jpg",
        strength=0.7,
        num_inference_steps=20,
        width=512, 
        height=512,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == mock_response.content
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux-pro/v1.1/redux",
        arguments={
            "image_url": "https://example.com/image.jpg",
            "num_inference_steps": 20,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_prompt_strength": 0.7,
            "prompt": "test prompt",
            "image_size": {"width": 512, "height": 512},
        },
        with_logs=False,
    )

def test_flux_pro_canny_make_api_request(mocker, mock_fal_response_with_url, mock_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
    mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
    mocker.patch('requests.get', return_value=mock_response)
    service = FalAIFluxProCanny()
    parameters = FluxImage2ImageParameters(
        prompt="test prompt",
        file_url="https://example.com/image.jpg",
        num_inference_steps=20,
        width=512, 
        height=512,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == mock_response.content
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux-pro/v1/canny",
        arguments={
            "control_image_url": "https://example.com/image.jpg",
            "num_inference_steps": 20,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": "test prompt",
        },
        with_logs=False,
    )

def test_flux_pro_depth_make_api_request(mocker, mock_fal_response_with_url, mock_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
    mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
    mocker.patch('requests.get', return_value=mock_response)
    service = FalAIFluxProDepth()
    parameters = FluxImage2ImageParameters(
        prompt="test prompt",
        file_url="https://example.com/image.jpg",
        num_inference_steps=20,
        width=512, 
        height=512,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == mock_response.content
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux-pro/v1/depth",
        arguments={
            "control_image_url": "https://example.com/image.jpg",
            "num_inference_steps": 20,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": "test prompt",
        },
        with_logs=False,
    )

def test_omnigen_v1_make_api_request(mocker, mock_fal_response_with_url, mock_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
    mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
    mocker.patch('requests.get', return_value=mock_response)
    service = OmnigenV1()
    parameters = OmnigenParameters(
        prompt="test prompt",
        file_url="https://example.com/image.jpg",
        num_inference_steps=20,
        width=512, 
        height=512,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == mock_response.content
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/omnigen-v1",
        arguments={
            "prompt": "test prompt",
            "image_size": {"width": 512, "height": 512},
            "num_inference_steps": 20,
            "guidance_scale": 2.5,
            "img_guidance_scale": 1.6,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "input_image_urls": ["https://example.com/image.jpg"]
        },
        with_logs=False,
    )

def test_flux_pulid_make_api_request(mocker, mock_fal_response_with_url, mock_response):
    # Arrange
    mocker.patch('fal_client.subscribe', return_value=mock_fal_response_with_url)
    mocker.patch('src.utils.Globals.make_request', return_value=mock_response)
    mocker.patch('requests.get', return_value=mock_response)
    service = FalAIFluxPulID()
    parameters = FluxImage2ImageParameters(
        prompt="test prompt",
        negative_prompt="bad quality",
        file_url="https://example.com/image.jpg",
        width=512, 
        height=512,
        batch=1
    )
    
    # Act
    result = service.make_api_request(parameters)
    
    # Assert
    assert result == mock_response.content
    fal_client.subscribe.assert_called_once_with(
        "fal-ai/flux-pulid",
        arguments={
            "reference_image_url": "https://example.com/image.jpg",
            "image_size": {"width": 512, "height": 512},
            "num_inference_steps": 20,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": "test prompt",
            "negative_prompt": "bad quality",
            "enable_safety_checker": True,
            "id_weight": 1,
        },
        with_logs=False,
    )
