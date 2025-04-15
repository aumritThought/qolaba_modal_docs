import pytest
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import json

from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters, IdeoGramRemixParameters
from src.FastAPIServer.services.ApiServices.IdeogramService import IdeoGramText2Image, IdeogramRemix
from src.utils.Constants import IDEOGRAM_ASPECT_RATIO, OUTPUT_IMAGE_EXTENSION

# Shared test data and fixtures
@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.content = b"test_image_data"
    mock.json.return_value = {"data": [{"url": "https://test-url.com/image.jpg"}]}
    return mock

@pytest.fixture
def mock_image():
    img = MagicMock(spec=Image.Image)
    return img

# Test IdeoGramText2Image service
def test_ideogram_text2image_init(mocker):
    mocker.patch('src.FastAPIServer.services.IService.IService.__init__', return_value=None)
    service = IdeoGramText2Image()
    assert isinstance(service, IdeoGramText2Image)

def test_ideogram_text2image_remote(mocker, mock_response):
    mocker.patch('src.FastAPIServer.services.IService.IService.__init__', return_value=None)
    mocker.patch('src.FastAPIServer.services.ApiServices.IdeogramService.IdeoGramText2Image.make_api_request', return_value=mock_response.content)
    mocker.patch('src.utils.Globals.convert_to_aspect_ratio', return_value="1:1")
    mocker.patch('src.utils.Globals.prepare_response', return_value={"images": ["test_image_data"]})
    
    service = IdeoGramText2Image()
    
    params = {
        "prompt": "test prompt",
        "width": 512,
        "height": 512,
        "batch": 2,
        "magic_prompt_option": "AUTO",
        "negative_prompt": "",
        "style_type": "AUTO"
    }
    
    result = service.remote(params)
    
    assert "result" in result
    assert len(result["result"]) == 2

# Test IdeogramRemix service
def test_ideogram_remix_init(mocker):
    mocker.patch('src.FastAPIServer.services.IService.IService.__init__', return_value=None)
    service = IdeogramRemix()
    assert isinstance(service, IdeogramRemix)


def test_ideogram_remix_remote(mocker, mock_response):
    mocker.patch('src.FastAPIServer.services.IService.IService.__init__', return_value=None)
    mocker.patch('src.FastAPIServer.services.ApiServices.IdeogramService.IdeogramRemix.make_api_request', return_value=mock_response.content)
    mocker.patch('src.utils.Globals.convert_to_aspect_ratio', return_value="1:1")
    mocker.patch('src.utils.Globals.prepare_response', return_value={"images": ["test_image_data"]})
    
    service = IdeogramRemix()
    
    params = {
        "prompt": "test prompt",
        "width": 512,
        "height": 512,
        "batch": 2,
        "file_url": "https://example.com/image.jpg",
        "strength": 0.7,
        "magic_prompt_option": "AUTO",
        "negative_prompt": "",
        "style_type": "AUTO",
        "color_palette": "EMBER"
    }
    
    result = service.remote(params)
    
    assert "result" in result
    assert len(result["result"]) == 2
