from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.Globals import (
    convert_image_to_bytes,
    get_image_from_url,
    get_seed_generator,
    invert_bw_image_color,
    make_request,
    prepare_response,
    simple_boundary_blur,
)


def test_convert_image_to_bytes():
    # Test RGB image (JPEG)
    rgb_image = Image.new("RGB", (100, 100), color="red")
    rgb_bytes = convert_image_to_bytes(rgb_image, quality=95)
    assert isinstance(rgb_bytes, bytes)
    assert len(rgb_bytes) > 0

    # Test RGBA image (PNG)
    rgba_image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    rgba_bytes = convert_image_to_bytes(rgba_image)
    assert isinstance(rgba_bytes, bytes)
    assert len(rgba_bytes) > 0


def test_prepare_response():
    result = ["test_image_data"]
    nsfw_flags = [False]
    time_data = 1.5
    runtime = 2.0
    extension = "jpg"

    response = prepare_response(result, nsfw_flags, time_data, runtime, extension)

    assert response["result"] == result
    assert response["Has_NSFW_Content"] == nsfw_flags
    assert response["time"]["startup_time"] == time_data
    assert response["time"]["runtime"] == runtime
    assert response["extension"] == extension

    # Test with dict result
    dict_result = {"key": "value"}
    dict_response = prepare_response(dict_result, nsfw_flags, time_data, runtime)
    assert dict_response["result"] == dict_result
    assert dict_response["extension"] is None


def test_make_request(mocker):
    # Mock successful GET request
    mock_get = mocker.patch("requests.get")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    mock_get.return_value = mock_response

    response = make_request(
        "https://example.com", "GET", headers={"Accept": "application/json"}
    )
    mock_get.assert_called_once_with(
        "https://example.com", headers={"Accept": "application/json"}
    )
    assert response.status_code == 200

    # Mock successful POST request
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = mock_response

    json_data = {"key": "value"}
    response = make_request("https://example.com", "POST", json_data=json_data)
    mock_post.assert_called_once_with(
        "https://example.com", data=json_data, headers=None, files=None, json=None
    )

    # Test error handling
    mock_post.reset_mock()
    error_response = MagicMock()
    error_response.status_code = 400
    error_response.text = "Error occurred"
    mock_post.return_value = error_response

    with pytest.raises(Exception) as excinfo:
        make_request("https://example.com", "POST")
    assert "Error occurred" in str(excinfo.value)

    # Test content moderation error
    error_response.text = "content moderation failed"
    with pytest.raises(Exception) as excinfo:
        make_request("https://example.com", "POST")
    assert "NSFW" in str(excinfo.value)


@pytest.mark.parametrize("input_method", ["invalid_method", "GET", "POST"])
def test_make_request_methods(mocker, input_method):
    if input_method == "invalid_method":
        with pytest.raises(Exception) as excinfo:
            make_request("https://example.com", input_method)
        assert "Invalid request method" in str(excinfo.value)
    else:
        mock_method = mocker.patch(f"requests.{input_method.lower()}")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_method.return_value = mock_response

        response = make_request("https://example.com", input_method)
        assert response.status_code == 200


def test_get_image_from_url(mocker):
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.content = b"image_data"
    mocker.patch("src.utils.Globals.make_request", return_value=mock_response)

    # Mock PIL Image
    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image
    mocker.patch("PIL.Image.open", return_value=mock_image)
    mocker.patch("src.utils.Globals.resize_image", return_value=mock_image)

    # Test successful image fetch
    result = get_image_from_url("https://example.com/image.jpg")
    assert result == mock_image

    # Test with resizing disabled
    result = get_image_from_url("https://example.com/image.jpg", rs_image=False)
    assert result == mock_image

    # Test error handling
    mocker.patch(
        "src.utils.Globals.make_request", side_effect=Exception("Network error")
    )

    with pytest.raises(Exception) as excinfo:
        get_image_from_url("https://example.com/image.jpg")
    assert "Image URL Error" in str(excinfo.value)


def test_invert_bw_image_color(mocker):
    # Create test image and mock numpy operations
    test_image = Image.new("L", (10, 10), color=0)  # Black image
    test_array = np.zeros((10, 10), dtype=np.uint8)
    inverted_array = 255 * np.ones((10, 10), dtype=np.uint8)

    mocker.patch("numpy.array", return_value=test_array)
    mock_from_array = mocker.patch(
        "PIL.Image.fromarray", return_value=Image.new("L", (10, 10), color=255)
    )

    # Test inversion
    result = invert_bw_image_color(test_image)

    # The result should be a white image (inverted from black)
    np.array.assert_called_once_with(test_image)
    mock_from_array.assert_called_once()


def test_simple_boundary_blur():
    test_image = Image.new("RGB", (100, 100), color="red")
    result = simple_boundary_blur(test_image, blur_radius=5)

    # The result should be a PIL Image
    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)


def test_get_seed_generator():
    seed = 42
    generator = get_seed_generator(seed)

    # Test that the generator is properly seeded
    assert isinstance(generator, torch.Generator)

    # Generate two random numbers with the same seed
    val1 = torch.randn(1, generator=generator).item()

    # Reset generator with same seed
    generator = get_seed_generator(seed)
    val2 = torch.randn(1, generator=generator).item()

    # They should be identical
    assert val1 == val2
