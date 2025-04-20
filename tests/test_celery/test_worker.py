import pytest
import io
from PIL import Image
from unittest.mock import MagicMock
from src.FastAPIServer.celery.Worker import (
    initialize_shared_object,
    upload_image,
    upload_low_res_image,
    on_worker_init,
    get_service_list,
    create_task,
    task_gen,
    get_task_status,
)
from src.data_models.ModalAppSchemas import APIInput


@pytest.fixture
def mock_image_data():
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def test_initialize_shared_object(mocker):
    mock_registry = mocker.patch("src.FastAPIServer.celery.Worker.ServiceRegistry")
    mocker.patch("src.FastAPIServer.celery.Worker.ServiceContainer")

    initialize_shared_object()

    mock_registry.assert_called_once()
    mock_registry.return_value.register_internal_services.assert_called_once()


def test_upload_image(mocker, mock_image_data):
    mock_upload = mocker.patch(
        "src.FastAPIServer.celery.Worker.upload_data_gcp",
        return_value="https://storage.gcp.com/image.png",
    )

    index, url = upload_image(0, mock_image_data, "png", upscale=True)

    assert index == 0
    assert url == "https://storage.gcp.com/image.png"
    mock_upload.assert_called_once_with(mock_image_data, "png", True)


def test_upload_low_res_image(mocker, mock_image_data):
    mock_compress = mocker.patch(
        "src.FastAPIServer.celery.Worker.compress_image",
        return_value=b"compressed_data",
    )
    mock_upload = mocker.patch(
        "src.FastAPIServer.celery.Worker.upload_data_gcp",
        return_value="https://storage.gcp.com/lowres.png",
    )
    mock_image = mocker.patch(
        "src.FastAPIServer.celery.Worker.Image.open", return_value=MagicMock()
    )

    index, url = upload_low_res_image(1, mock_image_data, "png")

    assert index == 1
    assert url == "https://storage.gcp.com/lowres.png"
    assert mock_compress.called
    assert mock_upload.called
    mock_upload.assert_called_once_with(mock_compress.return_value, "png")


def test_on_worker_init(mocker):
    mock_initialize = mocker.patch(
        "src.FastAPIServer.celery.Worker.initialize_shared_object"
    )

    on_worker_init()

    mock_initialize.assert_called_once()


def test_get_service_list(mocker):
    mock_registry = mocker.patch("src.FastAPIServer.celery.Worker.service_registry")
    mock_registry.get_available_services.return_value = ["service1", "service2"]

    result = get_service_list()

    assert "output" in result
    assert result["output"] == ["service1", "service2"]
    mock_registry.get_available_services.assert_called_once()


def test_create_task_api_service(mocker):
    # Mock time
    mock_time = mocker.patch(
        "src.FastAPIServer.celery.Worker.time.time", side_effect=[0, 5.0]
    )

    # Mock registry and service
    mock_registry = mocker.patch("src.FastAPIServer.celery.Worker.service_registry")
    mock_service = MagicMock()
    mock_service().remote.return_value = {
        "result": [b"image_data"],
        "extension": "png",
        "Has_NSFW_Content": [False],
        "low_res_urls": None,
        "Has_copyrighted_Content": [False],
        "time": {"startup_time": 1.0, "runtime": 2.0},
    }
    mock_registry.get_service.return_value = mock_service

    # Set up registry properties
    mock_registry.api_services = ["test_api"]
    mock_registry.modal_services = []

    # Mock upload functions
    mock_upload_image = mocker.patch(
        "src.FastAPIServer.celery.Worker.upload_image",
        return_value=(0, "https://storage.gcp.com/image.png"),
    )
    mock_upload_low_res = mocker.patch(
        "src.FastAPIServer.celery.Worker.upload_low_res_image",
        return_value=(0, "https://storage.gcp.com/lowres.png"),
    )
    mock_copyright = mocker.patch(
        "src.FastAPIServer.celery.Worker.identify_copy_righted_materials",
        return_value=(False, 0),
    )

    # Create test parameters
    parameters = {
        "app_id": "test_api",
        "parameters": {"prompt": "test"},
        "init_parameters": {},
        "inference_type": "a10g",
        "upscale": False,
        "check_copyright_content": True,
        "celery": False,
    }

    # Call function
    result = create_task(parameters)

    # Verify result
    assert result["status"] == "SUCCESS"
    assert "time_required" in result
    assert "output" in result
    assert isinstance(result["output"], dict)
    assert "result" in result["output"]
    assert "Has_NSFW_Content" in result["output"]
    assert "low_res_urls" in result["output"]

    # Verify calls
    mock_registry.get_service.assert_called_once_with("test_api")
    mock_service().remote.assert_called_once()
    mock_upload_image.assert_called_once()


def test_create_task_modal_service(mocker):
    # Mock time
    mock_time = mocker.patch(
        "src.FastAPIServer.celery.Worker.time.time", side_effect=[0, 5.0]
    )

    # Mock registry and service
    mock_registry = mocker.patch("src.FastAPIServer.celery.Worker.service_registry")
    mock_service = MagicMock()
    mock_with_options = MagicMock()
    mock_with_init = MagicMock()

    # Set up return values
    mock_with_options.return_value = mock_with_init
    mock_service.with_options.return_value = mock_with_options
    mock_with_init.run_inference.remote.return_value = {
        "result": ["https://storage.gcp.com/result.png"],
        "extension": None,
        "Has_NSFW_Content": [False],
        "low_res_urls": None,
        "Has_copyrighted_Content": [False],
        "time": {"startup_time": 1.0, "runtime": 3.0},
    }

    # Set up registry
    mock_registry.get_service.return_value = mock_service
    mock_registry.api_services = []
    mock_registry.modal_services = ["test_modal"]

    # Create test parameters
    parameters = {
        "app_id": "test_modal",
        "parameters": {"prompt": "test"},
        "init_parameters": {"model": "test_model"},
        "inference_type": "a10g",
        "upscale": False,
        "check_copyright_content": False,
        "celery": False,
    }

    # Call function
    result = create_task(parameters)

    # Verify result
    assert result["status"] == "SUCCESS"
    assert result["output"]["result"] == ["https://storage.gcp.com/result.png"]
    assert "time_required" in result

    # Verify calls
    mock_registry.get_service.assert_called_once_with("test_modal")
    mock_service.with_options.assert_called_once_with(gpu="a10g")
    mock_with_init.run_inference.remote.assert_called_once_with({"prompt": "test"})


def test_create_task_exception(mocker):
    # Mock registry and container
    mock_registry = mocker.patch("src.FastAPIServer.celery.Worker.service_registry")
    mock_container = mocker.patch("src.FastAPIServer.celery.Worker.container")
    mock_logger = mocker.patch("src.FastAPIServer.celery.Worker.logger")

    # Setup registry mocks: No services available, get_service returns None
    mock_registry.api_services = []
    mock_registry.modal_services = []
    # Let get_service be called, but it won't find the service in the lists later
    mock_registry.get_service.return_value = None

    # Create test parameters
    parameters = {
        "app_id": "nonexistent_app",
        "parameters": {"prompt": "test"},
        "init_parameters": {},
        "inference_type": "a10g",
        "upscale": False,
        "check_copyright_content": False,
        "celery": False,
    }

    # Call the function directly
    result = create_task(parameters)

    # Assert the structure and status of the returned dictionary
    assert isinstance(result, dict)
    assert result.get("status") == "FAILED"
    assert result.get("error") == "Service Not Found"
    assert result.get("error_data") is not None
    assert "Given APP Id 'nonexistent_app' is not available" in result.get(
        "error_data", ""
    )
    assert result.get("input") == parameters
    assert result.get("output") == {}
    assert result.get("time_required") == {}

    # Verify logger was called
    # Ensure the calls match what happens *before* the exception is caught
    mock_logger.error.assert_called_once_with("App ID not found: nonexistent_app")
    mock_logger.exception.assert_called_once()


def test_task_gen_celery(mocker):
    # Mock celery task
    mock_task = MagicMock()
    mock_task.id = "task123"
    mock_task.status = "PENDING"
    mock_delay = mocker.patch(
        "src.FastAPIServer.celery.Worker.create_task.delay", return_value=mock_task
    )

    # Create test parameters
    parameters = APIInput(
        app_id="test_app",
        parameters={"prompt": "test"},
        init_parameters={},
        inference_type="a10g",
        upscale=False,
        check_copyright_content=False,
        celery=True,
    )

    # Call function
    result = task_gen(parameters)

    # Verify result
    assert result["task_id"] == "task123"
    assert result["status"] == "PENDING"
    mock_delay.assert_called_once_with(parameters.model_dump())


def test_task_gen_direct(mocker):
    # Mock create_task
    mock_create = mocker.patch(
        "src.FastAPIServer.celery.Worker.create_task",
        return_value={"status": "SUCCESS"},
    )

    # Create test parameters
    parameters = APIInput(
        app_id="test_app",
        parameters={"prompt": "test"},
        init_parameters={},
        inference_type="a10g",
        upscale=False,
        check_copyright_content=False,
        celery=False,
    )

    # Call function
    result = task_gen(parameters)

    # Verify result
    assert result == {"status": "SUCCESS"}
    mock_create.assert_called_once_with(parameters.model_dump())


def test_get_task_status(mocker):
    # Mock AsyncResult
    mock_async_result = MagicMock()
    mock_async_result.status = "SUCCESS"
    mock_async_result.result = {"output": "test result"}
    mocker.patch(
        "src.FastAPIServer.celery.Worker.celery.AsyncResult",
        return_value=mock_async_result,
    )

    # Call function
    result = get_task_status("task123")

    # Verify result
    assert result.status == "SUCCESS"
    assert result.result == {"output": "test result"}
