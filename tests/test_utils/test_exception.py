import json

from fastapi.responses import JSONResponse

from src.utils.Exceptions import handle_exceptions


def test_handle_exceptions_no_error():
    @handle_exceptions
    def test_func():
        return {"message": "success"}

    result = test_func()

    assert result == {"message": "success"}


def test_handle_exceptions_with_error():
    @handle_exceptions
    def test_func():
        raise Exception("Test exception")

    # Call the function
    response = test_func()

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert json.loads(response.body.decode("utf-8"))["error"] == "Internal Error"
    assert (
        json.loads(response.body.decode("utf-8"))["error_data"] == "('Test exception',)"
    )
