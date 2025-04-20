import traceback
import warnings
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.data_models.ModalAppSchemas import APITaskResponse
from src.utils.Constants import INTERNAL_ERROR


def handle_exceptions(func: Callable):
    """
    Decorator for catching and formatting exceptions in route handlers.

    This decorator wraps API endpoint functions to provide consistent error handling
    and response formatting across the application. It catches both HTTPExceptions
    (which may be deliberately raised) and unexpected exceptions, converting them
    into standardized JSONResponse objects with appropriate status codes.

    The decorator maintains full traceback information for debugging purposes by
    logging it as a warning. For general exceptions, it attempts to extract structured
    error information from the exception arguments, falling back to a generic internal
    error message if the expected format is not present.

    Args:
        func (Callable): The function to wrap with exception handling

    Returns:
        Callable: A wrapped version of the input function with exception handling
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs)

        except HTTPException as http_exc:
            traceback_str = "".join(
                traceback.format_exception(None, http_exc, http_exc.__traceback__)
            )

            task_response = APITaskResponse(
                error="HTTP Exception", error_data=str(http_exc.detail), status="FAILED"
            )

            warnings.warn(traceback_str)
            return JSONResponse(
                content=task_response.model_dump(), status_code=http_exc.status_code
            )

        except Exception as exc:
            traceback_str = "".join(
                traceback.format_exception(None, exc, exc.__traceback__)
            )
            try:
                error, error_data = exc.args
            except:
                error_data = str(exc.args)
                error = INTERNAL_ERROR

            task_response = APITaskResponse(
                error=str(error), error_data=str(error_data), status="FAILED"
            )

            warnings.warn(str(traceback_str))
            return JSONResponse(content=task_response.model_dump(), status_code=500)

    return wrapper
