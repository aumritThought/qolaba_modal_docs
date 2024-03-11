from fastapi import HTTPException
from fastapi.responses import JSONResponse
import warnings
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from typing import Callable, Any
from functools import wraps
import traceback
from src.data_models.ModalAppSchemas import APITaskResponse


async def handle_Request_exceptions(request: Request, exc: Exception):
    if isinstance(exc, RequestValidationError):
        custom_error = {
            "errors": [
                {
                    "field": error["loc"][0],
                    "message": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors()
            ]
        }
        task_response = APITaskResponse(
            error="Validation error",
            error_data=custom_error,
            status="FAILED"
        )

        return JSONResponse(content=task_response.model_dump(), status_code=422)


def handle_exceptions(func: Callable):
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs)

        except HTTPException as http_exc:
            traceback_str = "".join(
                traceback.format_exception(None, http_exc, http_exc.__traceback__)
            )

            task_response = APITaskResponse(
                error="HTTP Exception", 
                error_data=str(http_exc.detail),
                status="FAILED"
            )

            warnings.warn(traceback_str)
            return JSONResponse(content=task_response.model_dump(), status_code=http_exc.status_code)

        except Exception as exc:
            traceback_str = "".join(
                traceback.format_exception(None, exc, exc.__traceback__)
            )
            try:
                error, error_data = exc.args
            except:
                error_data = str(exc.args)
                error = "Internal Error"

            task_response = APITaskResponse(
                error=str(error),
                error_data=str(error_data),
                status="FAILED"
            )

            warnings.warn(str(traceback_str))
            return JSONResponse(content=task_response.model_dump(), status_code=500)

    return wrapper
