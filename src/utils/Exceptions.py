from fastapi import HTTPException
from fastapi.responses import JSONResponse
import warnings
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from typing import Callable, Any
from functools import wraps
import traceback
from src.data_models.ModalAppSchemas import APITaskResponse
from src.utils.Constants import INTERNAL_ERROR

async def handle_Request_exceptions(request: Request, exc: Exception):
    """
    Asynchronous exception handler for FastAPI request validation errors.
    
    This function catches RequestValidationError exceptions raised during request processing
    and transforms them into a standardized JSON response format. It extracts field-specific
    validation errors and returns them in a structured format that can be easily consumed
    by frontend applications.
    
    Args:
        request (Request): The FastAPI request object that caused the exception
        exc (Exception): The exception that was raised during request processing
        
    Returns:
        JSONResponse: A standardized error response with validation details and 422 status code
        
    Notes:
        This function is designed to be registered as an exception handler with FastAPI's
        exception_handler decorator, allowing consistent error handling across the API.
    """
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
                error = INTERNAL_ERROR

            task_response = APITaskResponse(
                error=str(error),
                error_data=str(error_data),
                status="FAILED"
            )

            warnings.warn(str(traceback_str))
            return JSONResponse(content=task_response.model_dump(), status_code=500)

    return wrapper
