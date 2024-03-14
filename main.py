from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.data_models.ModalAppSchemas import APIInput, APITaskResponse, TaskStatus
from src.FastAPIServer.celery.Worker import task_gen, get_task_status
from src.utils.Globals import check_token
from src.utils.Constants import app_dict
from src.utils.Exceptions import handle_Request_exceptions, handle_exceptions
import uvicorn, os
from fastapi.exceptions import RequestValidationError

app = FastAPI()
app.exception_handler(RequestValidationError)(handle_Request_exceptions)
auth_scheme = HTTPBearer()


@app.post("/generate_content", response_model=APITaskResponse)
@handle_exceptions
def generate_content(
    parameters: dict,
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    api_parameters = APIInput(**parameters)
    api_parameters.parameters = parameters
    api_parameters.init_parameters = app_dict[api_parameters.app_id]["init_parameters"]
    api_parameters.app_id = app_dict[api_parameters.app_id]["app_id"]
    check_token(api_key)
    return task_gen(api_parameters)

@app.get("/list_of_Apps", response_model=APITaskResponse)
@handle_exceptions
def get_app_list(
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    check_token(api_key)
    return app_dict


@app.post("/tasks", response_model=APITaskResponse)
def get_status(parameters : TaskStatus,
               api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    check_token(api_key)
    task_result = get_task_status(parameters.task_id)

    if task_result.status=="FAILURE":
        raise task_result.result
    
    task_Response = APITaskResponse(
            task_id=parameters.task_id,
            input=parameters.model_dump(),
            status=task_result.status,
            output= task_result.result
            ).model_dump()
    
    return task_Response


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        timeout_keep_alive=9000,
        workers=int(os.getenv("NUM_WORKERS")),
    )