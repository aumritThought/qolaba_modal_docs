from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.data_models.ModalAppSchemas import APIInput, APITaskResponse
from src.FastAPIServer.celery.Worker import task_gen, get_service_list
from src.utils.Globals import check_token
from src.utils.Exceptions import handle_Request_exceptions, handle_exceptions
import uvicorn, os
from fastapi.exceptions import RequestValidationError

app = FastAPI()
app.exception_handler(RequestValidationError)(handle_Request_exceptions)
auth_scheme = HTTPBearer()


@app.post("/generate_content", response_model=APITaskResponse)
@handle_exceptions
def generate_content(
    parameters: APIInput,
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    check_token(api_key)
    return task_gen(parameters)

@app.get("/list_of_Apps", response_model=APITaskResponse)
@handle_exceptions
def get_app_list(
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    check_token(api_key)
    return get_service_list()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        timeout_keep_alive=9000,
        workers=int(os.getenv("NUM_WORKERS")),
    )