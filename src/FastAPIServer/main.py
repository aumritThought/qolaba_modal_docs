from fastapi import FastAPI, UploadFile
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.utils.Globals import *
from data_models.Schemas import *
# from celery.Worker import task_gen, get_task_status
from src.utils.Exceptions import handle_Request_exceptions, handle_exceptions
import uvicorn
from fastapi.exceptions import RequestValidationError

# from src.routes import api
# from src.routes import health, ServerlessDeployments
# from src.core.container import Container
from dotenv import load_dotenv

app = FastAPI()
app.exception_handler(RequestValidationError)(handle_Request_exceptions)
auth_scheme = HTTPBearer()

load_dotenv()


@app.post("/generate_content", response_model=TaskResponse)
@handle_exceptions
def get_text_to_image(
    parameters: Text2ImageParameters,
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    check_token(api_key)
    if(parameters.category == TEXT_TO_IMAGE_IDENTIFIER):
        return task_gen(parameters.text2image_parameters, TEXT_TO_IMAGE_IDENTIFIER)
