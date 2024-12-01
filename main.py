from dotenv import load_dotenv
load_dotenv()
from pillow_heif import register_heif_opener
register_heif_opener()

from fastapi import FastAPI, UploadFile, Body
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.data_models.ModalAppSchemas import APIInput, APITaskResponse, TaskStatus, OpenAITTSParameters
from src.FastAPIServer.celery.Worker import task_gen, get_task_status, initialize_shared_object
from src.utils.Globals import check_token, upload_to_gcp
from src.utils.Constants import app_dict, INTERNAL_ERROR, OUTPUT_IMAGE_EXTENSION
from src.utils.Exceptions import handle_Request_exceptions, handle_exceptions
import uvicorn, os, io
from fastapi.exceptions import RequestValidationError
from transparent_background import Remover
from src.FastAPIServer.services.ApiServices.OpenAIService import OpenAITexttoSpeech
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI()
app.exception_handler(RequestValidationError)(handle_Request_exceptions)
auth_scheme = HTTPBearer()

@app.on_event("startup")
def startup_event():
    Remover(device="cpu") # Do not remove this line. it will download the weights which are not directly possible through celery due to ray and celery error
    initialize_shared_object()

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
    task_response = APITaskResponse()
    task_response.output = app_dict
    return task_response.model_dump()


@app.post("/tasks", response_model=APITaskResponse)
@handle_exceptions
def get_status(parameters : TaskStatus,
               api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    check_token(api_key)
    task_result = get_task_status(parameters.task_id)

    if task_result.status == "FAILURE":
        
        try:
            error = task_result.info.args[0]
            error_details = task_result.info.args[1]
        except Exception as er:
            error = INTERNAL_ERROR
            error_details = str(task_result.info)
        raise Exception(error, error_details)
    
    if(task_result.status == "PENDING"):
        task_Response = APITaskResponse(
                task_id=parameters.task_id,
                input=parameters.model_dump(),
                status=task_result.status,
        ).model_dump()
        
        return task_Response
    else:
        return task_result.result


@app.post("/upload_data_GCP", response_model=APITaskResponse)
@handle_exceptions
def upload_file(file: UploadFile, file_type : str = Body(..., embed=True), api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    check_token(api_key)
    if(file_type == OUTPUT_IMAGE_EXTENSION):
        image = Image.open(io.BytesIO(file.file.read()))
        url = upload_to_gcp(image, file_type)
    else:
        url = upload_to_gcp(file.file.read(), file_type)
    task_Response = APITaskResponse(
                output= {"url": url},
                status="SUCCESS"
    ).model_dump()
    return task_Response

@app.post("/generate_stream_audio", response_model=APITaskResponse)
@handle_exceptions
def upload_file(parameters: OpenAITTSParameters, api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    check_token(api_key)
    tts = OpenAITexttoSpeech()
    return StreamingResponse(tts.remote(parameters), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        timeout_keep_alive=9000,
        workers=int(os.getenv("NUM_WORKERS")),
    )