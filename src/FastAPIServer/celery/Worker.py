from celery import Celery
import time, io
from celery.signals import worker_init, worker_process_init
from src.utils.Constants import REDIS_URL, CELERY_RESULT_EXPIRATION_TIME, CELERY_MAX_RETRY, CELERY_SOFT_LIMIT, OUTPUT_IMAGE_EXTENSION
from src.data_models.ModalAppSchemas import APIInput, APITaskResponse, TaskResponse
from celery.result import AsyncResult
from src.FastAPIServer.services.ServiceContainer import ServiceContainer, ServiceRegistry
from src.utils.Globals import upload_data_gcp, compress_image
from PIL import Image
import concurrent.futures
from PIL import Image
import io

celery = Celery(
    "task_scheduler",
    broker = REDIS_URL,
    backend = REDIS_URL,
)

celery.conf.update(
    result_expires = CELERY_RESULT_EXPIRATION_TIME
)

celery.conf.CELERY_TASK_SOFT_TIME_LIMIT = CELERY_RESULT_EXPIRATION_TIME

celery.conf.CELERY_TASK_TIME_LIMIT = CELERY_RESULT_EXPIRATION_TIME

celery.conf.worker_init = worker_init


def initialize_shared_object():
    global service_registry
    global container 

    container = ServiceContainer()
    service_registry = ServiceRegistry(container)
    service_registry.register_internal_services() 
    
    pass

def upload_image(index, image_data, extension, upscale=False):
    url = upload_data_gcp(image_data, extension, upscale)
    return index, url

def upload_low_res_image(index, image_data, extension):
    low_rs_image = Image.open(io.BytesIO(image_data))
    low_rs_image = compress_image(low_rs_image)
    url = upload_data_gcp(low_rs_image, extension)
    return index, url

@worker_process_init.connect
def on_worker_init(**kwargs):
    initialize_shared_object()


def get_service_list()-> dict:
    task_response = APITaskResponse(output=service_registry.get_available_services())
    return task_response.model_dump()



@celery.task(name="AI_task", time_limit = CELERY_RESULT_EXPIRATION_TIME, max_retries = CELERY_MAX_RETRY, soft_time_limit = CELERY_SOFT_LIMIT)
def create_task(parameters: dict) -> dict:
    st = time.time()
    parameters : APIInput = APIInput(**parameters)
    app = service_registry.get_service(parameters.app_id)

    if(parameters.app_id in service_registry.api_services):
        if(parameters.app_id == "sdxlreplacebackground_api"):
            output_data = TaskResponse(**app(container.bg_remover()).remote(parameters.parameters))
        else:
            output_data = TaskResponse(**app().remote(parameters.parameters))

    elif(parameters.app_id in service_registry.modal_services):
        
        app = app.with_options(gpu=parameters.inference_type)

        app = app(parameters.init_parameters)
        
        output_data = TaskResponse(**app.run_inference.remote(parameters.parameters))
        
    else:
        raise Exception("Given APP Id is not available")


    if output_data.extension is not None:
        urls = [None] * len(output_data.result)
        low_res_urls = [None] * len(output_data.result)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            full_res_futures = [executor.submit(upload_image, i, img, output_data.extension, parameters.upscale) 
                                for i, img in enumerate(output_data.result)]

            if output_data.extension == OUTPUT_IMAGE_EXTENSION:
                low_res_futures = [executor.submit(upload_low_res_image, i, img, output_data.extension) 
                                for i, img in enumerate(output_data.result)]
            else:
                low_res_futures = []

            for future in concurrent.futures.as_completed(full_res_futures):
                index, url = future.result()
                urls[index] = url

            for future in concurrent.futures.as_completed(low_res_futures):
                index, url = future.result()
                low_res_urls[index] = url

        output_data.result = urls
        output_data.low_res_urls = low_res_urls

        if(output_data == None or output_data == {} or output_data == ""):
            raise Exception("Received an empty response from model")

    time_required = time.time() - st
    
    if (output_data.time.runtime< time_required < output_data.time.runtime + output_data.time.startup_time):
        output_data.time.startup_time = 0
    
    output_data.time.runtime = time_required - output_data.time.startup_time

    task_response = APITaskResponse(
         status="SUCCESS",
         input = parameters.model_dump(),
         output = {"result" : output_data.result, "Has_NSFW_Content" : output_data.Has_NSFW_Content, "low_res_urls" : output_data.low_res_urls},
         time_required = output_data.time.model_dump()
    )

    return task_response.model_dump()


def task_gen(parameters: APIInput) -> dict:

    if parameters.celery == True:
        task: AsyncResult = create_task.delay(parameters.model_dump())

        task_response = APITaskResponse(
            task_id=task.id, input=parameters.model_dump(), status=task.status
        ).model_dump()

        return task_response
    else:
        return create_task(parameters.model_dump())
    
def get_task_status(id : str) -> AsyncResult:
    task_update = celery.AsyncResult(id)
    return task_update

