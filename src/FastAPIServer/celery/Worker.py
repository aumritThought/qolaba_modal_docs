from celery import Celery
import modal, time, os
from celery.signals import worker_init, worker_process_init
from src.utils.Constants import REDIS_URL, CELERY_RESULT_EXPIRATION_TIME, CELERY_MAX_RETRY, CELERY_SOFT_LIMIT
from src.data_models.ModalAppSchemas import APIInput, APITaskResponse, TaskResponse
from dotenv import load_dotenv
from celery.result import AsyncResult
from src.FastAPIServer.services.ServiceContainer import ServiceContainer, ServiceRegistry
from src.FastAPIServer.services.IService import IService
load_dotenv()

# celery = Celery(
#     "task_scheduler",
#     broker = REDIS_URL,
#     backend = REDIS_URL,
# )

# celery.conf.update(
#     result_expires = CELERY_RESULT_EXPIRATION_TIME
# )

# celery.conf.CELERY_TASK_SOFT_TIME_LIMIT = CELERY_RESULT_EXPIRATION_TIME

# celery.conf.CELERY_TASK_TIME_LIMIT = CELERY_RESULT_EXPIRATION_TIME


# @celery.task( name="AI_task", time_limit = CELERY_RESULT_EXPIRATION_TIME, max_retries = CELERY_MAX_RETRY, soft_time_limit = CELERY_SOFT_LIMIT)

container = ServiceContainer()
service_registry = ServiceRegistry(container)
service_registry.register_internal_services() 

def get_service_list()-> dict:
    task_response = APITaskResponse(output=service_registry.get_available_services())
    return task_response.model_dump()

def create_task(parameters: APIInput) -> dict:

    st = time.time()

    app = service_registry.get_service(parameters.app_id)
    if(parameters.app_id in service_registry.api_services):
        output_data = app().remote(parameters.parameters)
    elif(parameters.app_id in service_registry.modal_services):
        app = app(parameters.init_parameters)
        output_data = app.run_inference.remote(parameters.parameters)
    else:
        raise Exception("Given APP Id is not available", "Internal Error")

    output_data = TaskResponse(**output_data)
    time_required = time.time() - st

    if (output_data.time.runtime< time_required < output_data.time.runtime + output_data.time.startup_time):
        output_data.time.startup_time = 0
    
    output_data.time.runtime = time_required - output_data.time.startup_time

    task_response = APITaskResponse(
         status="SUCCESS",
         input = parameters.model_dump(),
         output = {"result" : output_data.result, "Has_NSFW_Content" : output_data.time},
         time_required = output_data.time.model_dump()
    )

    return task_response.model_dump()


def task_gen(parameters: APIInput) -> dict:

    # if parameters.celery == True:
    #     task: AsyncResult = create_task.delay(parameters)

    #     task_response = BuildTaskResponse(
    #         task_id=task.id, parameters=parameters, status=task.status
    #     ).prepare_response()

    #     return task_response
    # else:
        return create_task(parameters)



# def get_task_status(id : str) -> AsyncResult:
#     result: AsyncResult = celery.AsyncResult(id)
#     return result
