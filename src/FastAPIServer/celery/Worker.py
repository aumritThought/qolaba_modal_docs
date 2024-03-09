from celery import Celery
import modal, time, os
from celery.signals import worker_init, worker_process_init
from src.utils.Constants import REDIS_URL, CELERY_RESULT_EXPIRATION_TIME, CELERY_MAX_RETRY, CELERY_SOFT_LIMIT
from src.utils.Globals import get_modal_apps, BuildTaskResponse, ParametersModification
from data_models.Schemas import (
    Text2ImageParameters,
    Image2ImageParameters,
    Text2TextParameters,
    VideoParameters,
)
from typing import Union
from src.utils.Constants import api_apps
from dotenv import load_dotenv
from celery.result import AsyncResult

load_dotenv()

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


    # class AIGeneration:
    # def __init__(self) -> None:
    #     self.apps = get_modal_apps()
    #     self.app_functions: dict[str, modal.Function] = {}

    #     for app_id in self.apps.keys():
    #         if not (app_id in api_apps.keys()):
    #             func = modal.Function.lookup(
    #                 self.apps[app_id],
    #                 "stableDiffusion.run_inference",
    #                 environment_name=os.environ["environment"],
    #             )
    #             self.app_functions[app_id] = func
    #         else:
    #             self.app_functions[app_id] = globals()[api_apps[app_id]]()





@celery.task( name="AI_task", time_limit = CELERY_RESULT_EXPIRATION_TIME, max_retries = CELERY_MAX_RETRY, soft_time_limit = CELERY_SOFT_LIMIT)
def create_task( parameters: Union[Text2ImageParameters, Image2ImageParameters, Text2TextParameters, VideoParameters]) -> dict:

    st = time.time()

    param_builder = ParametersModification(parameters)
    app_parameters, parameters = param_builder.get_app_specific_input_parameters()

    if parameters.app_id in api_apps.keys():
        dict_data = app_functions[parameters.app_id].remote(parameters)
    else:
        dict_data = app_functions[parameters.app_id].remote(**app_parameters)

    time_required = time.time() - st

    if (dict_data["time"]["runtime"]< time_required < dict_data["time"]["startup_time"] + dict_data["time"]["runtime"]):
        dict_data["time"]["startup_time"] = 0
    
    dict_data["time"]["runtime"] = time_required - dict_data["time"]["startup_time"]

    task_response = BuildTaskResponse(
            parameters=parameters,
            app_parameters=app_parameters,
            output=dict_data,
            status="SUCCESS",
        ).prepare_response()
    return task_response


def task_gen( parameters: Union[Text2ImageParameters, Image2ImageParameters, Text2TextParameters, VideoParameters], category: str) -> dict:
    if not category.lower() in get_modal_apps()[parameters.app_id].lower():
        raise Exception("Invalid App_id", f"App_id does not belongs to {category}")

    if parameters.celery == True:
        task: AsyncResult = create_task.delay(parameters)

        task_response = BuildTaskResponse(
            task_id=task.id, parameters=parameters, status=task.status
        ).prepare_response()

        return task_response
    else:
        return create_task(parameters)



def get_task_status(id : str) -> AsyncResult:
    result: AsyncResult = celery.AsyncResult(id)
    return result
