from dependency_injector import containers, providers

# Do not remove this line. It imports the classes from this file into memory because of that, it is easy to identify that they are subscriber of Iservice class
from src.FastAPIServer.services.ApiServices import DIDVideoService, ElvenLabsAudio, OpenAIService, SDXLService, ClaudeAIService
####

from src.FastAPIServer.services.IService import IService
from src.utils.Globals import get_clean_name
from fastapi import HTTPException
from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
import os
from modal import Cls
from transparent_background import Remover


class ServiceContainer(containers.DeclarativeContainer):
    bg_remover = providers.Singleton(Remover, device="cpu")


@synchronizer.create_blocking
async def list_apps() -> list[str]:
    config = {'server_url': 'https://api.modal.com', 'token_id': os.environ["TOKEN_ID"], 'token_secret': os.environ["TOKEN_SECRET"], 'task_id': None, "task_secret" : None}
    client = await _Client.from_env(config)
    env = ensure_env(os.environ["environment"])

    res: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=env)
    )
    list_apps = []
    for i in res.apps:
        if i.state == 3 and i.object_entity == "ap":
            list_apps.append(i.description)

    return list_apps


class ServiceRegistry:
    def __init__(self, container : ServiceContainer):
        self.api_services = []
        self.modal_services = []
        self.container = container

    def register_internal_services(self):
        for cls in IService.__subclasses__():
            service_name = get_clean_name(cls.__name__)
            service_name = f"{service_name}_api"
            setattr(self.container, service_name, providers.Singleton(cls))
            self.api_services.append(service_name)
        for cls in list_apps():      
            Model = Cls.lookup(cls, "stableDiffusion", environment_name = os.environ["environment"])
            cls = f"{cls}_modal"
            setattr(self.container, cls, Model)
            self.modal_services.append(cls)

    def register_new_modal_service(self, app_name):
        app_list = list_apps()
        if(app_name in app_list):
            Model = Cls.lookup(app_name, "stableDiffusion", environment_name = os.environ["environment"])
            app_name = f"{app_name}_modal"
            setattr(self.container, app_name, providers.Factory(Model))
            self.modal_services.append(app_name)
        else:
            raise Exception(f"Given {app_name} does not exist, Please provide proper name")

    def get_service(self, service_name : str):
        service_provider = getattr(self.container, service_name, None)
        if not service_provider:
            self.register_new_modal_service(service_name)
            service_provider = getattr(self.container, service_name, None)
        return service_provider
    
    def get_available_services(self) -> list:
        return self.api_services + self.modal_services

    # # Register SDK services


# Fetch SDK services and register all services
# sdk_service_classes = get_sdk_services()
# register_services(ServiceContainer, sdk_service_classes)