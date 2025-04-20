import os

from dependency_injector import containers, providers
from modal import Cls
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2
from transparent_background import Remover

# Do not remove this line. It imports the classes from this file into memory because of that, it is easy to identify that they are subscriber of Iservice class
####
from src.FastAPIServer.services.ApiServices.OpenAIService import GeminiAIImageCheck
from src.FastAPIServer.services.IService import IService
from src.utils.Globals import get_clean_name


class ServiceContainer(containers.DeclarativeContainer):
    bg_remover = providers.Singleton(Remover, device="cpu")
    image_checker = providers.Singleton(GeminiAIImageCheck)


@synchronizer.create_blocking
async def list_apps() -> list[str]:
    """
    Asynchronously retrieves available Modal applications from the Modal API.

    This function connects to the Modal platform API using authentication credentials
    from environment variables and retrieves the list of registered applications.
    The returned list is used to dynamically register Modal-based services in the
    ServiceRegistry.

    Returns:
        list[str]: List of application names registered in Modal
    """
    config = {
        "server_url": "https://api.modal.com",
        "token_id": os.environ["TOKEN_ID"],
        "token_secret": os.environ["TOKEN_SECRET"],
        "task_id": None,
        "task_secret": None,
    }
    client = await _Client.from_env(config)
    env = ensure_env(os.environ["environment"])

    res: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=env)
    )
    list_apps = []
    for i in res.apps:
        # if i.state == 3 and i.object_entity == "ap":
        list_apps.append(i.description)

    return list_apps


class ServiceRegistry:
    """
    Central registry for managing and accessing services.

    The ServiceRegistry dynamically discovers and registers both API-based services
    (derived from IService) and Modal-based services (from Modal's platform). It provides
    a unified interface for accessing all services through their unique app_ids.
    """

    def __init__(self, container: ServiceContainer):
        """
        Initializes the service registry with a dependency container.

        Args:
            container (ServiceContainer): The dependency injection container to use
                for registering and retrieving services
        """
        self.api_services = []
        self.modal_services = []
        self.container = container

    def register_internal_services(self):
        """
        Discovers and registers all available services.

        This method performs two types of service registration:
        1. API Services: Finds all classes that inherit from IService and registers them
           as singletons in the container with standardized naming
        2. Modal Services: Queries the Modal platform for available applications and
           registers them in the container

        This automatic discovery approach allows new services to be added with minimal
        configuration changes.
        """
        for cls in IService.__subclasses__():
            service_name = get_clean_name(cls.__name__)
            service_name = f"{service_name}_api"
            setattr(self.container, service_name, providers.Singleton(cls))
            self.api_services.append(service_name)
        for cls in list_apps():
            try:
                Model = Cls.lookup(
                    cls, "stableDiffusion", environment_name=os.environ["environment"]
                )
                cls = f"{cls}_modal"
                setattr(self.container, cls, Model)
                self.modal_services.append(cls)
            except:
                pass

    def register_new_modal_service(self, app_name):
        """
        Registers a new Modal service by name.

        This method attempts to find and register a Modal application that wasn't
        previously registered. It's used for on-demand registration when a service
        is requested but not yet available in the container.

        Args:
            app_name (str): Name of the Modal application to register

        Raises:
            Exception: If the specified application doesn't exist in Modal
        """
        app_list = list_apps()
        if app_name in app_list:
            Model = Cls.lookup(
                app_name, "stableDiffusion", environment_name=os.environ["environment"]
            )
            app_name = f"{app_name}_modal"
            setattr(self.container, app_name, providers.Factory(Model))
            self.modal_services.append(app_name)
        else:
            raise Exception(
                f"Given {app_name} does not exist, Please provide proper name"
            )

    def get_service(self, service_name: str):
        """
        Retrieves a service provider by name with lazy registration.

        This method attempts to find a service in the container by its app_id.
        If the service isn't registered yet, it attempts to register it as a
        Modal service before returning.

        Only Modal service needs to be registered. The reason is that we could deploy
        new modal service in between. It could be fetched from modal API. However, we
        could not create any subclass of IService directly without modifying code.
        So we do not need to register IService subclasses.

        Args:
            service_name (str): The unique identifier of the service to retrieve

        Returns:
            The service provider from the container

        Raises:
            Exception: If the service cannot be found or registered
        """
        service_provider = getattr(self.container, service_name, None)
        if not service_provider:
            self.register_new_modal_service(service_name)
            service_provider = getattr(self.container, service_name, None)
        return service_provider

    def get_available_services(self) -> list:
        """
        Lists all registered services.

        Returns a combined list of all registered API and Modal services,
        which can be used to inform clients about available functionality.

        Returns:
            list: Combined list of api_services and modal_services identifiers
        """
        return self.api_services + self.modal_services

    # # Register SDK services


# Fetch SDK services and register all services
# sdk_service_classes = get_sdk_services()
# register_services(ServiceContainer, sdk_service_classes)
