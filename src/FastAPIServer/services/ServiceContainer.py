import os

from dependency_injector import containers, providers
from modal import Cls
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2
from transparent_background import Remover

from src.FastAPIServer.services.ApiServices.ClaudeAIService import (
    PromptParrot,
    VideoPromptParrot,
)
from src.FastAPIServer.services.ApiServices.ElvenLabsAudio import ElvenLabsAudio
from src.FastAPIServer.services.ApiServices.FalAIService import (
    FalAIFluxProText2Image,
    FalAIFluxDevText2Image,
    FalAIFluxschnellText2Image,
    FalAIFluxDevImage2Image,
    FalAIRefactorV3Text2Image,
    FalAISD35LargeText2Image,
    FalAISD35LargeTurboText2Image,
    FalAISD35MediumText2Image,
    FalAIFlux3Inpainting,
    FalAIFlux3ReplaceBackground,
    FalAIFluxProRedux,
    FalAIFluxProCanny,
    FalAIFluxProDepth,
    OmnigenV1,
    FalAIFluxPulID,
    Veo2,
    Kling2Master,
)
from src.FastAPIServer.services.ApiServices.IdeogramService import (
    IdeoGramText2Image,
)  # Renamed to avoid conflict
from src.FastAPIServer.services.ApiServices.LeonardoService import LeonardoText2Image
from src.FastAPIServer.services.ApiServices.LumaLabsService import LumaVideo
from src.FastAPIServer.services.ApiServices.OpenAIService import (
    DalleText2Image,
    GeminiAIImageCheck,
    GPTText2Image
)
from src.FastAPIServer.services.ApiServices.RunWayService import RunwayImage2Video
from src.FastAPIServer.services.ApiServices.SDXLService import SDXL3Text2Image
from src.FastAPIServer.services.ApiServices.VertexAIService import (
    ImageGenText2Image,
    VertexAIVeo,
    VertexAIVeo3Fast,
    VertexAIVeo3,
    VeoRouterService,
)


from src.FastAPIServer.services.IService import IService
from src.utils.Globals import get_clean_name
from loguru import logger


class ServiceContainer(containers.DeclarativeContainer):
    bg_remover = providers.Singleton(Remover, device="cpu")
    image_checker = providers.Singleton(GeminiAIImageCheck)
    # --- Explicit Providers for Router Dependencies ---
    vertexaiveo_api = providers.Singleton(VertexAIVeo)
    veo2_api = providers.Singleton(Veo2)  # This is Fal Veo2
    
    # --- Explicit Providers for New Vertex AI Veo Models ---
    vertexaiveo3fast_api = providers.Singleton(VertexAIVeo3Fast)
    vertexaiveo3_api = providers.Singleton(VertexAIVeo3)

    # --- {{ CORRECTED Router Factory Definition }} ---
    # Define the router factory, injecting the PROVIDERS using .provider
    veorouterservice_api = providers.Factory(
        VeoRouterService,
        vertex_veo_provider=vertexaiveo_api.provider,  # Pass the provider callable
        fal_veo_provider=veo2_api.provider,  # Pass the provider callable
    )


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
           as singletons in the container with standardized naming (unless explicitly defined above)
        2. Modal Services: Queries the Modal platform for available applications and
           registers them in the container

        This automatic discovery approach allows new services to be added with minimal
        configuration changes. Explicit providers in ServiceContainer override registrations here.
        """
        logger.info("Starting service registration...")
        
        # --- {{ Keep Original IService Loop }} ---
        discovered_classes = IService.__subclasses__()
        logger.info(f"Discovered {len(discovered_classes)} IService subclasses: {[cls.__name__ for cls in discovered_classes]}")
        
        # Check specifically for our Vertex AI classes
        vertex_classes = [cls for cls in discovered_classes if 'Vertex' in cls.__name__]
        logger.info(f"Vertex AI classes found: {[cls.__name__ for cls in vertex_classes]}")
        
        for cls in discovered_classes:
            # Skip the router class if found here, it's handled explicitly above
            if cls == VeoRouterService:
                logger.info(f"Skipping {cls.__name__} (handled explicitly)")
                continue

            service_name = get_clean_name(cls.__name__)
            service_name_api = f"{service_name}_api"
            logger.info(f"Processing class {cls.__name__} -> {service_name_api}")

            # Check if this provider was already explicitly defined in ServiceContainer
            if not hasattr(self.container, service_name_api):
                # Only register via Singleton if not explicitly defined
                setattr(self.container, service_name_api, providers.Singleton(cls))
                # Add to list only if registered here
                if service_name_api not in self.api_services:
                    self.api_services.append(service_name_api)
                logger.info(f"Successfully registered {service_name_api} for class {cls.__name__}")
            elif service_name_api not in self.api_services:
                # If it *was* explicitly defined but not in the list yet, add it.
                # Handles cases like veo2_api which is explicit but needs to be listed.
                self.api_services.append(service_name_api)
                logger.info(f"Added existing service {service_name_api} to api_services list")

        # --- {{ Keep Original Modal Loop }} ---
        for cls_name in list_apps():
            try:
                Model = Cls.lookup(
                    cls_name,
                    "stableDiffusion",
                    environment_name=os.environ["environment"],
                )
                modal_service_name = f"{cls_name}_modal"
                # Check if explicitly defined or already added
                if (
                    not hasattr(self.container, modal_service_name)
                    and modal_service_name not in self.modal_services
                ):
                    setattr(self.container, modal_service_name, Model)
                    self.modal_services.append(modal_service_name)
            except Exception as e:
                print(f"Warning: Could not lookup/register Modal app '{cls_name}': {e}")
                pass

        # --- {{ Add Router to List (Minimal Change) }} ---
        # Ensure the explicitly defined router service is in the list for discovery
        router_service_name = "veorouterservice_api"
        if router_service_name not in self.api_services:
            self.api_services.append(router_service_name)
            
        # Add our new explicitly defined Vertex AI services
        new_vertex_services = ["vertexaiveo2_api", "vertexaiveo3fast_api", "vertexaiveo3_api"]
        for service_name in new_vertex_services:
            if service_name not in self.api_services:
                self.api_services.append(service_name)
        # --- End Minimal Addition ---
        
        logger.info(f"Service registration complete. API services: {self.api_services}")
        logger.info(f"Modal services: {self.modal_services}")

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
                "internal error",
                f"Given {app_name} does not exist, Please provide proper name",
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
