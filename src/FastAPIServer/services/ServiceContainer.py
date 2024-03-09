from dependency_injector import containers, providers
from src.services.ApiServices.IService import IService
from src.utils.Globals import get_clean_name
from fastapi import HTTPException

class ServiceContainer(containers.DeclarativeContainer):
    pass


class ServiceRegistry:
    def __init__(self):
        self._services = []

    def register_internal_services(self, container : ServiceContainer):
        for cls in IService.__subclasses__():
            service_name = get_clean_name(cls.__name__)
            setattr(container, service_name, providers.Singleton(cls))
            self._services.append(service_name)

    def get_service(self, service_name : str, container : ServiceContainer):
        service_name = get_clean_name(service_name)
        service_provider = getattr(container, service_name, None)
        if not service_provider:
            raise HTTPException(status_code=404, detail="Service not found")
        return service_provider()
    
    def get_available_services(self) -> list:
        return self._services

    # # Register SDK services
    # for cls in sdk_service_classes:
    #     service_id = cls.__name__.lower()
    #     setattr(container, service_id, providers.Singleton(cls))

# Fetch SDK services and register all services
# sdk_service_classes = get_sdk_services()
# register_services(ServiceContainer, sdk_service_classes)