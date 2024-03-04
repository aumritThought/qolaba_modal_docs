from src.data_models.ModalAppSchemas import StubConfiguration, StubNames

stub_names = StubNames()

stub_dictionary : dict[str, StubConfiguration] = {
    stub_names.sdxl_text_to_image : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 200),
    stub_names.sdxl_image_to_image : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 200),
    stub_names.sdxl_controlnet : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 200),
}
















