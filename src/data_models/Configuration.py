from src.data_models.ModalAppSchemas import StubConfiguration, StubNames

stub_names = StubNames()

stub_dictionary : dict[str, StubConfiguration] = {
    stub_names.sdxl_text_to_image : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.sdxl_image_to_image : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.sdxl_controlnet : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.ultrasharp_upscaler : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.image_variation : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.face_consistent : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.background_removal : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.stable_video_diffusion : StubConfiguration(gpu = "a100", memory = 20000, container_idle_timeout = 60, num_containers=1),
    stub_names.illusion_diffusion : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.qr_code_generation : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.frnd_face_consistent : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
    stub_names.stable_cascade_text_to_image : StubConfiguration(gpu = "a10g", memory = 11000, container_idle_timeout = 60, num_containers=1),
}
















