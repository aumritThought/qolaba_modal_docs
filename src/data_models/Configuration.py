from src.data_models.ModalAppSchemas import StubConfiguration, StubNames

stub_names = StubNames()
"""
Configuration dictionary for Modal app stubs.

This dictionary provides a centralized configuration for all Modal application stubs, allowing for:
- Consistent infrastructure management across all services
- Single source of truth for resource requirements
- Easy comparison and auditing of resource allocation
- Simple scaling adjustments when requirements change
- Reduced configuration duplication throughout the codebase

Each entry configures:
- gpu: GPU type required (a10g, a100, h100) based on model complexity and performance needs
- memory: Memory allocation in MB to ensure stable model loading and inference
- container_idle_timeout: Time in seconds before container shuts down when idle to optimize costs
- num_containers: Number of containers to provision for handling concurrent requests
"""
stub_dictionary: dict[str, StubConfiguration] = {
    stub_names.sdxl_text_to_image: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.sdxl_image_to_image: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.sdxl_controlnet: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.ultrasharp_upscaler: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.image_variation: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.face_consistent: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.background_removal: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.stable_video_diffusion: StubConfiguration(
        gpu="a100", memory=20000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.illusion_diffusion: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.qr_code_generation: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.frnd_face_consistent: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.stable_cascade_text_to_image: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.oot_diffusion: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.hair_fast: StubConfiguration(
        gpu="a10g", memory=11000, container_idle_timeout=60, num_containers=1
    ),
    stub_names.flux_refiner: StubConfiguration(
        gpu="h100", memory=11000, container_idle_timeout=60, num_containers=1
    ),
}
