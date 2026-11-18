"""Shared GPU configuration settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class GPUSettings(BaseSettings):
    """GPU service configuration settings."""

    model_config = SettingsConfigDict(env_prefix="GPU_")

    redis_url: str = "redis://localhost:6379"
    idle_timeout: int = 300  # seconds before an idle model is unloaded (GPU_IDLE_TIMEOUT)
