"""Shared GPU configuration settings."""

from pydantic_settings import BaseSettings


class GPUSettings(BaseSettings):
    """GPU service configuration settings."""

    redis_url: str = "redis://localhost:6379"
    idle_timeout: int = 300  # seconds

    class Config:
        env_prefix = "GPU_"
