"""GPU Common - Shared GPU utilities for AI services."""

from gpu_common.lock import GPULock
from gpu_common.model_manager import GPUModelManager
from gpu_common.memory import free_gpu_memory, is_gpu_oom
from gpu_common.worker import GPUOutOfMemoryError, GPUWorker, WorkerCrashed, WorkerError
from gpu_common.health import check_gpu_available, get_gpu_memory_info
from gpu_common.config import GPUSettings

__version__ = "0.3.0"

__all__ = [
    "GPULock",
    "GPUModelManager",
    "check_gpu_available",
    "get_gpu_memory_info",
    "GPUSettings",
    "free_gpu_memory",
    "is_gpu_oom",
    "GPUWorker",
    "WorkerError",
    "WorkerCrashed",
    "GPUOutOfMemoryError",
]
