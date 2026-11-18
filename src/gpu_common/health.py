"""GPU health check utilities."""

from gpu_common.memory import _torch


def check_gpu_available() -> bool:
    """True if a CUDA GPU is visible to this process.

    Deliberately does not allocate anything on the GPU: that would open a CUDA
    context (hundreds of MB) in the calling process for as long as it lives, and
    services call this from /health every few seconds.
    """
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage information for this process.

    Opens a CUDA context: call it only from processes that already use the GPU
    (e.g. inside a GPUWorker handler), never from a web server's /health.

    Returns:
        Dictionary with GPU memory statistics:
        - available: Whether GPU is available
        - total_mb: Total GPU memory in MB (if available)
        - allocated_mb: Currently allocated memory in MB (if available)
        - cached_mb: Cached (reserved) memory in MB (if available)
    """
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "total_mb": torch.cuda.get_device_properties(0).total_memory // 1024 // 1024,
        "allocated_mb": torch.cuda.memory_allocated(0) // 1024 // 1024,
        "cached_mb": torch.cuda.memory_reserved(0) // 1024 // 1024,
    }
