"""GPU health check utilities."""

import torch


def check_gpu_available() -> bool:
    """Verify GPU is accessible and has memory.

    Returns:
        True if GPU is available and functional, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    try:
        test = torch.zeros(1, device='cuda')
        del test
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        return False


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage information.

    Returns:
        Dictionary with GPU memory statistics:
        - available: Whether GPU is available
        - total_mb: Total GPU memory in MB (if available)
        - allocated_mb: Currently allocated memory in MB (if available)
        - cached_mb: Cached (reserved) memory in MB (if available)
    """
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "total_mb": torch.cuda.get_device_properties(0).total_memory // 1024 // 1024,
        "allocated_mb": torch.cuda.memory_allocated(0) // 1024 // 1024,
        "cached_mb": torch.cuda.memory_reserved(0) // 1024 // 1024,
    }
