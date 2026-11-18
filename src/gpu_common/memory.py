"""Return freed GPU memory to the driver.

PyTorch is optional: services on ONNX Runtime or CTranslate2 alone can use gpu_common
without installing it. Those frameworks keep their own memory pools, which are only
released by destroying their sessions/models — pass a ``release_fn`` to
``GPUModelManager`` for that.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def free_gpu_memory() -> None:
    """Collect garbage, then hand PyTorch's cached GPU blocks back to the driver.

    The order matters: models inside reference cycles are only freed by the cycle
    collector, and memory they release after ``empty_cache()`` stays reserved by the
    caching allocator until the next call.
    """
    gc.collect()
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
