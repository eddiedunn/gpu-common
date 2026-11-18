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


def is_gpu_oom(exc: BaseException | None) -> bool:
    """True if a GPU out-of-memory error is anywhere in the exception chain.

    Matches torch's OutOfMemoryError and the "out of memory" messages CTranslate2,
    cuDNN, cuBLAS and ONNX Runtime raise as plain RuntimeErrors.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc).lower()
        if (
            type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in text
            or "cuda_error_out_of_memory" in text
            or "failed to allocate memory" in text
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False
