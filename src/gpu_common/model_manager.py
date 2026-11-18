"""Generic GPU model manager with lazy loading and idle unload."""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Callable, Generic, TypeVar

from gpu_common.config import GPUSettings
from gpu_common.memory import free_gpu_memory

T = TypeVar("T")


def _as_timedelta(value: timedelta | float | int) -> timedelta:
    return value if isinstance(value, timedelta) else timedelta(seconds=value)


class GPUModelManager(Generic[T]):
    """Loads a model on first use and unloads it after it sits idle.

    Unloading only gives memory back if the manager holds the *only* reference to the
    model: callers must not keep it in attributes or module globals, only in locals for
    the duration of a request.

    ``release_fn`` is called with the model before it is dropped. Use it for frameworks
    whose GPU memory ``torch.cuda.empty_cache()`` cannot free (ONNX Runtime sessions,
    CTranslate2 models) — e.g. delete the sessions or call ``unload_model()``.

    ``idle_timeout`` defaults to the ``GPU_IDLE_TIMEOUT`` env var (seconds, default 300).
    """

    def __init__(
        self,
        load_fn: Callable[[], T],
        idle_timeout: timedelta | float | int | None = None,
        cleanup_interval: timedelta | float | int = timedelta(seconds=30),
        release_fn: Callable[[T], None] | None = None,
        name: str = "model",
    ):
        self._load_fn = load_fn
        self._release_fn = release_fn
        self._name = name
        self._model: T | None = None
        self._last_used: datetime | None = None
        self._idle_timeout = _as_timedelta(
            GPUSettings().idle_timeout if idle_timeout is None else idle_timeout
        )
        self._cleanup_interval = _as_timedelta(cleanup_interval)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def idle_timeout(self) -> timedelta:
        return self._idle_timeout

    async def start(self) -> None:
        """Start the idle-unload loop. Without this, models are never unloaded."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the idle-unload loop and unload the model."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        await self.unload()

    async def get(self) -> T:
        """Return the model, loading it if necessary, and mark it used."""
        async with self._lock:
            if self._model is None:
                self._logger.info("Loading %s into GPU memory", self._name)
                # Hand the model back through a box, not as the executor future's
                # result: asyncio/concurrent futures linger after completion and would
                # keep the model (and its GPU memory) alive past unload.
                box: list[T] = []
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: box.append(self._load_fn()))
                self._model = box.pop()
                self._logger.info("%s loaded", self._name)
            self._last_used = datetime.now()
            return self._model

    def warm(self) -> asyncio.Task:
        """Start loading in the background (e.g. on a wake word) without waiting."""
        return asyncio.create_task(self.get())

    def touch(self) -> None:
        """Mark the model used, for long jobs that hold it past the idle timeout."""
        if self._model is not None:
            self._last_used = datetime.now()

    async def unload(self) -> None:
        """Unload the model now and give its GPU memory back."""
        async with self._lock:
            self._release_locked(reason="unload requested")

    def _release_locked(self, reason: str) -> None:
        if self._model is None:
            return
        model, self._model, self._last_used = self._model, None, None
        self._logger.info("Unloading %s (%s)", self._name, reason)
        try:
            if self._release_fn is not None:
                self._release_fn(model)
        except Exception:
            # Log the traceback as text: a LogRecord holding exc_info would keep the
            # failed frame, and the model in it, alive.
            self._logger.error(
                "release_fn for %s failed; dropping it anyway\n%s", self._name, traceback.format_exc()
            )
        finally:
            del model
            free_gpu_memory()
        self._logger.info("%s unloaded, GPU memory freed", self._name)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval.total_seconds())
            try:
                async with self._lock:
                    if (
                        self._model is not None
                        and self._last_used is not None
                        and datetime.now() - self._last_used > self._idle_timeout
                    ):
                        self._release_locked(reason=f"idle for {self._idle_timeout}")
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never let one failure stop idle unloading for the life of the process.
                self._logger.error("Idle unload of %s failed\n%s", self._name, traceback.format_exc())
