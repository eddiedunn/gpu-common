"""Generic GPU model manager with lazy loading and idle cleanup."""

import torch
import asyncio
import logging
from typing import TypeVar, Generic, Callable
from datetime import datetime, timedelta

T = TypeVar('T')  # Model type

class GPUModelManager(Generic[T]):
    """Manages GPU model lifecycle with lazy loading and cleanup."""

    def __init__(
        self,
        load_fn: Callable[[], T],
        idle_timeout: timedelta = timedelta(minutes=5),
        cleanup_interval: timedelta = timedelta(seconds=30)
    ):
        self._load_fn = load_fn
        self._model: T | None = None
        self._last_used: datetime | None = None
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)

    async def start(self):
        """Start the cleanup background task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop cleanup task and unload model."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.unload()

    async def get(self) -> T:
        """Get model, loading if necessary."""
        async with self._lock:
            if self._model is None:
                self._logger.info("Loading model into GPU memory")
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(None, self._load_fn)
                self._logger.info("Model loaded successfully")
            self._last_used = datetime.now()
            return self._model

    async def unload(self):
        """Explicitly unload model and free GPU memory."""
        async with self._lock:
            if self._model is not None:
                self._logger.info("Unloading model from GPU memory")
                del self._model
                self._model = None
                self._last_used = None

                # Critical: Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                self._logger.info("GPU memory freed")

    async def _cleanup_loop(self):
        """Background task to unload idle models."""
        while True:
            await asyncio.sleep(self._cleanup_interval.total_seconds())
            async with self._lock:
                if (
                    self._model is not None
                    and self._last_used is not None
                    and datetime.now() - self._last_used > self._idle_timeout
                ):
                    self._logger.info(f"Model idle for {self._idle_timeout}, unloading")
                    # Unload directly here since we already hold the lock
                    del self._model
                    self._model = None
                    self._last_used = None

                    # Critical: Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    self._logger.info("GPU memory freed")
