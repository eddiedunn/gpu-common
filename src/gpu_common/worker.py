"""Run a service's GPU models in a child process that exits when idle.

Unloading a model inside a long-lived process does not reliably give its GPU memory
back: CUDA contexts, CTranslate2 and ONNX Runtime pools, and references cached deep
inside libraries keep gigabytes reserved. Ending the process always does — the driver
reclaims everything it held. ``GPUWorker`` keeps the web server in the parent (so
/health and the API stay up) and does all GPU work in a child it can stop.

The parent must never touch CUDA itself, or it holds a context of its own: no models,
no ``torch.zeros(..., device="cuda")`` health probes.

Usage::

    # mypkg/gpu_handler.py — imported only in the child
    class Handler:
        def __init__(self):            # runs in the child; load models here or lazily
            self.engine = load_engine()
        def transcribe(self, path):     # args and return value must be picklable
            return self.engine.transcribe(path)

    worker = GPUWorker("mypkg.gpu_handler:Handler", name="transcription engine")
    await worker.start()               # idle-stop loop
    result = await worker.call("transcribe", "/tmp/a.wav")
    await worker.stop()
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing
import os
import traceback
from datetime import datetime, timedelta
from typing import Any

from gpu_common.config import GPUSettings
from gpu_common.memory import is_gpu_oom

logger = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    """A call failed inside the GPU worker process."""

    def __init__(self, type_name: str, message: str, remote_traceback: str = ""):
        super().__init__(f"{type_name}: {message}")
        self.type_name = type_name
        self.message = message
        self.remote_traceback = remote_traceback


class GPUOutOfMemoryError(WorkerError):
    """The GPU ran out of memory inside the worker. The worker has been stopped, so a
    retry starts from a clean process."""


class WorkerCrashed(WorkerError):
    """The worker process died mid-call (killed, segfault, driver error)."""


def _as_timedelta(value: timedelta | float | int) -> timedelta:
    return value if isinstance(value, timedelta) else timedelta(seconds=value)


def _resolve(factory: str):
    module_name, _, attr = factory.partition(":")
    if not attr:
        raise ValueError(f"factory must look like 'package.module:Name', got {factory!r}")
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _child_main(conn, factory: str) -> None:
    """Entry point of the worker process."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s [worker]: %(message)s",
        force=True,
    )
    log = logging.getLogger("gpu_common.worker.child")
    try:
        handler = _resolve(factory)()
    except BaseException as exc:
        log.exception("worker failed to start")
        conn.send(("err", type(exc).__name__, str(exc), traceback.format_exc(), is_gpu_oom(exc)))
        return
    conn.send(("ready",))
    while True:
        try:
            method, args, kwargs = conn.recv()
        except (EOFError, OSError):
            return  # parent went away
        try:
            reply = ("ok", getattr(handler, method)(*args, **kwargs))
        except BaseException as exc:
            log.exception("%s failed", method)
            reply = ("err", type(exc).__name__, str(exc), traceback.format_exc(), is_gpu_oom(exc))
        try:
            conn.send(reply)
        except Exception as exc:  # e.g. an unpicklable result
            conn.send(("err", type(exc).__name__, f"could not send result: {exc}", "", False))


class GPUWorker:
    """Owns one worker process: starts it on first call, stops it after ``idle_timeout``.

    Calls run one at a time. ``idle_timeout`` defaults to ``GPU_IDLE_TIMEOUT`` (seconds,
    default 300).
    """

    def __init__(
        self,
        factory: str,
        idle_timeout: timedelta | float | int | None = None,
        cleanup_interval: timedelta | float | int = timedelta(seconds=30),
        name: str = "GPU worker",
        stop_grace: float = 10.0,
    ):
        self._factory = factory
        self._name = name
        self._idle_timeout = _as_timedelta(
            GPUSettings().idle_timeout if idle_timeout is None else idle_timeout
        )
        self._cleanup_interval = _as_timedelta(cleanup_interval)
        self._stop_grace = stop_grace
        self._ctx = multiprocessing.get_context("spawn")  # CUDA cannot survive fork
        self._proc = None
        self._conn = None
        self._last_used: datetime | None = None
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    @property
    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @property
    def idle_timeout(self) -> timedelta:
        return self._idle_timeout

    async def start(self) -> None:
        """Start the idle-stop loop. Without it the worker is never stopped."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the idle loop and the worker."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        await self.unload()

    async def unload(self) -> None:
        """Stop the worker now; the next call starts a new one."""
        async with self._lock:
            await asyncio.to_thread(self._terminate, "unload requested")

    def warm(self) -> asyncio.Task:
        """Start the worker in the background (e.g. on a wake word)."""
        return asyncio.create_task(self._warm())

    async def _warm(self) -> None:
        async with self._lock:
            if not self.is_loaded:
                await asyncio.to_thread(self._spawn)
            self._last_used = datetime.now()

    def touch(self) -> None:
        if self.is_loaded:
            self._last_used = datetime.now()

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Run ``handler.method(*args, **kwargs)`` in the worker and return its result."""
        async with self._lock:
            if not self.is_loaded:
                await asyncio.to_thread(self._spawn)
            try:
                reply = await asyncio.to_thread(self._roundtrip, (method, args, kwargs))
            except asyncio.CancelledError:
                # The child is still busy with this request and the pipe is out of step;
                # stopping it is the only clean way back (and frees the GPU).
                self._terminate("request cancelled", grace=0)
                raise
            except (EOFError, OSError) as exc:
                code = self._proc.exitcode if self._proc is not None else None
                self._terminate("worker died", grace=0)
                raise WorkerCrashed(
                    "WorkerCrashed", f"{self._name} process died during {method} (exit code {code})"
                ) from exc
            finally:
                self._last_used = datetime.now()
            if reply[0] == "ok":
                return reply[1]
            _, type_name, message, tb, oom = reply
            if oom:
                self._terminate("GPU out of memory")
                raise GPUOutOfMemoryError(type_name, message, tb)
            raise WorkerError(type_name, message, tb)

    # -- blocking helpers, run in a thread --------------------------------------

    def _roundtrip(self, msg):
        self._conn.send(msg)
        return self._conn.recv()

    def _spawn(self) -> None:
        logger.info("Starting %s process", self._name)
        started = datetime.now()
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_child_main, args=(child_conn, self._factory), name=self._name, daemon=True
        )
        proc.start()
        child_conn.close()
        self._proc, self._conn = proc, parent_conn
        try:
            reply = parent_conn.recv()
        except (EOFError, OSError) as exc:
            self._terminate("worker died while starting")
            raise WorkerCrashed(
                "WorkerCrashed", f"{self._name} process died while starting"
            ) from exc
        if reply[0] != "ready":
            self._terminate("worker failed to start")
            _, type_name, message, tb, oom = reply
            raise (GPUOutOfMemoryError if oom else WorkerError)(type_name, message, tb)
        logger.info("%s ready in %.1fs", self._name, (datetime.now() - started).total_seconds())

    def _terminate(self, reason: str, grace: float | None = None) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        self._last_used = None
        if proc is None:
            return
        logger.info("Stopping %s process (%s)", self._name, reason)
        if conn is not None:
            conn.close()  # the child sees EOF and exits on its own when idle
        proc.join(self._stop_grace if grace is None else grace)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        logger.info("%s stopped, GPU memory released", self._name)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval.total_seconds())
            try:
                async with self._lock:
                    if (
                        self.is_loaded
                        and self._last_used is not None
                        and datetime.now() - self._last_used > self._idle_timeout
                    ):
                        await asyncio.to_thread(self._terminate, f"idle for {self._idle_timeout}")
                    elif self._proc is not None and not self._proc.is_alive():
                        await asyncio.to_thread(self._terminate, "worker exited")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("Idle stop of %s failed\n%s", self._name, traceback.format_exc())
