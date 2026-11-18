"""Tests for GPUWorker: the GPU work runs in a child process that is stopped when idle."""

import asyncio
import os
from datetime import timedelta

import pytest

from gpu_common import GPUOutOfMemoryError, GPUWorker, WorkerCrashed, WorkerError, is_gpu_oom


def worker(factory="worker_handlers:Echo", **kw) -> GPUWorker:
    kw.setdefault("cleanup_interval", timedelta(seconds=0.1))
    return GPUWorker(factory, **kw)


@pytest.mark.asyncio
async def test_starts_on_first_call_in_another_process_and_reuses_it():
    w = worker()
    assert not w.is_loaded
    assert await w.call("echo", {"a": [1, 2]}) == {"a": [1, 2]}
    pid = await w.call("pid")
    assert pid != os.getpid() and w.is_loaded
    assert await w.call("pid") == pid
    await w.stop()
    assert not w.is_loaded


@pytest.mark.asyncio
async def test_idle_worker_process_is_stopped_and_restarted_on_demand():
    w = worker(idle_timeout=0.2)
    await w.start()
    first = await w.call("pid")
    await asyncio.sleep(1.0)
    assert not w.is_loaded, "idle worker process was not stopped"
    assert await w.call("pid") != first
    await w.stop()


@pytest.mark.asyncio
async def test_errors_come_back_with_type_and_remote_traceback_and_keep_worker():
    w = worker()
    pid = await w.call("pid")
    with pytest.raises(WorkerError) as info:
        await w.call("boom")
    assert info.value.type_name == "ValueError" and "bad input" in str(info.value)
    assert "def boom" in info.value.remote_traceback or "boom" in info.value.remote_traceback
    assert await w.call("pid") == pid
    await w.stop()


@pytest.mark.asyncio
async def test_out_of_memory_stops_worker_so_a_retry_starts_clean():
    w = worker()
    pid = await w.call("pid")
    with pytest.raises(GPUOutOfMemoryError):
        await w.call("oom")
    assert not w.is_loaded
    assert await w.call("pid") != pid
    await w.stop()


@pytest.mark.asyncio
async def test_crashed_worker_raises_and_next_call_starts_a_new_one():
    w = worker()
    await w.call("pid")
    with pytest.raises(WorkerCrashed):
        await w.call("die")
    assert not w.is_loaded
    assert await w.call("echo", 5) == 5
    await w.stop()


@pytest.mark.asyncio
async def test_cancelled_call_stops_the_busy_worker():
    w = worker()
    await w.call("pid")
    task = asyncio.create_task(w.call("slow", 30))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not w.is_loaded
    assert await w.call("echo", "ok") == "ok"
    await w.stop()


@pytest.mark.asyncio
async def test_failed_start_raises_worker_error():
    w = worker("worker_handlers:FailsToStart")
    with pytest.raises(WorkerError) as info:
        await w.call("anything")
    assert "no model file" in str(info.value) and not w.is_loaded


@pytest.mark.asyncio
async def test_unpicklable_result_is_an_error_not_a_hang():
    w = worker()
    with pytest.raises(WorkerError):
        await asyncio.wait_for(w.call("unpicklable"), 20)
    await w.stop()


@pytest.mark.asyncio
async def test_calls_are_serialized():
    w = worker()
    results = await asyncio.gather(w.call("slow", 0.3), w.call("echo", 1))
    assert results == ["done", 1]
    await w.stop()


def test_is_gpu_oom_walks_the_chain():
    try:
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        except RuntimeError as inner:
            raise ValueError("transcription failed") from inner
    except ValueError as outer:
        assert is_gpu_oom(outer)
    assert not is_gpu_oom(ValueError("decoder exploded"))
