"""Tests for GPUModelManager."""

import asyncio
import gc
import weakref
from datetime import timedelta

import pytest

from gpu_common import GPUModelManager


class DummyModel:
    def __init__(self):
        self.released = False
        # A reference cycle: the model is only freed by the cycle collector,
        # which is what made the old unload leave GPU memory behind.
        self.me = self


def fast(**kw) -> GPUModelManager:
    return GPUModelManager(cleanup_interval=timedelta(seconds=0.05), **kw)


@pytest.mark.asyncio
async def test_loads_lazily_on_first_get():
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        return DummyModel()

    m = fast(load_fn=load)
    assert calls == 0 and not m.is_loaded
    a = await m.get()
    b = await m.get()
    assert a is b and calls == 1 and m.is_loaded
    await m.stop()


@pytest.mark.asyncio
async def test_unload_frees_model_even_inside_a_reference_cycle():
    m = fast(load_fn=DummyModel)
    ref = weakref.ref(await m.get())
    await m.unload()
    assert ref() is None, "model survived unload; gc.collect() must run before empty_cache()"
    assert not m.is_loaded


@pytest.mark.asyncio
async def test_release_fn_runs_before_drop():
    released = []
    m = fast(load_fn=DummyModel, release_fn=lambda model: released.append(model))
    model = await m.get()
    await m.unload()
    assert released == [model]


@pytest.mark.asyncio
async def test_failing_release_fn_still_drops_model():
    def boom(_):
        raise RuntimeError("release failed")

    m = fast(load_fn=DummyModel, release_fn=boom)
    ref = weakref.ref(await m.get())
    await m.unload()
    gc.collect()
    assert ref() is None and not m.is_loaded


@pytest.mark.asyncio
async def test_idle_model_is_unloaded_by_cleanup_loop():
    m = fast(load_fn=DummyModel, idle_timeout=0.1)
    await m.start()
    await m.get()
    await asyncio.sleep(0.4)
    assert not m.is_loaded
    await m.stop()


@pytest.mark.asyncio
async def test_cleanup_loop_survives_a_failed_unload():
    attempts = 0

    def flaky(_):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("release failed")

    m = fast(load_fn=DummyModel, idle_timeout=0.05, release_fn=flaky)
    await m.start()
    await m.get()
    await asyncio.sleep(0.3)
    await m.get()  # load again; the loop must still be alive to unload it
    await asyncio.sleep(0.3)
    assert attempts == 2 and not m.is_loaded
    await m.stop()


@pytest.mark.asyncio
async def test_idle_timeout_accepts_seconds_and_env(monkeypatch):
    assert fast(load_fn=DummyModel, idle_timeout=90).idle_timeout == timedelta(seconds=90)
    monkeypatch.setenv("GPU_IDLE_TIMEOUT", "1800")
    assert fast(load_fn=DummyModel).idle_timeout == timedelta(seconds=1800)


@pytest.mark.asyncio
async def test_warm_loads_in_background():
    m = fast(load_fn=DummyModel)
    task = m.warm()
    await task
    assert m.is_loaded
    await m.stop()


@pytest.mark.asyncio
async def test_stop_unloads():
    m = fast(load_fn=DummyModel)
    await m.start()
    await m.get()
    await m.stop()
    assert not m.is_loaded
