"""Tests for GPUModelManager."""

import pytest
import asyncio
from gpu_common import GPUModelManager


class DummyModel:
    """Dummy model for testing."""
    def __init__(self):
        self.loaded = True


@pytest.mark.asyncio
async def test_model_manager_lazy_load():
    """Test that model loads lazily on first access."""
    loaded = False

    async def load_fn():
        nonlocal loaded
        loaded = True
        return DummyModel()

    manager = GPUModelManager(load_fn=load_fn, idle_timeout=300)
    assert not loaded

    model = await manager.get_model()
    assert loaded
    assert isinstance(model, DummyModel)

    await manager.stop()


@pytest.mark.asyncio
async def test_model_manager_idle_cleanup():
    """Test that model unloads after idle timeout."""
    async def load_fn():
        return DummyModel()

    manager = GPUModelManager(load_fn=load_fn, idle_timeout=2)
    await manager.start()

    # Load model
    model = await manager.get_model()
    assert model is not None

    # Wait for idle timeout
    await asyncio.sleep(3)

    # Model should be unloaded
    assert manager._model is None

    await manager.stop()


@pytest.mark.asyncio
async def test_model_manager_explicit_unload():
    """Test explicit model unloading."""
    async def load_fn():
        return DummyModel()

    manager = GPUModelManager(load_fn=load_fn, idle_timeout=300)

    model = await manager.get_model()
    assert manager._model is not None

    await manager.unload()
    assert manager._model is None

    await manager.stop()
