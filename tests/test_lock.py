"""Tests for GPULock."""

import pytest
from gpu_common import GPULock


@pytest.mark.asyncio
async def test_lock_acquire_release():
    """Test basic lock acquisition and release."""
    lock = GPULock()
    await lock.connect()

    async with lock.acquire(timeout=10, blocking_timeout=5):
        # Lock is held
        pass

    await lock.disconnect()


@pytest.mark.asyncio
async def test_lock_timeout():
    """Test lock acquisition timeout."""
    lock1 = GPULock()
    lock2 = GPULock()

    await lock1.connect()
    await lock2.connect()

    async with lock1.acquire(timeout=10, blocking_timeout=1):
        # Try to acquire with another client - should timeout
        with pytest.raises(TimeoutError):
            async with lock2.acquire(timeout=10, blocking_timeout=1):
                pass

    await lock1.disconnect()
    await lock2.disconnect()
