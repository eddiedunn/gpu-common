"""Redis-based GPU lock for exclusive access coordination."""

import redis.asyncio as redis
from contextlib import asynccontextmanager


class GPULock:
    """Manages exclusive GPU access via Redis distributed lock."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize GPU lock.

        Args:
            redis_url: Redis connection URL
        """
        self._redis = redis.from_url(redis_url)
        self._lock_name = "gpu:exclusive"

    async def connect(self):
        """Verify Redis connection."""
        await self._redis.ping()

    async def disconnect(self):
        """Close Redis connection."""
        await self._redis.aclose()

    @asynccontextmanager
    async def acquire(self, timeout: int = 1800, blocking_timeout: int = 300):
        """Acquire exclusive GPU access.

        Args:
            timeout: Lock auto-expires after this many seconds (prevents deadlock)
            blocking_timeout: Max seconds to wait for lock acquisition

        Raises:
            TimeoutError: If lock cannot be acquired within blocking_timeout

        Yields:
            None: Lock is held for the duration of the context
        """
        # CORRECT API: blocking_timeout is passed to lock(), not acquire()
        lock = self._redis.lock(
            self._lock_name,
            timeout=timeout,
            blocking_timeout=blocking_timeout
        )
        acquired = await lock.acquire(blocking=True)
        if not acquired:
            raise TimeoutError(f"Could not acquire GPU lock after {blocking_timeout}s")
        try:
            yield
        finally:
            try:
                await lock.release()
            except redis.exceptions.LockNotOwnedError:
                pass  # Lock expired, already released
