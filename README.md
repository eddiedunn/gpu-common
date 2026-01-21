# GPU Common

Shared GPU utilities for AI services - infrastructure for coordinating exclusive GPU access across multiple services.

## Features

- **GPULock**: Redis-based distributed lock for exclusive GPU access
- **GPUModelManager**: Generic model manager with lazy loading and idle timeout cleanup
- **Health Checks**: GPU availability and memory monitoring utilities
- **Configuration**: Shared settings with environment variable support

## Installation

```bash
pip install ./dist/gpu_common-0.1.0-py3-none-any.whl
```

## Usage

### GPULock

```python
from gpu_common import GPULock

lock = GPULock(redis_url="redis://localhost:6379")
await lock.connect()

async with lock.acquire(timeout=3600, blocking_timeout=300):
    # Exclusive GPU access for model operations
    result = await model.process(data)

await lock.disconnect()
```

### GPUModelManager

```python
from gpu_common import GPUModelManager

async def load_my_model():
    # Load your model here
    return MyModel()

manager = GPUModelManager(load_fn=load_my_model, idle_timeout=300)
await manager.start()

# Model loads lazily on first use
model = await manager.get_model()
result = model.process(data)

await manager.stop()  # Cleanup
```

### Health Checks

```python
from gpu_common import check_gpu_available, get_gpu_memory_info

if check_gpu_available():
    memory_info = get_gpu_memory_info()
    print(f"GPU Memory: {memory_info['allocated_mb']} MB / {memory_info['total_mb']} MB")
```

## Development

Built for use by:
- Embed service (sentence transformers)
- Transcribe service (Whisper)
- LLM service (local models)
- Future GPU-dependent services

## License

Internal use only.
