from __future__ import annotations

import asyncio
import functools
import random
from typing import Callable, Type, Tuple


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Exponential backoff with jitter for async functions."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
