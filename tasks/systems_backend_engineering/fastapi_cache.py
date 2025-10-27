"""FastAPI response caching decorator backed by Redis."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
)

from redis.asyncio import Redis

__all__ = ["cache", "CacheBackendProtocol"]

logger = logging.getLogger("fastapi.cache")

P = ParamSpec("P")
R = TypeVar("R")


class CacheBackendProtocol(Protocol):
    """Minimal protocol satisfied by :class:`redis.asyncio.Redis`."""

    async def get(self, key: str) -> Optional[bytes]:
        """Return a cached payload for *key* or ``None`` when missing."""

    async def set(self, key: str, value: bytes, ex: int) -> Any:
        """Persist *value* for *ex* seconds."""


class _RedisFactory:
    def __init__(self, factory: Optional[Callable[[], Redis]] = None) -> None:
        self._factory = factory
        self._client: Optional[Redis] = None
        self._lock = asyncio.Lock()

    async def get(self) -> Redis:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                if self._factory is not None:
                    self._client = self._factory()
                else:
                    self._client = Redis.from_url(
                        "redis://localhost", encoding="utf-8", decode_responses=False
                    )
            return self._client


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "dict"):
        return value.dict()  # type: ignore[no-any-return]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def _coerce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("Redis backend returned non-bytes payload")


def cache(
    ttl: int = 60,
    *,
    namespace: str = "fastapi-cache",
    redis_factory: Optional[Callable[[], Redis]] = None,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]],
    Callable[P, Awaitable[R]],
]:
    """Cache decorator for FastAPI endpoints.

    Parameters
    ----------
    ttl:
        Number of seconds to retain cached responses. Must be positive.
    namespace:
        Prefix applied to generated cache keys. Enables multi-tenant isolation.
    redis_factory:
        Optional callable returning a :class:`redis.asyncio.Redis` instance. When
        omitted a singleton client targeting ``redis://localhost`` is created on
        demand.
    """

    if ttl <= 0:
        raise ValueError("ttl must be a positive integer")

    redis_singleton = _RedisFactory(redis_factory)

    def decorator(func: Callable[P, Coroutine[Any, Any, R]]):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("cache decorator requires an async function")

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            redis_client = await redis_singleton.get()
            key_payload = json.dumps(
                [func.__module__, func.__qualname__, args, kwargs],
                sort_keys=True,
                default=_json_default,
            )
            cache_key = f"{namespace}:{key_payload}"
            cached = await redis_client.get(cache_key)
            if cached is not None:
                logger.info("cache.hit", extra={"key": cache_key, "ttl": ttl})
                return json.loads(_coerce_bytes(cached))

            result = await func(*args, **kwargs)
            serialised = json.dumps(result, default=_json_default).encode("utf-8")
            await redis_client.set(cache_key, serialised, ex=ttl)
            logger.info("cache.miss", extra={"key": cache_key, "ttl": ttl})
            return result

        return wrapper

    return decorator
