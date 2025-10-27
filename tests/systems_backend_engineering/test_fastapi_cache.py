from __future__ import annotations

import json
from typing import Dict

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tasks.systems_backend_engineering.fastapi_cache import cache


@pytest.mark.asyncio
async def test_cache_decorator_hits_and_misses(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="fastapi.cache")
    fake_redis = fakeredis.aioredis.FakeRedis()
    app = FastAPI()
    counter: Dict[str, int] = {"calls": 0}

    @app.get("/items/{item_id}")
    @cache(ttl=30, namespace="tests", redis_factory=lambda: fake_redis)
    async def read_item(item_id: int) -> Dict[str, int]:
        counter["calls"] += 1
        return {"item_id": item_id, "value": item_id * 2}

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        first = await client.get("/items/5")
        second = await client.get("/items/5")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json() == {"item_id": 5, "value": 10}
    assert counter["calls"] == 1

    hits = [rec for rec in caplog.records if rec.message == "cache.hit"]
    misses = [rec for rec in caplog.records if rec.message == "cache.miss"]
    assert hits and misses

    keys = await fake_redis.keys("tests:*")
    assert len(keys) == 1
    ttl = await fake_redis.ttl(keys[0])
    assert ttl > 0


@pytest.mark.asyncio
async def test_cache_rejects_non_async_functions() -> None:
    with pytest.raises(TypeError):
        cache()(lambda: None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cache_handles_unserialisable_results() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()

    @cache(redis_factory=lambda: fake_redis)
    async def endpoint() -> object:
        return {"set": {1, 2, 3}}

    result = await endpoint()
    keys = await fake_redis.keys("fastapi-cache:*")
    assert len(keys) == 1
    raw = await fake_redis.get(keys[0])
    assert result == {"set": {1, 2, 3}}
    assert raw is not None
    assert json.loads(raw) == {"set": [1, 2, 3]}
