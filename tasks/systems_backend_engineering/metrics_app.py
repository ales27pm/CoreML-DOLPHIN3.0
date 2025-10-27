"""FastAPI application exposing Prometheus-compatible latency metrics."""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

__all__ = ["app", "REQUEST_LATENCY", "instrument_route"]

app = FastAPI()
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency",
    labelnames=("method", "endpoint"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    return response


def instrument_route(path: str, handler: Callable[..., object]) -> None:
    """Register a JSON endpoint primarily used for testing instrumentation."""

    @app.get(path)
    async def _handler() -> JSONResponse:  # pragma: no cover - wrapper
        payload = handler()
        return JSONResponse(payload)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REQUEST_LATENCY), media_type=CONTENT_TYPE_LATEST)
