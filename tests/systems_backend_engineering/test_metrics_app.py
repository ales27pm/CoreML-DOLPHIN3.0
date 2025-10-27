from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from tasks.systems_backend_engineering.metrics_app import app


@pytest.mark.asyncio
async def test_metrics_endpoint_records_latency() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        for _ in range(3):
            response = await client.get("/health/live")
            assert response.status_code == 200
        metrics_response = await client.get("/metrics")

    assert metrics_response.status_code == 200
    body = metrics_response.text
    count_lines = [
        line
        for line in body.splitlines()
        if line.startswith("request_latency_seconds_count")
    ]
    assert any(
        'endpoint="/health/live"' in line and 'method="GET"' in line
        for line in count_lines
    )
    assert "request_latency_seconds_bucket" in body
