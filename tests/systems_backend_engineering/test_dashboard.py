from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from tasks.systems_backend_engineering.dashboard import (
    DEFAULT_QUERIES,
    HTTPStatusError,
    HTTPTimeoutError,
    MetricFetchError,
    fetch_metric,
    summarize_dashboard,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPStatusError(self.status_code, f"status: {self.status_code}")


class _StubSession:
    def __init__(self, responses: deque[Any]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> Any:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._responses:
            raise RuntimeError("No stubbed response available")
        result = self._responses.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


def test_fetch_metric_success_returns_float() -> None:
    payload = {"data": {"result": [{"metric": {}, "value": ["1730000000", "123.4"]}]}}
    session = _StubSession(deque([_FakeResponse(payload)]))
    value = fetch_metric("demo_query", session=session, url="http://example")
    assert pytest.approx(value) == 123.4
    assert session.calls[0]["params"]["query"] == "demo_query"
    assert not session.closed


def test_fetch_metric_empty_result_returns_zero() -> None:
    payload = {"data": {"result": []}}
    session = _StubSession(deque([_FakeResponse(payload)]))
    value = fetch_metric("missing", session=session, url="http://example")
    assert value == 0.0


def test_fetch_metric_raises_on_http_error() -> None:
    session = _StubSession(deque([_FakeResponse({}, status_code=500)]))
    with pytest.raises(MetricFetchError):
        fetch_metric("bad", session=session)


def test_summarize_dashboard_fetches_all_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = deque(
        [
            _FakeResponse({"data": {"result": [{"value": ["0", "1"]}]}}),
            _FakeResponse({"data": {"result": [{"value": ["0", "2"]}]}}),
            _FakeResponse({"data": {"result": [{"value": ["0", "3"]}]}}),
        ]
    )
    session = _StubSession(payloads)
    summary = summarize_dashboard(DEFAULT_QUERIES, session=session, url="http://example")
    assert summary == {
        "http_requests_per_second": 1.0,
        "latency_p99": 2.0,
        "error_rate": 3.0,
    }
    assert len(session.calls) == len(DEFAULT_QUERIES)


def test_summarize_dashboard_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _StubSession(deque([HTTPTimeoutError("timeout")]))
    with pytest.raises(MetricFetchError):
        summarize_dashboard({"latency": "demo"}, session=session)
