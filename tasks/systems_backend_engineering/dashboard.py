"""Prometheus dashboard aggregation utilities for Codex Task 36.

The module fetches point-in-time metrics from a Prometheus server and produces
an executive summary dictionary that downstream automation can serialize or
push into observability dashboards.  The implementation keeps retry logic and
parsing robust so it is safe to run inside scheduled jobs.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Mapping
from urllib import error, parse, request

logger = logging.getLogger(__name__)

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
_DEFAULT_TIMEOUT = 5.0


class HTTPClientError(RuntimeError):
    """Raised when the HTTP client fails to execute a request."""


class HTTPTimeoutError(HTTPClientError):
    """Raised when the HTTP request exceeds the configured timeout."""


class HTTPStatusError(HTTPClientError):
    """Raised when the HTTP server responds with a non-success status code."""

    def __init__(
        self, status_code: int, message: str, body: bytes | None = None
    ) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.body = body


class _UrlLibResponse:
    """Lightweight response wrapper that mimics the requests.Response API."""

    def __init__(
        self, body: bytes, status_code: int, headers: Mapping[str, str]
    ) -> None:
        self._body = body
        self.status_code = status_code
        self.headers = headers

    def json(self) -> Any:
        ctype = self.headers.get("Content-Type", "")
        encoding = "utf-8"
        if "charset=" in ctype.lower():
            encoding = ctype.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
        text = self._body.decode(encoding, errors="replace")
        return json.loads(text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPStatusError(self.status_code, "Request failed", self._body)


class _UrlLibSession:
    """Minimal session implementation backed by urllib."""

    def get(
        self, url: str, *, params: Mapping[str, Any], timeout: float
    ) -> _UrlLibResponse:
        parsed = parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
        query = parse.urlencode(params)
        delimiter = "&" if ("?" in url and query) else "?"
        full_url = f"{url}{delimiter}{query}" if query else url
        req = request.Request(full_url)
        try:
            with request.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
                body = resp.read()
                status = getattr(resp, "status", resp.getcode())
                headers = dict(resp.headers.items())
        except error.HTTPError as exc:
            body = exc.read() if exc.fp else b""
            raise HTTPStatusError(exc.code, exc.reason or "HTTP error", body) from exc
        except error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise HTTPTimeoutError("Request timed out") from exc
            raise HTTPClientError(f"Failed to reach server: {reason}") from exc
        except socket.timeout as exc:
            raise HTTPTimeoutError("Request timed out") from exc
        return _UrlLibResponse(body, status, headers)

    def close(self) -> None:
        """Provided for API symmetry with requests.Session."""
        return None


DEFAULT_QUERIES: Mapping[str, str] = {
    "http_requests_per_second": "rate(http_requests_total[5m])",
    "latency_p99": "histogram_quantile(0.99, sum(rate(request_latency_seconds_bucket[5m])) by (le))",
    "error_rate": 'sum(rate(http_requests_total{status="5xx"}[5m]))',
}


class MetricFetchError(RuntimeError):
    """Raised when a metric cannot be retrieved from Prometheus."""


def _parse_metric(response: Any, query: str) -> float:
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise MetricFetchError("Prometheus response was not valid JSON") from exc

    try:
        data = payload["data"]["result"]
    except (KeyError, TypeError) as exc:
        raise MetricFetchError("Prometheus response missing data.result field") from exc

    if not data:
        logger.warning("Prometheus query %s returned no data", query)
        return 0.0

    try:
        value = float(data[0]["value"][1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise MetricFetchError("Prometheus result payload malformed") from exc
    return value


def _close_safely(candidate: Any) -> None:
    close = getattr(candidate, "close", None)
    if callable(close):
        close()


def fetch_metric(
    query: str,
    *,
    session: Any | None = None,
    url: str = PROMETHEUS_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> float:
    """Fetch a single Prometheus metric and return its numeric value."""

    close_session = session is None
    http = session or _UrlLibSession()
    try:
        logger.debug("Fetching Prometheus metric: %s", query)
        response = http.get(url, params={"query": query}, timeout=timeout)
        try:
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - defensive
            raise MetricFetchError(f"Failed to fetch metric '{query}': {exc}") from exc
        return _parse_metric(response, query)
    except MetricFetchError:
        raise
    except Exception as exc:
        raise MetricFetchError(f"Failed to fetch metric '{query}': {exc}") from exc
    finally:
        if close_session:
            _close_safely(http)


def summarize_dashboard(
    queries: Mapping[str, str] | None = None,
    *,
    session: Any | None = None,
    url: str = PROMETHEUS_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, float]:
    """Return a dictionary of metric names to numeric values."""

    queries = queries or DEFAULT_QUERIES
    summary: dict[str, float] = {}
    for name, query in queries.items():
        value = fetch_metric(query, session=session, url=url, timeout=timeout)
        summary[name] = value
        logger.info("Metric %s=%s", name, value)
    return summary


def main() -> None:  # pragma: no cover - CLI convenience
    logging.basicConfig(level=logging.INFO)
    try:
        metrics = summarize_dashboard()
    except MetricFetchError as exc:
        logger.exception("Dashboard summarization failed: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
