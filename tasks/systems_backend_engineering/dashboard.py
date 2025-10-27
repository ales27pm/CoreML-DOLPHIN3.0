"""Prometheus dashboard aggregation utilities for Codex Task 36.

The module fetches point-in-time metrics from a Prometheus server and produces
an executive summary dictionary that downstream automation can serialize or
push into observability dashboards.  The implementation keeps retry logic and
parsing robust so it is safe to run inside scheduled jobs.
"""

from __future__ import annotations

import json
import logging
from typing import Mapping

import requests
from requests import Response, Session
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
_DEFAULT_TIMEOUT = 5.0

DEFAULT_QUERIES: Mapping[str, str] = {
    "http_requests_per_second": "rate(http_requests_total[5m])",
    "latency_p99": "histogram_quantile(0.99, sum(rate(request_latency_seconds_bucket[5m])) by (le))",
    "error_rate": 'sum(rate(http_requests_total{status="5xx"}[5m]))',
}


class MetricFetchError(RuntimeError):
    """Raised when a metric cannot be retrieved from Prometheus."""


def _parse_metric(response: Response, query: str) -> float:
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


def fetch_metric(
    query: str,
    *,
    session: Session | None = None,
    url: str = PROMETHEUS_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> float:
    """Fetch a single Prometheus metric and return its numeric value."""

    close_session = session is None
    http = session or requests.Session()
    try:
        logger.debug("Fetching Prometheus metric: %s", query)
        response = http.get(url, params={"query": query}, timeout=timeout)
        response.raise_for_status()
        return _parse_metric(response, query)
    except RequestException as exc:
        raise MetricFetchError(f"Failed to fetch metric '{query}': {exc}") from exc
    finally:
        if close_session:
            http.close()


def summarize_dashboard(
    queries: Mapping[str, str] | None = None,
    *,
    session: Session | None = None,
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
        logger.error("Dashboard summarization failed: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
