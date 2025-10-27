from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tasks.data_quality.validation import (
    DataQualityError,
    DataQualityReport,
    DataQualityViolation,
    validate_dataset,
)


def _base_dataframe() -> pd.DataFrame:
    now = datetime.now(tz=timezone.utc)
    return pd.DataFrame(
        {
            "price": [9.99, 12.5, 0.0],
            "sku": ["A-1", "B-2", "C-3"],
            "updated_at": [now, now - timedelta(minutes=5), now - timedelta(minutes=10)],
        }
    )


def test_validate_dataset_success() -> None:
    df = _base_dataframe()
    report = validate_dataset(df, dataset_name="catalogue")
    assert isinstance(report, DataQualityReport)
    assert report.is_valid
    assert report.record_count == len(df)


def test_validate_dataset_negative_price() -> None:
    df = _base_dataframe()
    df.loc[1, "price"] = -1
    with pytest.raises(DataQualityError) as exc:
        validate_dataset(df, dataset_name="catalogue")
    assert exc.value.report.violations[0].code == "negative_price"


def test_validate_dataset_missing_sku() -> None:
    df = _base_dataframe()
    df.loc[2, "sku"] = ""
    with pytest.raises(DataQualityError) as exc:
        validate_dataset(df)
    violation = exc.value.report.violations[0]
    assert violation.code == "missing_sku"
    assert 2 in violation.row_indices


def test_validate_dataset_staleness() -> None:
    df = _base_dataframe()
    df.loc[2, "updated_at"] = datetime.now(tz=timezone.utc) - timedelta(days=3)
    with pytest.raises(DataQualityError) as exc:
        validate_dataset(df)
    assert exc.value.report.violations[0].code == "stale_records"


def test_validate_dataset_non_strict_returns_report() -> None:
    df = _base_dataframe()
    df.loc[0, "price"] = -5
    report = validate_dataset(df, strict=False)
    assert isinstance(report, DataQualityReport)
    assert not report.is_valid
    assert any(isinstance(v, DataQualityViolation) for v in report.violations)
