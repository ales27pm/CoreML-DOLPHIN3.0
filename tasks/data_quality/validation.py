"""Data quality validation utilities for analytics pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Iterable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityViolation:
    """Represents a single data quality violation."""

    code: str
    column: str
    message: str
    row_indices: tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "column": self.column,
            "message": self.message,
            "row_indices": list(self.row_indices),
        }


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of the validation pass."""

    dataset_name: str
    record_count: int
    generated_at: datetime
    violations: tuple[DataQualityViolation, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "record_count": self.record_count,
            "generated_at": self.generated_at.isoformat(),
            "violations": [violation.to_dict() for violation in self.violations],
            "metadata": dict(self.metadata),
        }

    def summary(self) -> str:
        if self.is_valid:
            return f"Dataset '{self.dataset_name}' passed validation ({self.record_count} rows)."
        violations = ", ".join(violation.code for violation in self.violations)
        return (
            f"Dataset '{self.dataset_name}' failed validation with {len(self.violations)} "
            f"violations: {violations}"
        )


class DataQualityError(RuntimeError):
    """Raised when validation fails in strict mode."""

    def __init__(self, report: DataQualityReport) -> None:
        super().__init__(report.summary())
        self.report = report


def _coerce_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True)
    except Exception as exc:  # pragma: no cover - passthrough for pandas errors
        raise DataQualityError(
            DataQualityReport(
                dataset_name="unknown",
                record_count=0,
                generated_at=datetime.now(tz=timezone.utc),
                violations=(
                    DataQualityViolation(
                        code="invalid_datetime",
                        column=str(series.name or "updated_at"),
                        message=str(exc),
                        row_indices=tuple(),
                    ),
                ),
            )
        ) from exc


def validate_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    required_columns: Iterable[str] = ("price", "sku", "updated_at"),
    max_staleness: pd.Timedelta | None = None,
    strict: bool = True,
) -> DataQualityReport:
    """Validate a dataset and return a structured report.

    Parameters
    ----------
    df:
        Input dataframe expected to contain pricing data.
    dataset_name:
        Human readable identifier for reporting purposes.
    required_columns:
        Collection of column names that must exist in the dataframe.
    max_staleness:
        Maximum allowed interval between oldest and newest records.
    strict:
        When ``True`` the function raises :class:`DataQualityError` when violations are detected.

    Returns
    -------
    DataQualityReport
        Summary containing validation metadata and violations.
    """

    if max_staleness is None:
        max_staleness = pd.Timedelta(days=1)

    missing = [column for column in required_columns if column not in df.columns]
    violations: list[DataQualityViolation] = []

    if missing:
        violations.append(
            DataQualityViolation(
                code="missing_columns",
                column=",".join(missing),
                message=f"Missing required columns: {', '.join(missing)}",
            )
        )
    else:
        price_series = pd.to_numeric(df["price"], errors="coerce")
        invalid_price_mask = price_series.isna() & df["price"].notna()
        invalid_price_positions = tuple(idx for idx, flag in enumerate(invalid_price_mask) if flag)
        if invalid_price_positions:
            violations.append(
                DataQualityViolation(
                    code="invalid_price",
                    column="price",
                    message="Price values must be numeric",
                    row_indices=invalid_price_positions,
                )
            )

        negative_mask = price_series.lt(0, fill_value=False)
        negative_positions = tuple(idx for idx, flag in enumerate(negative_mask) if flag)
        if negative_positions:
            violations.append(
                DataQualityViolation(
                    code="negative_price",
                    column="price",
                    message="Detected negative pricing values",
                    row_indices=negative_positions,
                )
            )

        sku_series = df["sku"]
        sku_missing_mask = sku_series.isna() | (sku_series.astype(str).str.strip() == "")
        missing_positions = tuple(idx for idx, flag in enumerate(sku_missing_mask) if flag)
        if missing_positions:
            violations.append(
                DataQualityViolation(
                    code="missing_sku",
                    column="sku",
                    message="Found records with missing SKU",
                    row_indices=missing_positions,
                )
            )

        if len(df) > 0:
            timestamps = _coerce_datetime(df["updated_at"])
            staleness = timestamps.max() - timestamps.min()
            if pd.isna(staleness) or staleness > max_staleness:
                violations.append(
                    DataQualityViolation(
                        code="stale_records",
                        column="updated_at",
                        message=(
                            "Dataset contains stale records beyond allowed threshold "
                            f"({max_staleness})."
                        ),
                    )
                )

    report = DataQualityReport(
        dataset_name=dataset_name,
        record_count=len(df),
        generated_at=datetime.now(tz=timezone.utc),
        violations=tuple(violations),
        metadata={"max_staleness": str(max_staleness)},
    )

    if report.is_valid:
        logger.info("Data quality validation passed for %s", dataset_name)
    else:
        logger.warning(
            "Data quality validation failed for %s with %d violations",
            dataset_name,
            len(violations),
        )
        if strict:
            raise DataQualityError(report)

    return report
