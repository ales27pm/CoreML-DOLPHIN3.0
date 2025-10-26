from __future__ import annotations

from pathlib import Path
from typing import Sequence

import csv
import pytest

from tasks.core_algorithmic_foundations.knapsack_optimizer import (
    AlgorithmProfile,
    KnapsackInputError,
    knapsack_bottom_up,
    knapsack_top_down,
    profile_algorithms,
    write_profiles_to_csv,
)


DATASET = {
    "capacity": 50,
    "weights": [3, 4, 7, 8, 9, 11, 13, 15, 19, 21],
    "values": [4, 5, 10, 11, 13, 17, 19, 23, 29, 31],
}


def test_knapsack_variants_produce_identical_result() -> None:
    result_top = knapsack_top_down(DATASET["capacity"], DATASET["weights"], DATASET["values"])
    result_bottom = knapsack_bottom_up(
        DATASET["capacity"], DATASET["weights"], DATASET["values"]
    )
    assert result_top == result_bottom == 75


def test_knapsack_rejects_invalid_inputs() -> None:
    with pytest.raises(KnapsackInputError):
        knapsack_top_down(-1, [1], [1])
    with pytest.raises(KnapsackInputError):
        knapsack_bottom_up(0, [1], [1, 2])
    with pytest.raises(KnapsackInputError):
        knapsack_top_down(0, [True], [1])  # type: ignore[list-item]
    with pytest.raises(KnapsackInputError):
        knapsack_bottom_up(0, [1], [-1])


def test_profile_algorithms_enforces_memory_ratio() -> None:
    top_profile, bottom_profile = profile_algorithms(
        DATASET["capacity"], DATASET["weights"], DATASET["values"]
    )
    assert top_profile.result == bottom_profile.result
    assert bottom_profile.peak_bytes <= int(top_profile.peak_bytes * 0.70) or top_profile.peak_bytes == 0


def test_write_profiles_to_csv(tmp_path: Path) -> None:
    target = tmp_path / "profiles.csv"
    profiles = (
        AlgorithmProfile(name="top_down", result=1, time_seconds=0.123456789, peak_bytes=1000),
        AlgorithmProfile(name="bottom_up", result=1, time_seconds=0.023456789, peak_bytes=600),
    )
    write_profiles_to_csv(target, profiles)

    with target.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    assert rows[0] == ["algorithm", "result", "time_seconds", "peak_bytes"]
    assert rows[1] == ["top_down", "1", "0.123456789", "1000"]
    assert rows[2] == ["bottom_up", "1", "0.023456789", "600"]


@pytest.mark.parametrize(
    "capacity,weights,values",
    [
        (0, [], []),
        (10, [1, 2, 3], [4, 5, 6]),
        (10, [3, 4, 7], [4, 5, 10]),
    ],
)
def test_profile_algorithms_handles_various_inputs(
    capacity: int, weights: Sequence[int], values: Sequence[int]
) -> None:
    top_profile, bottom_profile = profile_algorithms(
        capacity, weights, values, enforce_memory_ratio=False
    )
    assert top_profile.result == bottom_profile.result
    assert top_profile.peak_bytes >= 0
    assert bottom_profile.peak_bytes >= 0
