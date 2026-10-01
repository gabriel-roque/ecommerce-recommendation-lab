"""Leak-free chronological benchmark split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    train: Any
    validation: Any
    test: Any


def _best_boundary(cumulative: np.ndarray, target: float, choices: np.ndarray) -> int:
    distances = np.abs(cumulative[choices - 1] - target)
    return int(choices[int(np.argmin(distances))])


def temporal_split_indices(
    timestamps: Sequence[int] | np.ndarray,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chronological indices while keeping equal timestamps together."""
    values = np.asarray(timestamps)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size < 3:
        raise ValueError("Temporal split requires at least three events")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test set")

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    _, counts = np.unique(sorted_values, return_counts=True)
    if counts.size < 3:
        raise ValueError("Temporal split requires at least three distinct timestamps")
    cumulative = np.cumsum(counts)
    first_choices = np.arange(1, counts.size - 1)
    first_group = _best_boundary(cumulative, values.size * train_fraction, first_choices)
    second_choices = np.arange(first_group + 1, counts.size)
    second_group = _best_boundary(
        cumulative,
        values.size * (train_fraction + validation_fraction),
        second_choices,
    )
    train_end = int(cumulative[first_group - 1])
    validation_end = int(cumulative[second_group - 1])
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def _timestamps(events: Any, column: str) -> np.ndarray:
    try:
        values = events[column]
    except (KeyError, TypeError, IndexError):
        values = [row[column] for row in events]
    return np.asarray(values)


def _take(events: Any, indices: np.ndarray) -> Any:
    if hasattr(events, "iloc"):
        return events.iloc[indices].reset_index(drop=True)
    if hasattr(events, "take"):
        return events.take(indices)
    if isinstance(events, np.ndarray):
        return events[indices]
    return [events[int(index)] for index in indices]


def temporal_split(
    events: Any,
    *,
    timestamp_column: str = "timestamp",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> TemporalSplit:
    train, validation, test = temporal_split_indices(
        _timestamps(events, timestamp_column), train_fraction, validation_fraction
    )
    result = TemporalSplit(_take(events, train), _take(events, validation), _take(events, test))
    train_times = _timestamps(result.train, timestamp_column)
    validation_times = _timestamps(result.validation, timestamp_column)
    test_times = _timestamps(result.test, timestamp_column)
    if not (
        train_times.max() < validation_times.min()
        and validation_times.max() < test_times.min()
    ):
        raise AssertionError("Temporal split leaked equal or newer timestamps")
    return result
