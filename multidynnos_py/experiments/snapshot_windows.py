"""Temporal window utilities for snapshot experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SnapshotWindow:
    """A uniformly sampled temporal window."""

    index: int
    start: float
    end: float
    include_end: bool = False

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def contains(self, time: float) -> bool:
        if self.include_end:
            return self.start <= time <= self.end
        return self.start <= time < self.end

    def overlaps_interval(self, start: float, end: float) -> bool:
        if start == end:
            return self.contains(start)
        if self.include_end:
            return start <= self.end and end > self.start
        return start < self.end and end > self.start


def compute_snapshot_windows(
    sample_times: Sequence[float],
    n_snapshots: int,
) -> list[SnapshotWindow]:
    """Split the full layout time range uniformly into ``n_snapshots`` windows."""

    if n_snapshots <= 0:
        raise ValueError("n_snapshots must be positive.")
    if not sample_times:
        raise ValueError("sample_times cannot be empty.")

    start_time = min(float(time) for time in sample_times)
    end_time = max(float(time) for time in sample_times)
    if end_time < start_time:
        raise ValueError("sample_times produced an invalid time range.")

    boundaries = np.linspace(start_time, end_time, n_snapshots + 1)
    return [
        SnapshotWindow(
            index=index,
            start=float(boundaries[index]),
            end=float(boundaries[index + 1]),
            include_end=index == n_snapshots - 1,
        )
        for index in range(n_snapshots)
    ]
