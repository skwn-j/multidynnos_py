"""Temporal primitives used by the dynamic graph data model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True, order=True)
class Interval:
    """A time interval with configurable open or closed boundaries."""

    start: float
    end: float
    left_closed: bool = True
    right_closed: bool = True

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Interval end {self.end} is before start {self.start}.")
        if self.end == self.start and not (self.left_closed and self.right_closed):
            raise ValueError("Zero-width intervals must be closed on both sides.")

    @classmethod
    def closed(cls, start: float, end: float) -> Interval:
        return cls(start=start, end=end, left_closed=True, right_closed=True)

    @classmethod
    def left_open_right_closed(cls, start: float, end: float) -> Interval:
        return cls(start=start, end=end, left_closed=False, right_closed=True)

    @classmethod
    def open(cls, start: float, end: float) -> Interval:
        return cls(start=start, end=end, left_closed=False, right_closed=False)

    @classmethod
    def from_start_duration(cls, start: float, duration: float) -> Interval:
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}.")
        return cls.closed(start, start + duration)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, time: float) -> bool:
        if self.start < time < self.end:
            return True
        if time == self.start:
            return self.left_closed
        if time == self.end:
            return self.right_closed
        return False

    def overlaps(self, other: Interval) -> bool:
        if self.end < other.start or other.end < self.start:
            return False
        if self.end == other.start:
            return self.right_closed and other.left_closed
        if other.end == self.start:
            return other.right_closed and self.left_closed
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "left_closed": self.left_closed,
            "right_closed": self.right_closed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interval:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            left_closed=bool(data.get("left_closed", True)),
            right_closed=bool(data.get("right_closed", True)),
        )

    def as_start_duration(self) -> tuple[float, float]:
        if not isfinite(self.duration):
            raise ValueError("Cannot export infinite intervals as start-duration rows.")
        return self.start, self.duration


def presence_at(intervals: list[Interval], time: float) -> bool:
    """Return true when any interval contains the supplied time."""

    return any(interval.contains(time) for interval in intervals)
