"""Finite temporal graphs and piecewise linear space-time trajectories.

Port of the relevant Ocotillo Interval/Evolution semantics. See PORTING.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    left_closed: bool = True
    right_closed: bool = True

    def __post_init__(self):
        if not (isfinite(self.start) and isfinite(self.end)):
            raise ValueError("Time bounds must be finite")
        if self.end < self.start:
            raise ValueError("Interval end precedes start")
        if self.start == self.end and not (self.left_closed and self.right_closed):
            raise ValueError("A point interval must be closed at both ends")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, time: float) -> bool:
        return ((time > self.start or (time == self.start and self.left_closed))
                and (time < self.end or (time == self.end and self.right_closed)))

    def intersection(self, other: Interval) -> Interval | None:
        start, end = max(self.start, other.start), min(self.end, other.end)
        if end < start:
            return None
        left = self.contains(start) and other.contains(start)
        right = self.contains(end) and other.contains(end)
        if end == start and not (left and right):
            return None
        return Interval(start, end, left, right)

    def __str__(self) -> str:
        return (f"{'[' if self.left_closed else '('}{float(self.start)},"
                f"{float(self.end)}{']' if self.right_closed else ')'}")


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Union without filling gaps or missing open boundary points."""
    result: list[Interval] = []
    for interval in sorted(intervals, key=lambda i: (i.start, not i.left_closed, i.end)):
        if not result:
            result.append(interval)
            continue
        previous = result[-1]
        joins = interval.start < previous.end or (
            interval.start == previous.end and (previous.right_closed or interval.left_closed))
        if not joins:
            result.append(interval)
            continue
        end = max(previous.end, interval.end)
        right = ((previous.end == end and previous.right_closed)
                 or (interval.end == end and interval.right_closed))
        result[-1] = Interval(previous.start, end, previous.left_closed, right)
    return result


def intersect_intervals(first: Iterable[Interval], second: Iterable[Interval]) -> list[Interval]:
    """Intersection of two sets of presence intervals."""
    left, right = merge_intervals(first), merge_intervals(second)
    result = []
    a = b = 0
    while a < len(left) and b < len(right):
        overlap = left[a].intersection(right[b])
        if overlap is not None:
            result.append(overlap)
        if left[a].end < right[b].end:
            a += 1
        elif right[b].end < left[a].end:
            b += 1
        else:
            # Equal endpoints can also intersect the following open/closed run.
            if a + 1 < len(left) and left[a + 1].start == left[a].end:
                overlap = left[a + 1].intersection(right[b])
                if overlap is not None:
                    result.append(overlap)
            if b + 1 < len(right) and right[b + 1].start == right[b].end:
                overlap = left[a].intersection(right[b + 1])
                if overlap is not None:
                    result.append(overlap)
            a += 1
            b += 1
    return merge_intervals(result)


@dataclass
class PositionSegment:
    interval: Interval
    start: tuple[float, float]
    end: tuple[float, float]

    def at(self, time: float) -> np.ndarray:
        ratio = ((time - self.interval.start) / self.interval.duration
                 if self.interval.duration else 0.0)
        return np.asarray(self.start, dtype=float) * (1.0 - ratio) + np.asarray(self.end, dtype=float) * ratio


@dataclass
class Node:
    id: str
    presence: list[Interval] = field(default_factory=list)
    positions: list[PositionSegment] = field(default_factory=list)
    label: str | None = None


@dataclass
class Edge:
    id: str
    source: str
    target: str
    presence: list[Interval] = field(default_factory=list)


@dataclass
class TemporalGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    data_type: str = "event"
    directed: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def interval(self) -> Interval:
        intervals = [i for n in self.nodes.values() for i in n.presence]
        if not intervals:
            raise ValueError("The graph has no node presence")
        return Interval(min(i.start for i in intervals), max(i.end for i in intervals))

    def validate(self) -> None:
        """Require edges to exist only while both endpoint nodes exist."""
        edge_ids = set()
        for node_id, node in self.nodes.items():
            if node_id != node.id:
                raise ValueError(f"Node dictionary key differs from id: {node_id}")
            node.presence = merge_intervals(node.presence)
            for segment in node.positions:
                if len(segment.start) != 2 or len(segment.end) != 2:
                    raise ValueError(f"Position of {node.id} must contain two spatial coordinates")
                if not all(isfinite(v) for v in (*segment.start, *segment.end)):
                    raise ValueError(f"Non-finite position for node {node.id}")
        for edge in self.edges:
            if edge.id in edge_ids:
                raise ValueError(f"Duplicate edge id: {edge.id}")
            edge_ids.add(edge.id)
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"Unknown endpoint for edge {edge.id}")
            edge.presence = merge_intervals(edge.presence)
            endpoints = intersect_intervals(self.nodes[edge.source].presence,
                                            self.nodes[edge.target].presence)
            if intersect_intervals(edge.presence, endpoints) != edge.presence:
                raise ValueError(f"Edge {edge.id} exists outside its endpoint node presence")


def position_at(node: Node, time: float) -> np.ndarray:
    """Linear interpolation; clamp to the nearest segment outside its domain.

    Presence remains separate: callers must not interpret extrapolated positions
    as evidence that the node exists. A fresh node starts at the origin.
    """
    if not node.positions:
        return np.zeros(2)
    for segment in node.positions:
        if segment.interval.contains(time):
            return segment.at(time)
    nearest = min(node.positions, key=lambda s: min(abs(time - s.interval.start), abs(time - s.interval.end)))
    return nearest.at(min(max(time, nearest.interval.start), nearest.interval.end))
