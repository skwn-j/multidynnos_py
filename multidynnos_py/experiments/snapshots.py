"""Snapshot sampling and small-multiple export utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from multidynnos_py.layout.dynnoslice import DynamicLayout, TrajectoryPoint


@dataclass(frozen=True, slots=True)
class SnapshotPoint:
    """A node position in a sampled layout snapshot."""

    node_id: str
    x: float
    y: float

    def to_dict(self) -> dict[str, float | str]:
        return {"id": self.node_id, "x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class LayoutSnapshot:
    """A dynamic layout sampled at one time."""

    time: float
    nodes: dict[str, SnapshotPoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "nodes": [
                self.nodes[node_id].to_dict()
                for node_id in sorted(self.nodes)
            ],
        }


def sample_snapshots(layout: DynamicLayout, times: Iterable[float]) -> list[LayoutSnapshot]:
    """Sample a dynamic layout at requested times.

    Node positions are linearly interpolated between adjacent trajectory
    samples. A node is omitted when the requested time falls outside that
    node's sampled trajectory.
    """

    snapshots: list[LayoutSnapshot] = []
    for time in times:
        node_positions: dict[str, SnapshotPoint] = {}
        for node_id, trajectory in layout.trajectories.items():
            point = _interpolate_trajectory(trajectory, float(time))
            if point is not None:
                node_positions[node_id] = SnapshotPoint(node_id=node_id, x=point[0], y=point[1])
        snapshots.append(LayoutSnapshot(time=float(time), nodes=node_positions))
    return snapshots


def export_small_multiples_data(layout: DynamicLayout, times: Iterable[float], out_path: str | Path) -> dict[str, Any]:
    """Export JSON data for flat small-multiple dynamic graph stimuli."""

    from multidynnos_py.experiments.aspect_ratio import compute_aspect_ratio, compute_bounding_box
    from multidynnos_py.metrics.layout_metrics import crowding_metric

    requested_times = [float(time) for time in times]
    snapshots = sample_snapshots(layout, requested_times)
    data: dict[str, Any] = {
        "times": requested_times,
        "layout_bounding_box": compute_bounding_box(layout).to_dict(),
        "layout_aspect_ratio": compute_aspect_ratio(layout),
        "snapshots": [],
    }
    for snapshot in snapshots:
        snapshot_data = snapshot.to_dict()
        snapshot_data["bounding_box"] = compute_bounding_box(snapshot).to_dict()
        snapshot_data["aspect_ratio"] = compute_aspect_ratio(snapshot)
        snapshot_data["crowding"] = crowding_metric(snapshot)
        data["snapshots"].append(snapshot_data)

    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _interpolate_trajectory(trajectory: list[TrajectoryPoint], time: float) -> tuple[float, float] | None:
    if not trajectory:
        return None

    ordered = sorted(trajectory, key=lambda point: point.time)
    if time < ordered[0].time or time > ordered[-1].time:
        return None

    for point in ordered:
        if point.time == time:
            return point.x, point.y

    for left, right in zip(ordered, ordered[1:]):
        if left.time <= time <= right.time:
            if right.time == left.time:
                return left.x, left.y
            alpha = (time - left.time) / (right.time - left.time)
            return (
                left.x + (right.x - left.x) * alpha,
                left.y + (right.y - left.y) * alpha,
            )
    return None
