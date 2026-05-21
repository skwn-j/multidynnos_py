"""Aspect-ratio utilities for dynamic graph layout stimuli."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Protocol

from multidynnos_py.experiments.snapshots import LayoutSnapshot
from multidynnos_py.layout.dynnoslice import DynamicLayout, TrajectoryPoint


class _PointLike(Protocol):
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box for a snapshot or full dynamic layout."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "width": self.width,
            "height": self.height,
        }


def compute_bounding_box(snapshot_or_layout: LayoutSnapshot | DynamicLayout) -> BoundingBox:
    """Compute an axis-aligned bounding box for a snapshot or whole layout."""

    points = list(_iter_points(snapshot_or_layout))
    if not points:
        return BoundingBox(0.0, 0.0, 0.0, 0.0)
    return BoundingBox(
        min_x=min(point.x for point in points),
        min_y=min(point.y for point in points),
        max_x=max(point.x for point in points),
        max_y=max(point.y for point in points),
    )


def compute_aspect_ratio(snapshot_or_layout: LayoutSnapshot | DynamicLayout) -> float:
    """Compute width / height for a snapshot or whole layout."""

    box = compute_bounding_box(snapshot_or_layout)
    if box.height == 0.0:
        return 1.0 if box.width == 0.0 else inf
    return box.width / box.height


def affine_scale_layout(layout: DynamicLayout, target_aspect_ratio: float) -> DynamicLayout:
    """Return a copy of a layout globally scaled to the target aspect ratio."""

    if target_aspect_ratio <= 0:
        raise ValueError("target_aspect_ratio must be positive.")

    box = compute_bounding_box(layout)
    if box.width == 0.0 or box.height == 0.0:
        return _copy_layout(layout)

    current_aspect_ratio = box.width / box.height
    scale_x = 1.0
    scale_y = 1.0
    if current_aspect_ratio < target_aspect_ratio:
        scale_x = target_aspect_ratio / current_aspect_ratio
    else:
        scale_y = current_aspect_ratio / target_aspect_ratio

    center_x, center_y = box.center
    scaled_trajectories: dict[str, list[TrajectoryPoint]] = {}
    for node_id, points in layout.trajectories.items():
        scaled_trajectories[node_id] = [
            TrajectoryPoint(
                time=point.time,
                x=center_x + (point.x - center_x) * scale_x,
                y=center_y + (point.y - center_y) * scale_y,
            )
            for point in points
        ]

    return DynamicLayout(
        trajectories=scaled_trajectories,
        sample_times=list(layout.sample_times),
        config=layout.config,
    )


def _copy_layout(layout: DynamicLayout) -> DynamicLayout:
    return DynamicLayout(
        trajectories={node_id: list(points) for node_id, points in layout.trajectories.items()},
        sample_times=list(layout.sample_times),
        config=layout.config,
    )


def _iter_points(snapshot_or_layout: LayoutSnapshot | DynamicLayout) -> Iterable[_PointLike]:
    if isinstance(snapshot_or_layout, LayoutSnapshot):
        return snapshot_or_layout.nodes.values()

    return (
        point
        for trajectory in snapshot_or_layout.trajectories.values()
        for point in trajectory
    )
