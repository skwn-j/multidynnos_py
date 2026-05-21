from __future__ import annotations

from math import inf, isclose

import pytest

from multidynnos_py.experiments.aspect_ratio import (
    affine_scale_layout,
    compute_aspect_ratio,
    compute_bounding_box,
)
from multidynnos_py.experiments.snapshots import LayoutSnapshot, SnapshotPoint
from multidynnos_py.layout.dynnoslice import DynNoSliceConfig, DynamicLayout, TrajectoryPoint


def _layout() -> DynamicLayout:
    return DynamicLayout(
        trajectories={
            "A": [TrajectoryPoint(time=0.0, x=0.0, y=0.0)],
            "B": [TrajectoryPoint(time=0.0, x=4.0, y=2.0)],
        },
        sample_times=[0.0],
        config=DynNoSliceConfig(iterations=0, seed=5),
    )


def test_compute_bounding_box_and_aspect_ratio_for_snapshot() -> None:
    snapshot = LayoutSnapshot(
        time=0.0,
        nodes={
            "A": SnapshotPoint(node_id="A", x=0.0, y=0.0),
            "B": SnapshotPoint(node_id="B", x=4.0, y=2.0),
        },
    )

    box = compute_bounding_box(snapshot)

    assert box.min_x == 0.0
    assert box.max_y == 2.0
    assert box.width == 4.0
    assert box.height == 2.0
    assert compute_aspect_ratio(snapshot) == 2.0


def test_compute_aspect_ratio_handles_degenerate_boxes() -> None:
    empty = LayoutSnapshot(time=0.0, nodes={})
    flat = LayoutSnapshot(
        time=0.0,
        nodes={
            "A": SnapshotPoint(node_id="A", x=0.0, y=1.0),
            "B": SnapshotPoint(node_id="B", x=4.0, y=1.0),
        },
    )

    assert compute_aspect_ratio(empty) == 1.0
    assert compute_aspect_ratio(flat) == inf


def test_affine_scale_layout_matches_target_aspect_ratio_without_mutating_input() -> None:
    layout = _layout()

    scaled = affine_scale_layout(layout, target_aspect_ratio=1.0)

    assert isclose(compute_aspect_ratio(scaled), 1.0)
    assert compute_aspect_ratio(layout) == 2.0
    assert scaled.config is layout.config
    assert scaled.trajectories["A"][0].y == -1.0
    assert scaled.trajectories["B"][0].y == 3.0


def test_affine_scale_layout_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError):
        affine_scale_layout(_layout(), target_aspect_ratio=0.0)
