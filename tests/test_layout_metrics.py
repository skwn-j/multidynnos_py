from __future__ import annotations

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.data.temporal import Interval
from multidynnos_py.experiments.snapshots import LayoutSnapshot, SnapshotPoint
from multidynnos_py.layout.dynnoslice import DynNoSliceConfig, DynamicLayout, TrajectoryPoint
from multidynnos_py.metrics.layout_metrics import crowding_metric, movement_metric, stress_metric


def test_movement_metric_averages_node_path_lengths() -> None:
    layout = DynamicLayout(
        trajectories={
            "A": [
                TrajectoryPoint(time=0.0, x=0.0, y=0.0),
                TrajectoryPoint(time=1.0, x=3.0, y=4.0),
            ],
            "B": [
                TrajectoryPoint(time=0.0, x=1.0, y=1.0),
                TrajectoryPoint(time=1.0, x=1.0, y=3.0),
            ],
        },
        sample_times=[0.0, 1.0],
        config=DynNoSliceConfig(iterations=0),
    )

    assert movement_metric(layout) == 3.5


def test_crowding_metric_counts_close_node_pairs() -> None:
    snapshot = LayoutSnapshot(
        time=0.0,
        nodes={
            "A": SnapshotPoint(node_id="A", x=0.0, y=0.0),
            "B": SnapshotPoint(node_id="B", x=0.5, y=0.0),
            "C": SnapshotPoint(node_id="C", x=4.0, y=0.0),
        },
    )

    assert crowding_metric(snapshot) == 1
    assert crowding_metric(snapshot, min_distance=5.0) == 3


def test_stress_metric_is_zero_when_graph_distance_matches_layout_distance() -> None:
    graph = DynamicGraph()
    graph.add_node_presence("A", Interval.closed(0.0, 2.0))
    graph.add_node_presence("B", Interval.closed(0.0, 2.0))
    graph.add_node_presence("C", Interval.closed(0.0, 2.0))
    graph.add_edge_presence("A", "B", Interval.closed(0.0, 2.0))
    graph.add_edge_presence("B", "C", Interval.closed(0.0, 2.0))
    snapshot = LayoutSnapshot(
        time=1.0,
        nodes={
            "A": SnapshotPoint(node_id="A", x=0.0, y=0.0),
            "B": SnapshotPoint(node_id="B", x=1.0, y=0.0),
            "C": SnapshotPoint(node_id="C", x=2.0, y=0.0),
        },
    )

    assert stress_metric(snapshot, graph) == 0.0
