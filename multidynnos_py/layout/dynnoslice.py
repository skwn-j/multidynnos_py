"""A minimal single-level DynNoSlice-style dynamic layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from random import Random
from typing import Any

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.layout.constraints import apply_forces
from multidynnos_py.layout.forces import (
    PositionMap,
    add_force_maps,
    connection_attraction_forces,
    gravity_forces,
    repulsion_forces,
    temporal_smoothing_forces,
    zero_forces,
)


@dataclass(frozen=True, slots=True)
class DynNoSliceConfig:
    """Configuration for the minimal single-level dynamic layout."""

    tau: float = 1.0
    delta: float = 5.0
    iterations: int = 100
    learning_rate: float = 0.05
    seed: int = 73

    def __post_init__(self) -> None:
        if self.tau < 0:
            raise ValueError("tau must be non-negative.")
        if self.delta <= 0:
            raise ValueError("delta must be positive.")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """A sampled node trajectory point."""

    time: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class DynamicLayout:
    """A JSON-serializable dynamic layout result."""

    trajectories: dict[str, list[TrajectoryPoint]]
    sample_times: list[float]
    config: DynNoSliceConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "sample_times": self.sample_times,
            "trajectories": {
                node_id: [point.to_dict() for point in points]
                for node_id, points in sorted(self.trajectories.items())
            },
        }


def run_dynnoslice(
    graph: DynamicGraph,
    config: DynNoSliceConfig,
    *,
    initial_layout: DynamicLayout | None = None,
) -> DynamicLayout:
    """Run a compact DynNoSlice-inspired layout on a dynamic graph."""

    sample_times = _sample_times(graph)
    if initial_layout is None:
        positions = _initial_positions(graph, sample_times, config)
    else:
        positions = _positions_from_initial_layout(graph, sample_times, config, initial_layout)
    max_step = config.delta * 0.25

    for _ in range(config.iterations):
        forces = zero_forces(positions)
        add_force_maps(forces, gravity_forces(positions, sample_times))
        add_force_maps(
            forces,
            connection_attraction_forces(graph, positions, sample_times, delta=config.delta),
        )
        add_force_maps(
            forces,
            repulsion_forces(graph, positions, sample_times, delta=config.delta),
        )
        add_force_maps(
            forces,
            temporal_smoothing_forces(positions, tau=config.tau),
        )
        positions = apply_forces(
            positions,
            forces,
            learning_rate=config.learning_rate,
            max_step=max_step,
        )

    return DynamicLayout(
        trajectories=_to_trajectories(positions),
        sample_times=sample_times,
        config=config,
    )


def _sample_times(graph: DynamicGraph) -> list[float]:
    times: set[float] = set()
    for node in graph.nodes.values():
        for interval in node.appearances:
            _add_interval_sample_times(times, interval.start, interval.end)
    for edge in graph.edges.values():
        for interval in edge.appearances:
            _add_interval_sample_times(times, interval.start, interval.end)
    return sorted(times)


def _add_interval_sample_times(times: set[float], start: float, end: float) -> None:
    if not (isfinite(start) and isfinite(end)):
        return
    times.add(start)
    times.add(end)
    if end > start:
        times.add((start + end) / 2.0)


def _initial_positions(graph: DynamicGraph, sample_times: list[float], config: DynNoSliceConfig) -> PositionMap:
    rng = Random(config.seed)
    diameter = sqrt(max(len(graph.nodes), 1) * config.delta)
    base_positions = {
        node_id: (rng.random() * diameter, rng.random() * diameter)
        for node_id in sorted(graph.nodes)
    }

    positions: PositionMap = {}
    for node_id, node in graph.nodes.items():
        node_positions: dict[float, tuple[float, float]] = {}
        base_x, base_y = base_positions[node_id]
        for time in sample_times:
            if node.presence_at(time):
                node_positions[time] = (base_x, base_y)
        positions[node_id] = node_positions
    return positions


def _positions_from_initial_layout(
    graph: DynamicGraph,
    sample_times: list[float],
    config: DynNoSliceConfig,
    initial_layout: DynamicLayout,
) -> PositionMap:
    fallback = _initial_positions(graph, sample_times, config)
    positions: PositionMap = {}
    for node_id, node in graph.nodes.items():
        node_positions: dict[float, tuple[float, float]] = {}
        trajectory = initial_layout.trajectories.get(node_id, [])
        for time in sample_times:
            if not node.presence_at(time):
                continue
            sampled = _sample_trajectory(trajectory, time)
            node_positions[time] = sampled if sampled is not None else fallback[node_id][time]
        positions[node_id] = node_positions
    return positions


def _sample_trajectory(trajectory: list[TrajectoryPoint], time: float) -> tuple[float, float] | None:
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


def _to_trajectories(positions: PositionMap) -> dict[str, list[TrajectoryPoint]]:
    trajectories: dict[str, list[TrajectoryPoint]] = {}
    for node_id, time_positions in sorted(positions.items()):
        trajectories[node_id] = [
            TrajectoryPoint(time=time, x=coords[0], y=coords[1])
            for time, coords in sorted(time_positions.items())
        ]
    return trajectories
