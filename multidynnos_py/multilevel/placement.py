"""Placement strategies used when expanding coarse layouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt
from random import Random

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.layout.dynnoslice import DynamicLayout, DynNoSliceConfig, TrajectoryPoint
from multidynnos_py.multilevel.flattener import StaticGraphSummary, StaticSumPresenceFlattener

Vector = tuple[float, float]


@dataclass(frozen=True, slots=True)
class WeightedBarycenterPlacementStrategy:
    """Place expanded nodes near their coarse parent and weighted neighbors."""

    seed: int = 73
    jitter_scale: float = 0.1
    parent_weight: float = 0.7

    def place(
        self,
        fine_graph: DynamicGraph,
        coarse_layout: DynamicLayout,
        fine_to_coarse: dict[str, str],
        *,
        config: DynNoSliceConfig,
        static_summary: StaticGraphSummary | None = None,
    ) -> DynamicLayout:
        summary = static_summary or StaticSumPresenceFlattener().flatten(fine_graph)
        sample_times = _sample_times(fine_graph)
        fallback_positions = _fallback_positions(fine_graph, sample_times, config, self.seed)
        members_by_parent = _members_by_parent(fine_to_coarse)
        trajectories: dict[str, list[TrajectoryPoint]] = {}

        for node_id, node in sorted(fine_graph.nodes.items()):
            parent_id = fine_to_coarse[node_id]
            member_index = members_by_parent[parent_id].index(node_id)
            member_count = len(members_by_parent[parent_id])
            points: list[TrajectoryPoint] = []
            for time in sample_times:
                if not node.presence_at(time):
                    continue
                parent_position = _sample_layout_position(coarse_layout, parent_id, time)
                barycenter = self._neighbor_barycenter(
                    node_id,
                    time,
                    fine_graph,
                    coarse_layout,
                    fine_to_coarse,
                    summary,
                )
                base_position = _combine_positions(parent_position, barycenter, self.parent_weight)
                if base_position is None:
                    base_position = fallback_positions[node_id][time]
                x, y = _offset_position(
                    base_position,
                    member_index,
                    member_count,
                    radius=max(config.delta * self.jitter_scale, 0.0),
                )
                points.append(TrajectoryPoint(time=time, x=x, y=y))
            trajectories[node_id] = points

        return DynamicLayout(trajectories=trajectories, sample_times=sample_times, config=config)

    def _neighbor_barycenter(
        self,
        node_id: str,
        time: float,
        fine_graph: DynamicGraph,
        coarse_layout: DynamicLayout,
        fine_to_coarse: dict[str, str],
        summary: StaticGraphSummary,
    ) -> Vector | None:
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        for neighbor_id, static_weight in summary.neighbors(node_id).items():
            if not fine_graph.nodes[neighbor_id].presence_at(time):
                continue
            parent_id = fine_to_coarse[neighbor_id]
            position = _sample_layout_position(coarse_layout, parent_id, time)
            if position is None:
                continue
            weight = max(static_weight, 1e-9)
            weighted_x += position[0] * weight
            weighted_y += position[1] * weight
            total_weight += weight
        if total_weight == 0.0:
            return None
        return weighted_x / total_weight, weighted_y / total_weight


def _combine_positions(parent_position: Vector | None, barycenter: Vector | None, parent_weight: float) -> Vector | None:
    if parent_position is None:
        return barycenter
    if barycenter is None:
        return parent_position
    parent_fraction = min(max(parent_weight, 0.0), 1.0)
    barycenter_fraction = 1.0 - parent_fraction
    return (
        parent_position[0] * parent_fraction + barycenter[0] * barycenter_fraction,
        parent_position[1] * parent_fraction + barycenter[1] * barycenter_fraction,
    )


def _offset_position(position: Vector, index: int, count: int, *, radius: float) -> Vector:
    if count <= 1 or radius == 0.0:
        return position
    angle = (2.0 * pi * index) / count
    return position[0] + cos(angle) * radius, position[1] + sin(angle) * radius


def _members_by_parent(fine_to_coarse: dict[str, str]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = {}
    for node_id, parent_id in fine_to_coarse.items():
        members.setdefault(parent_id, []).append(node_id)
    for member_ids in members.values():
        member_ids.sort()
    return members


def _fallback_positions(
    graph: DynamicGraph,
    sample_times: list[float],
    config: DynNoSliceConfig,
    seed: int,
) -> dict[str, dict[float, Vector]]:
    rng = Random(seed)
    node_ids = sorted(graph.nodes)
    radius = sqrt(max(len(node_ids), 1) * config.delta)
    base_positions: dict[str, Vector] = {}
    for index, node_id in enumerate(node_ids):
        angle = (2.0 * pi * index) / max(len(node_ids), 1)
        base_positions[node_id] = (
            cos(angle) * radius + rng.random() * 1e-6,
            sin(angle) * radius + rng.random() * 1e-6,
        )

    return {
        node_id: {
            time: base_positions[node_id]
            for time in sample_times
            if graph.nodes[node_id].presence_at(time)
        }
        for node_id in node_ids
    }


def _sample_layout_position(layout: DynamicLayout, node_id: str, time: float) -> Vector | None:
    trajectory = sorted(layout.trajectories.get(node_id, []), key=lambda point: point.time)
    if not trajectory:
        return None
    if time < trajectory[0].time or time > trajectory[-1].time:
        return None
    for point in trajectory:
        if point.time == time:
            return point.x, point.y
    for left, right in zip(trajectory, trajectory[1:]):
        if left.time <= time <= right.time:
            if right.time == left.time:
                return left.x, left.y
            alpha = (time - left.time) / (right.time - left.time)
            return (
                left.x + (right.x - left.x) * alpha,
                left.y + (right.y - left.y) * alpha,
            )
    return None


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
