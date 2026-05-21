"""Metrics for flat dynamic graph layout experiments."""

from __future__ import annotations

from collections import deque
from math import sqrt

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.experiments.snapshots import LayoutSnapshot
from multidynnos_py.layout.dynnoslice import DynamicLayout, TrajectoryPoint


def movement_metric(layout: DynamicLayout) -> float:
    """Return average per-node trajectory length."""

    if not layout.trajectories:
        return 0.0

    total = 0.0
    counted_nodes = 0
    for trajectory in layout.trajectories.values():
        ordered = sorted(trajectory, key=lambda point: point.time)
        if not ordered:
            continue
        counted_nodes += 1
        for left, right in zip(ordered, ordered[1:]):
            total += _point_distance(left, right)
    return total / counted_nodes if counted_nodes else 0.0


def crowding_metric(snapshot: LayoutSnapshot, *, min_distance: float = 1.0) -> int:
    """Count node pairs closer than ``min_distance`` in one snapshot."""

    nodes = [snapshot.nodes[node_id] for node_id in sorted(snapshot.nodes)]
    crowding = 0
    for index, first in enumerate(nodes):
        for second in nodes[index + 1 :]:
            distance = sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)
            if distance < min_distance:
                crowding += 1
    return crowding


def stress_metric(snapshot: LayoutSnapshot, graph: DynamicGraph) -> float:
    """Compute unweighted static stress for active connected node pairs."""

    node_ids = sorted(snapshot.nodes)
    if len(node_ids) < 2:
        return 0.0

    adjacency = _active_adjacency(snapshot, graph)
    total_stress = 0.0
    connected_pairs = 0
    for index, source_id in enumerate(node_ids):
        distances = _shortest_paths(source_id, adjacency)
        source_position = snapshot.nodes[source_id]
        for target_id in node_ids[index + 1 :]:
            theoretical = distances.get(target_id)
            if theoretical is None:
                continue
            target_position = snapshot.nodes[target_id]
            spatial = sqrt(
                (source_position.x - target_position.x) ** 2
                + (source_position.y - target_position.y) ** 2
            )
            total_stress += ((theoretical - spatial) / theoretical) ** 2
            connected_pairs += 1
    return total_stress if connected_pairs else 0.0


def _active_adjacency(snapshot: LayoutSnapshot, graph: DynamicGraph) -> dict[str, set[str]]:
    node_ids = set(snapshot.nodes)
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in graph.active_edges(snapshot.time):
        if edge.source in node_ids and edge.target in node_ids:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    return adjacency


def _shortest_paths(source_id: str, adjacency: dict[str, set[str]]) -> dict[str, int]:
    distances = {source_id: 0}
    queue: deque[str] = deque([source_id])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _point_distance(left: TrajectoryPoint, right: TrajectoryPoint) -> float:
    return sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2)

