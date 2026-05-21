"""Conceptual force terms for the minimal DynNoSlice-style layout."""

from __future__ import annotations

from math import sqrt
from typing import TypeAlias

from multidynnos_py.data.graph import DynamicGraph

Vector: TypeAlias = tuple[float, float]
PositionMap: TypeAlias = dict[str, dict[float, Vector]]
ForceMap: TypeAlias = dict[str, dict[float, Vector]]

_EPSILON = 1e-9


def zero_forces(positions: PositionMap) -> ForceMap:
    return {
        node_id: {time: (0.0, 0.0) for time in time_positions}
        for node_id, time_positions in positions.items()
    }


def add_force(forces: ForceMap, node_id: str, time: float, vector: Vector) -> None:
    current = forces[node_id][time]
    forces[node_id][time] = (current[0] + vector[0], current[1] + vector[1])


def add_force_maps(base: ForceMap, addition: ForceMap) -> ForceMap:
    for node_id, time_forces in addition.items():
        for time, force in time_forces.items():
            add_force(base, node_id, time, force)
    return base


def gravity_forces(positions: PositionMap, sample_times: list[float], *, strength: float = 0.02) -> ForceMap:
    """Pull active nodes gently toward their centroid at each sampled time."""

    forces = zero_forces(positions)
    for time in sample_times:
        active_positions = [
            time_positions[time]
            for time_positions in positions.values()
            if time in time_positions
        ]
        if len(active_positions) < 2:
            continue
        center = (
            sum(position[0] for position in active_positions) / len(active_positions),
            sum(position[1] for position in active_positions) / len(active_positions),
        )
        for node_id, time_positions in positions.items():
            if time not in time_positions:
                continue
            vector = subtract(center, time_positions[time])
            add_force(forces, node_id, time, scale(vector, strength))
    return forces


def connection_attraction_forces(
    graph: DynamicGraph,
    positions: PositionMap,
    sample_times: list[float],
    *,
    delta: float,
    strength: float = 0.3,
) -> ForceMap:
    """Spring-like force on active edges with preferred distance ``delta``."""

    forces = zero_forces(positions)
    for time in sample_times:
        for edge in graph.active_edges(time):
            source_position = positions.get(edge.source, {}).get(time)
            target_position = positions.get(edge.target, {}).get(time)
            if source_position is None or target_position is None:
                continue

            vector = subtract(target_position, source_position)
            distance = magnitude(vector)
            if distance <= _EPSILON:
                continue
            unit = scale(vector, 1.0 / distance)
            amount = (distance - delta) * strength
            force = scale(unit, amount)
            add_force(forces, edge.source, time, force)
            add_force(forces, edge.target, time, scale(force, -1.0))
    return forces


def repulsion_forces(
    graph: DynamicGraph,
    positions: PositionMap,
    sample_times: list[float],
    *,
    delta: float,
    strength: float = 0.8,
) -> ForceMap:
    """Repel active node pairs at each sampled time."""

    forces = zero_forces(positions)
    preferred_distance = max(delta, _EPSILON)
    for time in sample_times:
        active_nodes = graph.active_nodes(time)
        for index, first in enumerate(active_nodes):
            first_position = positions.get(first.id, {}).get(time)
            if first_position is None:
                continue
            for second in active_nodes[index + 1 :]:
                second_position = positions.get(second.id, {}).get(time)
                if second_position is None:
                    continue
                vector = subtract(first_position, second_position)
                distance = max(magnitude(vector), _EPSILON)
                unit = scale(vector, 1.0 / distance)
                amount = strength * (preferred_distance * preferred_distance) / (distance * distance)
                force = scale(unit, amount)
                add_force(forces, first.id, time, force)
                add_force(forces, second.id, time, scale(force, -1.0))
    return forces


def temporal_smoothing_forces(
    positions: PositionMap,
    *,
    tau: float,
    strength: float = 0.25,
) -> ForceMap:
    """Straighten each node trajectory over sampled time points."""

    forces = zero_forces(positions)
    smoothing = max(tau, 0.0) * strength
    if smoothing <= 0:
        return forces

    for node_id, time_positions in positions.items():
        ordered_times = sorted(time_positions)
        if len(ordered_times) < 2:
            continue
        for index, time in enumerate(ordered_times):
            current = time_positions[time]
            if index == 0:
                target = time_positions[ordered_times[index + 1]]
                force = scale(subtract(target, current), 0.5 * smoothing)
            elif index == len(ordered_times) - 1:
                target = time_positions[ordered_times[index - 1]]
                force = scale(subtract(target, current), 0.5 * smoothing)
            else:
                previous_time = ordered_times[index - 1]
                next_time = ordered_times[index + 1]
                previous_position = time_positions[previous_time]
                next_position = time_positions[next_time]
                span = max(next_time - previous_time, _EPSILON)
                alpha = (time - previous_time) / span
                target = interpolate(previous_position, next_position, alpha)
                force = scale(subtract(target, current), smoothing)
            add_force(forces, node_id, time, force)
    return forces


def subtract(left: Vector, right: Vector) -> Vector:
    return (left[0] - right[0], left[1] - right[1])


def scale(vector: Vector, factor: float) -> Vector:
    return (vector[0] * factor, vector[1] * factor)


def magnitude(vector: Vector) -> float:
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1])


def interpolate(left: Vector, right: Vector, alpha: float) -> Vector:
    return (
        left[0] + (right[0] - left[0]) * alpha,
        left[1] + (right[1] - left[1]) * alpha,
    )

