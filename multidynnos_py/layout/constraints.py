"""Movement constraints for the minimal DynNoSlice-style layout."""

from __future__ import annotations

from multidynnos_py.layout.forces import ForceMap, PositionMap, Vector, magnitude, scale


def clamp_vector(vector: Vector, max_magnitude: float) -> Vector:
    current_magnitude = magnitude(vector)
    if current_magnitude == 0.0 or current_magnitude <= max_magnitude:
        return vector
    return scale(vector, max_magnitude / current_magnitude)


def apply_forces(
    positions: PositionMap,
    forces: ForceMap,
    *,
    learning_rate: float,
    max_step: float,
) -> PositionMap:
    """Apply forces with a per-iteration movement cap."""

    updated: PositionMap = {}
    for node_id, time_positions in positions.items():
        updated[node_id] = {}
        for time, position in time_positions.items():
            raw_step = scale(forces[node_id][time], learning_rate)
            step = clamp_vector(raw_step, max_step)
            updated[node_id][time] = (position[0] + step[0], position[1] + step[1])
    return updated

