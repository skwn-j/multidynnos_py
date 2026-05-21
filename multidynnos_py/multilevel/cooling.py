"""Per-level refinement schedules and trajectory post-processing."""

from __future__ import annotations

from dataclasses import replace

from multidynnos_py.layout.dynnoslice import DynamicLayout, DynNoSliceConfig, TrajectoryPoint


def dynnoslice_config_for_level(
    base_config: DynNoSliceConfig,
    *,
    refinement_index: int,
    total_refinement_steps: int,
) -> DynNoSliceConfig:
    """Return a conservative per-level DynNoSlice configuration."""

    if total_refinement_steps <= 1:
        progress = 1.0
    else:
        progress = refinement_index / (total_refinement_steps - 1)

    iteration_factor = 1.5 - 0.5 * progress
    learning_rate_factor = 1.0 - 0.25 * progress
    iterations = 0 if base_config.iterations == 0 else max(1, round(base_config.iterations * iteration_factor))
    learning_rate = base_config.learning_rate * learning_rate_factor
    return replace(base_config, iterations=iterations, learning_rate=learning_rate)


def smooth_flexible_trajectories(layout: DynamicLayout, *, passes: int = 1) -> DynamicLayout:
    """Apply a light fixed-sample smoothing pass to emulate flexible trajectories."""

    if passes <= 0:
        return layout

    current = layout
    for _ in range(passes):
        trajectories: dict[str, list[TrajectoryPoint]] = {}
        for node_id, trajectory in current.trajectories.items():
            ordered = sorted(trajectory, key=lambda point: point.time)
            if len(ordered) < 3:
                trajectories[node_id] = ordered
                continue
            smoothed = [ordered[0]]
            for previous, point, next_point in zip(ordered, ordered[1:], ordered[2:]):
                smoothed.append(
                    TrajectoryPoint(
                        time=point.time,
                        x=(previous.x + 2.0 * point.x + next_point.x) / 4.0,
                        y=(previous.y + 2.0 * point.y + next_point.y) / 4.0,
                    )
                )
            smoothed.append(ordered[-1])
            trajectories[node_id] = smoothed
        current = DynamicLayout(
            trajectories=trajectories,
            sample_times=list(current.sample_times),
            config=current.config,
        )
    return current
