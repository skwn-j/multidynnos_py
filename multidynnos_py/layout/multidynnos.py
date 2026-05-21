"""First working multilevel MultiDynNoS-style layout pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from math import cos, isfinite, pi, sin, sqrt
from shutil import which

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.layout.dynnoslice import DynamicLayout, DynNoSliceConfig, TrajectoryPoint, run_dynnoslice
from multidynnos_py.multilevel.coarsening import MultilevelHierarchy, build_independent_set_hierarchy
from multidynnos_py.multilevel.cooling import dynnoslice_config_for_level, smooth_flexible_trajectories
from multidynnos_py.multilevel.flattener import StaticGraphSummary, StaticSumPresenceFlattener
from multidynnos_py.multilevel.placement import WeightedBarycenterPlacementStrategy

Vector = tuple[float, float]


@dataclass(frozen=True, slots=True)
class MultiDynNoSConfig:
    """Configuration for the minimal multilevel pipeline."""

    dynnoslice_config: DynNoSliceConfig = field(default_factory=DynNoSliceConfig)
    max_levels: int = 4
    min_coarse_nodes: int = 2
    use_sfdp: bool = False
    sfdp_command: str = "sfdp"
    postprocess_passes: int = 1
    expansion_jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.max_levels < 1:
            raise ValueError("max_levels must be at least 1.")
        if self.min_coarse_nodes < 1:
            raise ValueError("min_coarse_nodes must be at least 1.")
        if self.postprocess_passes < 0:
            raise ValueError("postprocess_passes must be non-negative.")
        if self.expansion_jitter < 0.0:
            raise ValueError("expansion_jitter must be non-negative.")


def run_multidynnos(graph: DynamicGraph, config: MultiDynNoSConfig) -> DynamicLayout:
    """Run the first working MultiDynNoS-style multilevel pipeline."""

    hierarchy = build_independent_set_hierarchy(
        graph,
        max_levels=config.max_levels,
        min_coarse_nodes=config.min_coarse_nodes,
    )
    flattener = StaticSumPresenceFlattener()
    coarsest_graph = hierarchy.levels[-1]
    initial_layout = _static_initial_layout(coarsest_graph, config, flattener.flatten(coarsest_graph))
    total_refinements = len(hierarchy.levels)

    level_config = dynnoslice_config_for_level(
        config.dynnoslice_config,
        refinement_index=0,
        total_refinement_steps=total_refinements,
    )
    layout = run_dynnoslice(coarsest_graph, level_config, initial_layout=initial_layout)
    layout = smooth_flexible_trajectories(layout, passes=config.postprocess_passes)

    for refinement_index, step in enumerate(reversed(hierarchy.steps), start=1):
        level_config = dynnoslice_config_for_level(
            config.dynnoslice_config,
            refinement_index=refinement_index,
            total_refinement_steps=total_refinements,
        )
        placement = WeightedBarycenterPlacementStrategy(
            seed=config.dynnoslice_config.seed + refinement_index,
            jitter_scale=config.expansion_jitter,
        )
        expanded_layout = placement.place(
            step.fine_graph,
            layout,
            step.fine_to_coarse,
            config=level_config,
            static_summary=flattener.flatten(step.fine_graph),
        )
        layout = run_dynnoslice(step.fine_graph, level_config, initial_layout=expanded_layout)
        layout = smooth_flexible_trajectories(layout, passes=config.postprocess_passes)

    return layout


def _static_initial_layout(
    graph: DynamicGraph,
    config: MultiDynNoSConfig,
    summary: StaticGraphSummary,
) -> DynamicLayout:
    sample_times = _sample_times(graph)
    positions = _sfdp_positions(summary, config) if config.use_sfdp else None
    if positions is None:
        positions = _fallback_static_positions(graph, config.dynnoslice_config)

    trajectories: dict[str, list[TrajectoryPoint]] = {}
    for node_id, node in sorted(graph.nodes.items()):
        x, y = positions[node_id]
        trajectories[node_id] = [
            TrajectoryPoint(time=time, x=x, y=y)
            for time in sample_times
            if node.presence_at(time)
        ]
    return DynamicLayout(
        trajectories=trajectories,
        sample_times=sample_times,
        config=config.dynnoslice_config,
    )


def _sfdp_positions(summary: StaticGraphSummary, config: MultiDynNoSConfig) -> dict[str, Vector] | None:
    command = which(config.sfdp_command)
    if command is None:
        return None

    node_ids = sorted(summary.node_weights)
    name_by_node = {node_id: f"n{index}" for index, node_id in enumerate(node_ids)}
    node_by_name = {name: node_id for node_id, name in name_by_node.items()}
    dot_lines = ["graph G {"]
    for node_id in node_ids:
        dot_lines.append(f"  {name_by_node[node_id]};")
    for (source_id, target_id), weight in sorted(summary.edge_weights.items()):
        penwidth = max(1.0, min(weight, 8.0))
        dot_lines.append(f"  {name_by_node[source_id]} -- {name_by_node[target_id]} [penwidth={penwidth:g}];")
    dot_lines.append("}")

    try:
        completed = subprocess.run(
            [command, "-Tplain"],
            input="\n".join(dot_lines),
            text=True,
            capture_output=True,
            check=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    positions: dict[str, Vector] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "node" and fields[1] in node_by_name:
            positions[node_by_name[fields[1]]] = (float(fields[2]), float(fields[3]))

    if set(positions) != set(node_ids):
        return None
    return positions


def _fallback_static_positions(graph: DynamicGraph, config: DynNoSliceConfig) -> dict[str, Vector]:
    node_ids = sorted(graph.nodes)
    radius = sqrt(max(len(node_ids), 1) * config.delta)
    return {
        node_id: (
            cos((2.0 * pi * index) / max(len(node_ids), 1)) * radius,
            sin((2.0 * pi * index) / max(len(node_ids), 1)) * radius,
        )
        for index, node_id in enumerate(node_ids)
    }


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
