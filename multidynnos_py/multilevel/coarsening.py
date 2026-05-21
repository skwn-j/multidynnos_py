"""Independent-set coarsening for the multilevel layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.multilevel.flattener import StaticGraphSummary, StaticSumPresenceFlattener


@dataclass(frozen=True, slots=True)
class CoarseningStep:
    """One fine-to-coarse transition in a multilevel hierarchy."""

    fine_graph: DynamicGraph
    coarse_graph: DynamicGraph
    fine_to_coarse: dict[str, str]
    coarse_to_fine: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class MultilevelHierarchy:
    """Fine-to-coarse graph hierarchy."""

    levels: list[DynamicGraph]
    steps: list[CoarseningStep]


class IndependentSetCoarsener:
    """Greedy maximal independent-set coarsener."""

    def __init__(self, flattener: StaticSumPresenceFlattener | None = None) -> None:
        self._flattener = flattener or StaticSumPresenceFlattener()

    def coarsen(self, graph: DynamicGraph) -> CoarseningStep:
        summary = self._flattener.flatten(graph)
        anchors = self._select_independent_set(graph, summary)
        fine_to_coarse = self._assign_to_anchors(graph, summary, anchors)
        coarse_to_fine = _invert_mapping(fine_to_coarse)
        coarse_graph = _build_coarse_graph(graph, fine_to_coarse, coarse_to_fine)
        return CoarseningStep(
            fine_graph=graph,
            coarse_graph=coarse_graph,
            fine_to_coarse=fine_to_coarse,
            coarse_to_fine=coarse_to_fine,
        )

    def _select_independent_set(self, graph: DynamicGraph, summary: StaticGraphSummary) -> set[str]:
        selected: set[str] = set()
        ordered_nodes = sorted(
            graph.nodes,
            key=lambda node_id: (
                -summary.weighted_degree(node_id),
                -summary.node_weights.get(node_id, 0.0),
                node_id,
            ),
        )
        for node_id in ordered_nodes:
            if any(neighbor in selected for neighbor in summary.neighbors(node_id)):
                continue
            selected.add(node_id)
        return selected

    def _assign_to_anchors(
        self,
        graph: DynamicGraph,
        summary: StaticGraphSummary,
        anchors: set[str],
    ) -> dict[str, str]:
        fine_to_coarse: dict[str, str] = {}
        for node_id in sorted(graph.nodes):
            if node_id in anchors:
                fine_to_coarse[node_id] = node_id
                continue

            anchor_neighbors = [
                (neighbor_id, weight)
                for neighbor_id, weight in summary.neighbors(node_id).items()
                if neighbor_id in anchors
            ]
            if not anchor_neighbors:
                anchors.add(node_id)
                fine_to_coarse[node_id] = node_id
                continue

            anchor_id = max(anchor_neighbors, key=lambda item: (item[1], item[0]))[0]
            fine_to_coarse[node_id] = anchor_id
        return fine_to_coarse


def build_independent_set_hierarchy(
    graph: DynamicGraph,
    *,
    max_levels: int = 4,
    min_coarse_nodes: int = 2,
    coarsener: IndependentSetCoarsener | None = None,
) -> MultilevelHierarchy:
    """Build a fine-to-coarse hierarchy using independent-set coarsening."""

    if max_levels < 1:
        raise ValueError("max_levels must be at least 1.")
    if min_coarse_nodes < 1:
        raise ValueError("min_coarse_nodes must be at least 1.")

    active_coarsener = coarsener or IndependentSetCoarsener()
    levels = [graph]
    steps: list[CoarseningStep] = []
    current = graph

    while len(levels) < max_levels and len(current.nodes) > min_coarse_nodes:
        step = active_coarsener.coarsen(current)
        if len(step.coarse_graph.nodes) >= len(current.nodes):
            break
        steps.append(step)
        levels.append(step.coarse_graph)
        current = step.coarse_graph

    return MultilevelHierarchy(levels=levels, steps=steps)


def _invert_mapping(fine_to_coarse: dict[str, str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for fine_id, coarse_id in fine_to_coarse.items():
        grouped.setdefault(coarse_id, []).append(fine_id)
    return {
        coarse_id: tuple(sorted(fine_ids))
        for coarse_id, fine_ids in sorted(grouped.items())
    }


def _build_coarse_graph(
    graph: DynamicGraph,
    fine_to_coarse: dict[str, str],
    coarse_to_fine: dict[str, tuple[str, ...]],
) -> DynamicGraph:
    coarse_graph = DynamicGraph()

    for coarse_id, fine_ids in coarse_to_fine.items():
        coarse_graph.add_node(coarse_id)
        for fine_id in fine_ids:
            for interval in graph.nodes[fine_id].appearances:
                coarse_graph.add_node_presence(coarse_id, interval)

    for edge in graph.edges.values():
        source_id = fine_to_coarse[edge.source]
        target_id = fine_to_coarse[edge.target]
        if source_id == target_id:
            continue
        for interval in edge.appearances:
            coarse_graph.add_edge_presence(source_id, target_id, interval)

    return coarse_graph
