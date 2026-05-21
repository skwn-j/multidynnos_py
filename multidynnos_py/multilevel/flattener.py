"""Static flattening utilities for multilevel dynamic graph layout."""

from __future__ import annotations

from dataclasses import dataclass, field

from multidynnos_py.data.graph import DynamicGraph, canonical_edge_key


@dataclass(frozen=True, slots=True)
class StaticGraphSummary:
    """Weighted static graph obtained by summing temporal presence."""

    node_weights: dict[str, float]
    edge_weights: dict[tuple[str, str], float]
    adjacency: dict[str, dict[str, float]] = field(default_factory=dict)

    def edge_weight(self, source_id: str, target_id: str) -> float:
        return self.edge_weights.get(canonical_edge_key(source_id, target_id), 0.0)

    def weighted_degree(self, node_id: str) -> float:
        return sum(self.adjacency.get(node_id, {}).values())

    def neighbors(self, node_id: str) -> dict[str, float]:
        return self.adjacency.get(node_id, {})


class StaticSumPresenceFlattener:
    """Flatten a dynamic graph by summing node and edge presence durations."""

    def flatten(self, graph: DynamicGraph) -> StaticGraphSummary:
        node_weights = {
            node_id: sum(interval.duration for interval in node.appearances)
            for node_id, node in graph.nodes.items()
        }
        adjacency: dict[str, dict[str, float]] = {node_id: {} for node_id in graph.nodes}
        edge_weights: dict[tuple[str, str], float] = {}

        for edge in graph.edges.values():
            key = canonical_edge_key(edge.source, edge.target)
            weight = sum(interval.duration for interval in edge.appearances)
            if weight <= 0.0:
                weight = 1.0
            edge_weights[key] = edge_weights.get(key, 0.0) + weight
            adjacency.setdefault(edge.source, {})
            adjacency.setdefault(edge.target, {})
            adjacency[edge.source][edge.target] = adjacency[edge.source].get(edge.target, 0.0) + weight
            adjacency[edge.target][edge.source] = adjacency[edge.target].get(edge.source, 0.0) + weight

        return StaticGraphSummary(
            node_weights=node_weights,
            edge_weights=edge_weights,
            adjacency=adjacency,
        )
