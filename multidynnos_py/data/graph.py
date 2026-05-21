"""Dynamic graph data structures for the Phase 2 MultiDynNoS port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from multidynnos_py.data.temporal import Interval, presence_at


def canonical_edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    """Return a deterministic undirected key for a source-target pair."""

    if source_id == target_id:
        raise ValueError(f"Loop edges are not supported: {source_id!r}.")
    return tuple(sorted((source_id, target_id)))


def deterministic_edge_id(source_id: str, target_id: str) -> str:
    """Create a stable edge ID for custom inputs without explicit edge IDs."""

    source_key, target_key = canonical_edge_key(source_id, target_id)
    encoded_source = quote(source_key, safe="")
    encoded_target = quote(target_key, safe="")
    return f"e:{encoded_source}--{encoded_target}"


@dataclass(slots=True)
class Node:
    """A dynamic graph node with one or more presence intervals."""

    id: str
    appearances: list[Interval] = field(default_factory=list)

    def add_presence(self, interval: Interval) -> None:
        self.appearances.append(interval)
        self.appearances.sort(key=lambda item: (item.start, item.end))

    def presence_at(self, time: float) -> bool:
        return presence_at(self.appearances, time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "appearances": [interval.to_dict() for interval in self.appearances],
        }


@dataclass(slots=True)
class Edge:
    """A dynamic graph edge with one or more presence intervals."""

    id: str
    source: str
    target: str
    appearances: list[Interval] = field(default_factory=list)

    def add_presence(self, interval: Interval) -> None:
        self.appearances.append(interval)
        self.appearances.sort(key=lambda item: (item.start, item.end))

    def presence_at(self, time: float) -> bool:
        return presence_at(self.appearances, time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "appearances": [interval.to_dict() for interval in self.appearances],
        }


@dataclass(slots=True)
class DynamicGraph:
    """A minimal dynamic graph for custom MultiDynNoS input data."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    _edge_ids_by_key: dict[tuple[str, str], str] = field(default_factory=dict, init=False)

    def add_node(self, node_id: str) -> Node:
        if not node_id:
            raise ValueError("Node ID cannot be empty.")
        node = self.nodes.get(node_id)
        if node is None:
            node = Node(node_id)
            self.nodes[node_id] = node
        return node

    def add_node_presence(self, node_id: str, interval: Interval) -> Node:
        node = self.add_node(node_id)
        node.add_presence(interval)
        return node

    def add_edge(self, source_id: str, target_id: str, edge_id: str | None = None) -> Edge:
        if source_id not in self.nodes:
            raise ValueError(f"Edge source node {source_id!r} has not been defined.")
        if target_id not in self.nodes:
            raise ValueError(f"Edge target node {target_id!r} has not been defined.")

        edge_key = canonical_edge_key(source_id, target_id)
        existing_id = self._edge_ids_by_key.get(edge_key)
        if existing_id is not None:
            return self.edges[existing_id]

        canonical_source, canonical_target = edge_key
        new_edge_id = edge_id or deterministic_edge_id(canonical_source, canonical_target)
        if new_edge_id in self.edges:
            raise ValueError(f"Edge ID {new_edge_id!r} already exists.")

        edge = Edge(id=new_edge_id, source=canonical_source, target=canonical_target)
        self.edges[new_edge_id] = edge
        self._edge_ids_by_key[edge_key] = new_edge_id
        return edge

    def add_edge_presence(
        self,
        source_id: str,
        target_id: str,
        interval: Interval,
        edge_id: str | None = None,
    ) -> Edge:
        edge = self.add_edge(source_id, target_id, edge_id=edge_id)
        edge.add_presence(interval)
        return edge

    def get_edge_between(self, source_id: str, target_id: str) -> Edge | None:
        edge_id = self._edge_ids_by_key.get(canonical_edge_key(source_id, target_id))
        if edge_id is None:
            return None
        return self.edges[edge_id]

    def active_nodes(self, time: float) -> list[Node]:
        return sorted(
            (node for node in self.nodes.values() if node.presence_at(time)),
            key=lambda node: node.id,
        )

    def active_edges(self, time: float) -> list[Edge]:
        active_node_ids = {node.id for node in self.active_nodes(time)}
        return sorted(
            (
                edge
                for edge in self.edges.values()
                if edge.presence_at(time)
                and edge.source in active_node_ids
                and edge.target in active_node_ids
            ),
            key=lambda edge: edge.id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)],
            "edges": [self.edges[edge_id].to_dict() for edge_id in sorted(self.edges)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicGraph:
        graph = cls()
        for node_data in data.get("nodes", []):
            node_id = str(node_data["id"])
            node = graph.add_node(node_id)
            for interval_data in node_data.get("appearances", []):
                node.add_presence(Interval.from_dict(interval_data))

        for edge_data in data.get("edges", []):
            source = str(edge_data["source"])
            target = str(edge_data["target"])
            edge = graph.add_edge(source, target, edge_id=str(edge_data["id"]))
            for interval_data in edge_data.get("appearances", []):
                edge.add_presence(Interval.from_dict(interval_data))
        return graph

