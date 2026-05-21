from __future__ import annotations

from multidynnos_py.data.graph import DynamicGraph, deterministic_edge_id
from multidynnos_py.data.temporal import Interval


def test_dynamic_graph_active_nodes_and_edges() -> None:
    graph = DynamicGraph()
    graph.add_node_presence("Alice", Interval.closed(1.0, 5.0))
    graph.add_node_presence("Bob", Interval.closed(2.0, 4.0))
    graph.add_node_presence("Carol", Interval.closed(10.0, 11.0))
    graph.add_edge_presence("Alice", "Bob", Interval.closed(2.5, 3.5))

    assert [node.id for node in graph.active_nodes(3.0)] == ["Alice", "Bob"]
    assert [edge.id for edge in graph.active_edges(3.0)] == [deterministic_edge_id("Alice", "Bob")]
    assert graph.active_edges(4.5) == []
    assert [node.id for node in graph.active_nodes(10.5)] == ["Carol"]


def test_repeated_or_reversed_edge_uses_same_deterministic_id() -> None:
    graph = DynamicGraph()
    graph.add_node_presence("Alice", Interval.closed(0.0, 10.0))
    graph.add_node_presence("Bob", Interval.closed(0.0, 10.0))

    edge_a = graph.add_edge_presence("Alice", "Bob", Interval.closed(1.0, 2.0))
    edge_b = graph.add_edge_presence("Bob", "Alice", Interval.closed(3.0, 4.0))

    assert edge_a is edge_b
    assert edge_a.id == deterministic_edge_id("Bob", "Alice")
    assert len(graph.edges) == 1
    assert edge_a.presence_at(1.5)
    assert edge_a.presence_at(3.5)


def test_to_dict_roundtrip_preserves_graph() -> None:
    graph = DynamicGraph()
    graph.add_node_presence("Alice", Interval.closed(1.0, 5.0))
    graph.add_node_presence("Bob", Interval.closed(2.0, 4.0))
    graph.add_edge_presence("Alice", "Bob", Interval.closed(2.5, 3.5))

    restored = DynamicGraph.from_dict(graph.to_dict())

    assert restored.to_dict() == graph.to_dict()
    assert [edge.id for edge in restored.active_edges(3.0)] == [deterministic_edge_id("Alice", "Bob")]

