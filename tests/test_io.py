from __future__ import annotations

import json
from io import StringIO

import pytest

from multidynnos_py.data.graph import deterministic_edge_id
from multidynnos_py.data.io import (
    build_graph_from_appearances,
    export_graph_csv,
    export_graph_json,
    load_custom_graph,
    parse_edge_appearances,
    parse_node_appearances,
)


def test_parse_custom_node_and_edge_appearances() -> None:
    nodes = parse_node_appearances(["Alice,1,5", " Bob , 2 , 4.6 "])
    edges = parse_edge_appearances(["Alice,Bob,2.5,1"])

    assert nodes[0].node_id == "Alice"
    assert nodes[0].interval.start == 1.0
    assert nodes[0].interval.end == 6.0
    assert nodes[1].node_id == "Bob"
    assert edges[0].source_id == "Alice"
    assert edges[0].target_id == "Bob"


def test_invalid_rows_raise_by_default() -> None:
    with pytest.raises(ValueError):
        parse_node_appearances(["Alice,1"])

    with pytest.raises(ValueError):
        parse_edge_appearances(["Alice,Bob,not-a-number,1"])


def test_build_graph_from_appearances() -> None:
    graph = build_graph_from_appearances(
        parse_node_appearances(["Alice,1,5", "Bob,2,4.6", "Carol,1.5,3"]),
        parse_edge_appearances(["Alice,Bob,2.5,1", "Bob,Carol,2.1,0.6"]),
    )

    assert [node.id for node in graph.active_nodes(2.6)] == ["Alice", "Bob", "Carol"]
    assert [edge.id for edge in graph.active_edges(2.6)] == [
        deterministic_edge_id("Alice", "Bob"),
        deterministic_edge_id("Bob", "Carol"),
    ]


def test_load_custom_graph_from_files(tmp_path) -> None:
    node_file = tmp_path / "nodes.csv"
    edge_file = tmp_path / "edges.csv"
    node_file.write_text("Alice,1,5\nBob,2,4.6\n")
    edge_file.write_text("Alice,Bob,2.5,1\n")

    graph = load_custom_graph(node_file, edge_file)

    assert list(graph.nodes) == ["Alice", "Bob"]
    assert list(graph.edges) == [deterministic_edge_id("Alice", "Bob")]


def test_export_graph_json() -> None:
    graph = build_graph_from_appearances(
        parse_node_appearances(["Alice,1,5", "Bob,2,4.6"]),
        parse_edge_appearances(["Alice,Bob,2.5,1"]),
    )
    destination = StringIO()

    export_graph_json(graph, destination)
    payload = json.loads(destination.getvalue())

    assert payload["nodes"][0]["id"] == "Alice"
    assert payload["edges"][0]["id"] == deterministic_edge_id("Alice", "Bob")


def test_export_graph_csv_in_original_format() -> None:
    graph = build_graph_from_appearances(
        parse_node_appearances(["Alice,1,5", "Bob,2,4.6"]),
        parse_edge_appearances(["Alice,Bob,2.5,1"]),
    )
    node_destination = StringIO()
    edge_destination = StringIO()

    export_graph_csv(graph, node_destination, edge_destination)

    assert node_destination.getvalue() == "Alice,1,5\nBob,2,4.6\n"
    assert edge_destination.getvalue() == "Alice,Bob,2.5,1\n"

