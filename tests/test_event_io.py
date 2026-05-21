from __future__ import annotations

import csv
from io import StringIO

from multidynnos_py.data.event_io import (
    build_binned_graph_from_edge_events,
    build_graph_from_edge_events,
    export_binned_edge_event_snapshots,
    load_edge_event_graph,
    parse_edge_event_line,
    parse_edge_events,
    parse_edge_events_from_csv,
)
from multidynnos_py.data.graph import deterministic_edge_id


def test_parse_edge_event_line_accepts_whitespace_and_csv() -> None:
    whitespace = parse_edge_event_line("1 2 1082040961")
    comma = parse_edge_event_line("1,2,1082040961")

    assert whitespace == comma
    assert whitespace.source_id == "1"
    assert whitespace.target_id == "2"
    assert whitespace.timestamp == 1082040961.0


def test_parse_edge_events_skips_header_comments_and_blank_lines() -> None:
    events = parse_edge_events(
        [
            "# comment",
            "",
            "src,dst,timestamp",
            "A,B,1.5",
            "B C 2.5",
        ]
    )

    assert [event.source_id for event in events] == ["A", "B"]
    assert [event.target_id for event in events] == ["B", "C"]
    assert [event.timestamp for event in events] == [1.5, 2.5]


def test_parse_edge_events_uses_reordered_csv_header() -> None:
    events = parse_edge_events(
        [
            "timestamp,source,target",
            "10,A,B",
            "11,B,C",
        ],
        has_header=True,
    )

    assert [(event.source_id, event.target_id, event.timestamp) for event in events] == [
        ("A", "B", 10.0),
        ("B", "C", 11.0),
    ]


def test_parse_edge_events_from_csv_dict_reader() -> None:
    rows = csv.DictReader(StringIO("time,from,to\n2.5,A,B\n3.5,B,C\n"))

    events = parse_edge_events_from_csv(rows)

    assert [(event.source_id, event.target_id, event.timestamp) for event in events] == [
        ("A", "B", 2.5),
        ("B", "C", 3.5),
    ]


def test_build_graph_from_edge_events_uses_timestamp_intervals() -> None:
    events = parse_edge_events(["A B 1", "B C 2"])

    graph = build_graph_from_edge_events(events, event_duration=0.5)

    assert [node.id for node in graph.active_nodes(1.25)] == ["A", "B"]
    assert [edge.id for edge in graph.active_edges(1.25)] == [deterministic_edge_id("A", "B")]
    assert graph.active_edges(1.75) == []


def test_load_edge_event_graph_reads_test_data_format(tmp_path) -> None:
    event_file = tmp_path / "events.txt"
    event_file.write_text("1 2 1082040961\n3 4 1082155839\n", encoding="utf-8")

    graph = load_edge_event_graph(event_file)

    assert sorted(graph.nodes) == ["1", "2", "3", "4"]
    assert len(graph.edges) == 2
    assert [node.id for node in graph.active_nodes(1082040961.0)] == ["1", "2"]


def test_build_binned_graph_from_edge_events_aggregates_by_time_bin() -> None:
    events = parse_edge_events(["A,B,0", "A,B,1", "B,C,9", "B,C,10"])

    graph = build_binned_graph_from_edge_events(events, time_bins=2)

    assert len(graph.nodes["A"].appearances) == 1
    assert len(graph.get_edge_between("A", "B").appearances) == 1
    assert graph.get_edge_between("A", "B").appearances[0].start == 0.0
    assert graph.get_edge_between("A", "B").appearances[0].end == 5.0
    assert graph.get_edge_between("B", "C").appearances[0].start == 5.0
    assert graph.get_edge_between("B", "C").appearances[0].end == 10.0


def test_load_edge_event_graph_with_time_bins_reduces_temporal_detail(tmp_path) -> None:
    event_file = tmp_path / "events.csv"
    event_file.write_text("A,B,0\nA,B,1\nA,B,2\nA,B,3\n", encoding="utf-8")

    graph = load_edge_event_graph(event_file, time_bins=2)

    assert len(graph.get_edge_between("A", "B").appearances) == 2
    assert graph.active_edges(0.75)
    assert graph.active_edges(2.25)


def test_export_binned_edge_event_snapshots_writes_directed_counts_without_headers(tmp_path) -> None:
    events = parse_edge_events(["A,B,0", "A,B,1", "B,A,2", "C,B,10"])

    paths = export_binned_edge_event_snapshots(events, time_bins=2, output_dir=tmp_path / "snapshots")

    assert [path.name for path in paths] == ["snapshot_000.csv", "snapshot_001.csv"]
    assert paths[0].read_text(encoding="utf-8") == "A,B,2\nB,A,1\n"
    assert paths[1].read_text(encoding="utf-8") == "C,B,1\n"
