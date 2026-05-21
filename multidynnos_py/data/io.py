"""Input and export helpers for custom MultiDynNoS dynamic graphs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.data.temporal import Interval


@dataclass(frozen=True, slots=True)
class NodeAppearance:
    """A parsed node appearance row."""

    node_id: str
    start_time: float
    duration: float

    @property
    def interval(self) -> Interval:
        return Interval.from_start_duration(self.start_time, self.duration)


@dataclass(frozen=True, slots=True)
class EdgeAppearance:
    """A parsed edge appearance row."""

    source_id: str
    target_id: str
    start_time: float
    duration: float

    @property
    def interval(self) -> Interval:
        return Interval.from_start_duration(self.start_time, self.duration)


def _split_custom_line(line: str, expected_fields: int) -> list[str]:
    stripped = line.strip()
    if not stripped:
        raise ValueError("Blank lines do not contain an appearance row.")
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) != expected_fields:
        raise ValueError(f"Expected {expected_fields} comma-separated fields, got {len(fields)}: {line!r}")
    if any(field == "" for field in fields):
        raise ValueError(f"Appearance rows cannot contain empty fields: {line!r}")
    return fields


def parse_node_appearance_line(line: str) -> NodeAppearance:
    node_id, start_time, duration = _split_custom_line(line, 3)
    return NodeAppearance(
        node_id=node_id,
        start_time=float(start_time),
        duration=float(duration),
    )


def parse_edge_appearance_line(line: str) -> EdgeAppearance:
    source_id, target_id, start_time, duration = _split_custom_line(line, 4)
    return EdgeAppearance(
        source_id=source_id,
        target_id=target_id,
        start_time=float(start_time),
        duration=float(duration),
    )


def parse_node_appearances(lines: Iterable[str], *, skip_invalid: bool = False) -> list[NodeAppearance]:
    appearances: list[NodeAppearance] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            appearances.append(parse_node_appearance_line(line))
        except ValueError as exc:
            if not skip_invalid:
                raise ValueError(f"Invalid node appearance on line {line_number}: {exc}") from exc
    return appearances


def parse_edge_appearances(lines: Iterable[str], *, skip_invalid: bool = False) -> list[EdgeAppearance]:
    appearances: list[EdgeAppearance] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            appearances.append(parse_edge_appearance_line(line))
        except ValueError as exc:
            if not skip_invalid:
                raise ValueError(f"Invalid edge appearance on line {line_number}: {exc}") from exc
    return appearances


def build_graph_from_appearances(
    node_appearances: Iterable[NodeAppearance],
    edge_appearances: Iterable[EdgeAppearance],
) -> DynamicGraph:
    graph = DynamicGraph()

    for appearance in node_appearances:
        graph.add_node_presence(appearance.node_id, appearance.interval)

    for appearance in edge_appearances:
        graph.add_edge_presence(
            appearance.source_id,
            appearance.target_id,
            appearance.interval,
        )

    return graph


def load_custom_graph(
    node_file: str | Path,
    edge_file: str | Path,
    *,
    skip_invalid: bool = False,
) -> DynamicGraph:
    node_path = Path(node_file)
    edge_path = Path(edge_file)
    node_appearances = parse_node_appearances(node_path.read_text().splitlines(), skip_invalid=skip_invalid)
    edge_appearances = parse_edge_appearances(edge_path.read_text().splitlines(), skip_invalid=skip_invalid)
    return build_graph_from_appearances(node_appearances, edge_appearances)


def graph_to_json(graph: DynamicGraph, *, indent: int = 2) -> str:
    return json.dumps(graph.to_dict(), indent=indent, sort_keys=True)


def export_graph_json(graph: DynamicGraph, destination: str | Path | TextIO, *, indent: int = 2) -> None:
    payload = graph_to_json(graph, indent=indent)
    if hasattr(destination, "write"):
        destination.write(payload)
        destination.write("\n")
        return
    Path(destination).write_text(payload + "\n")


def iter_node_csv_rows(graph: DynamicGraph) -> Iterable[list[str]]:
    for node in sorted(graph.nodes.values(), key=lambda item: item.id):
        for interval in node.appearances:
            start, duration = interval.as_start_duration()
            yield [node.id, _format_float(start), _format_float(duration)]


def iter_edge_csv_rows(graph: DynamicGraph, *, include_edge_ids: bool = False) -> Iterable[list[str]]:
    for edge in sorted(graph.edges.values(), key=lambda item: item.id):
        for interval in edge.appearances:
            start, duration = interval.as_start_duration()
            row = [edge.source, edge.target, _format_float(start), _format_float(duration)]
            if include_edge_ids:
                row.append(edge.id)
            yield row


def export_graph_csv(
    graph: DynamicGraph,
    node_destination: str | Path | TextIO,
    edge_destination: str | Path | TextIO,
    *,
    include_edge_ids: bool = False,
) -> None:
    _write_csv_rows(node_destination, iter_node_csv_rows(graph))
    _write_csv_rows(edge_destination, iter_edge_csv_rows(graph, include_edge_ids=include_edge_ids))


def _write_csv_rows(destination: str | Path | TextIO, rows: Iterable[list[str]]) -> None:
    if hasattr(destination, "write"):
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerows(rows)
        return
    with Path(destination).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return f"{value:g}"

