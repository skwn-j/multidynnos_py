"""Readers for edge-event files with source, target, timestamp rows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from multidynnos_py.data.graph import DynamicGraph, canonical_edge_key
from multidynnos_py.data.temporal import Interval

HeaderIndices = dict[str, int]

_SOURCE_HEADERS = {"src", "source", "sourceid", "source_id", "sourcenode", "sourcenodeid", "from", "u"}
_TARGET_HEADERS = {
    "dst",
    "dest",
    "destination",
    "target",
    "targetid",
    "target_id",
    "targetnode",
    "targetnodeid",
    "to",
    "v",
}
_TIME_HEADERS = {"timestamp", "time", "t", "ts"}


@dataclass(frozen=True, slots=True)
class EdgeEvent:
    """A timestamped interaction between two nodes."""

    source_id: str
    target_id: str
    timestamp: float

    def interval(self, duration: float = 0.0) -> Interval:
        return Interval.from_start_duration(self.timestamp, duration)


def parse_edge_event_line(line: str) -> EdgeEvent:
    """Parse one ``src,dst,timestamp`` or whitespace-delimited event row."""

    return _parse_edge_event_fields(_split_event_line(line), line)


def parse_edge_events(
    lines: Iterable[str],
    *,
    skip_invalid: bool = False,
    has_header: bool | None = None,
) -> list[EdgeEvent]:
    """Parse timestamped edge events from comma, tab, or whitespace rows."""

    events: list[EdgeEvent] = []
    header_checked = False
    header_indices: HeaderIndices | None = None
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
            fields = _split_event_line(stripped)
            if has_header is True:
                header_indices = _header_indices(fields)
                continue
            if has_header is None:
                header_indices = _optional_header_indices(fields)
                if header_indices is not None:
                    continue

        try:
            events.append(_parse_edge_event_fields(_split_event_line(stripped), stripped, header_indices))
        except ValueError as exc:
            if not skip_invalid:
                raise ValueError(f"Invalid edge event on line {line_number}: {exc}") from exc
    return events


def parse_edge_events_from_csv(
    rows: Iterable[dict[str, str]],
    *,
    skip_invalid: bool = False,
) -> list[EdgeEvent]:
    """Parse events from ``csv.DictReader``-style rows."""

    events: list[EdgeEvent] = []
    for row_number, row in enumerate(rows, start=2):
        normalized = {_normalize_header_field(key): value for key, value in row.items()}
        try:
            source_id = _lookup_header_value(normalized, _SOURCE_HEADERS)
            target_id = _lookup_header_value(normalized, _TARGET_HEADERS)
            timestamp = _lookup_header_value(normalized, _TIME_HEADERS)
            events.append(_make_edge_event(source_id, target_id, timestamp, repr(row)))
        except ValueError as exc:
            if not skip_invalid:
                raise ValueError(f"Invalid edge event on CSV row {row_number}: {exc}") from exc
    return events


def _parse_edge_event_fields(
    fields: list[str],
    line: str,
    header_indices: HeaderIndices | None = None,
) -> EdgeEvent:
    if header_indices is None:
        if len(fields) != 3:
            raise ValueError(f"Expected 3 fields: source, target, timestamp; got {len(fields)}: {line!r}")
        source_id, target_id, timestamp = fields
    else:
        try:
            source_id = fields[header_indices["source"]]
            target_id = fields[header_indices["target"]]
            timestamp = fields[header_indices["timestamp"]]
        except IndexError as exc:
            raise ValueError(f"Row does not contain all header-defined columns: {line!r}") from exc
    return _make_edge_event(source_id, target_id, timestamp, line)


def _make_edge_event(source_id: str, target_id: str, timestamp: str, row_repr: str) -> EdgeEvent:
    source_id = source_id.strip()
    target_id = target_id.strip()
    if not source_id or not target_id:
        raise ValueError(f"Source and target IDs cannot be empty: {row_repr!r}")
    return EdgeEvent(source_id=source_id, target_id=target_id, timestamp=float(timestamp))


def _header_indices(fields: list[str]) -> HeaderIndices:
    indices = _optional_header_indices(fields)
    if indices is None:
        raise ValueError(
            "Header must include source, target, and timestamp columns. "
            f"Accepted source aliases: {sorted(_SOURCE_HEADERS)}; "
            f"target aliases: {sorted(_TARGET_HEADERS)}; "
            f"timestamp aliases: {sorted(_TIME_HEADERS)}."
        )
    return indices


def _optional_header_indices(fields: list[str]) -> HeaderIndices | None:
    normalized = [_normalize_header_field(field) for field in fields]
    source_index = _find_header_index(normalized, _SOURCE_HEADERS)
    target_index = _find_header_index(normalized, _TARGET_HEADERS)
    timestamp_index = _find_header_index(normalized, _TIME_HEADERS)
    if source_index is None or target_index is None or timestamp_index is None:
        return None
    return {
        "source": source_index,
        "target": target_index,
        "timestamp": timestamp_index,
    }


def _find_header_index(normalized_fields: list[str], aliases: set[str]) -> int | None:
    for index, field in enumerate(normalized_fields):
        if field in aliases:
            return index
    return None


def _lookup_header_value(row: dict[str, str], aliases: set[str]) -> str:
    for alias in aliases:
        if alias in row and row[alias] is not None:
            return row[alias]
    raise ValueError(f"Missing required column; accepted aliases are {sorted(aliases)}.")


def build_graph_from_edge_events(
    edge_events: Iterable[EdgeEvent],
    *,
    event_duration: float = 0.0,
    time_bins: int | None = None,
) -> DynamicGraph:
    """Build a dynamic graph from timestamped edge events."""

    if event_duration < 0.0:
        raise ValueError("event_duration must be non-negative.")
    events = list(edge_events)
    if time_bins is not None:
        return build_binned_graph_from_edge_events(events, time_bins=time_bins)

    graph = DynamicGraph()
    for event in events:
        interval = event.interval(event_duration)
        graph.add_node_presence(event.source_id, interval)
        graph.add_node_presence(event.target_id, interval)
        graph.add_edge_presence(event.source_id, event.target_id, interval)
    return graph


def export_binned_edge_event_snapshots(
    edge_events: Iterable[EdgeEvent],
    *,
    time_bins: int,
    output_dir: str | Path,
) -> list[Path]:
    """Write one headerless ``src,dst,count`` CSV per temporal bin."""

    if time_bins <= 0:
        raise ValueError("time_bins must be a positive integer.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    events = list(edge_events)
    edge_bins: dict[int, dict[tuple[str, str], int]] = {index: {} for index in range(time_bins)}

    if events:
        start_time = min(event.timestamp for event in events)
        end_time = max(event.timestamp for event in events)
        for event in events:
            bin_index = _time_bin_index(event.timestamp, start_time, end_time, time_bins)
            directed_key = (event.source_id, event.target_id)
            edge_bins[bin_index][directed_key] = edge_bins[bin_index].get(directed_key, 0) + 1

    written_paths: list[Path] = []
    for bin_index in range(time_bins):
        csv_path = output_path / f"snapshot_{bin_index:03d}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for (source_id, target_id), count in sorted(edge_bins[bin_index].items()):
                writer.writerow([source_id, target_id, str(count)])
        written_paths.append(csv_path)
    return written_paths


def build_binned_graph_from_edge_events(
    edge_events: Iterable[EdgeEvent],
    *,
    time_bins: int,
) -> DynamicGraph:
    """Build a dynamic graph after aggregating events into uniform time bins."""

    if time_bins <= 0:
        raise ValueError("time_bins must be a positive integer.")

    events = list(edge_events)
    graph = DynamicGraph()
    if not events:
        return graph

    start_time = min(event.timestamp for event in events)
    end_time = max(event.timestamp for event in events)
    boundaries = _time_bin_boundaries(start_time, end_time, time_bins)
    node_bins: dict[str, set[int]] = {}
    edge_bins: dict[tuple[str, str], set[int]] = {}

    for event in events:
        bin_index = _time_bin_index(event.timestamp, start_time, end_time, time_bins)
        node_bins.setdefault(event.source_id, set()).add(bin_index)
        node_bins.setdefault(event.target_id, set()).add(bin_index)
        edge_bins.setdefault(canonical_edge_key(event.source_id, event.target_id), set()).add(bin_index)

    for node_id, bins in sorted(node_bins.items()):
        for bin_index in sorted(bins):
            graph.add_node_presence(node_id, Interval.closed(boundaries[bin_index], boundaries[bin_index + 1]))

    for (source_id, target_id), bins in sorted(edge_bins.items()):
        for bin_index in sorted(bins):
            graph.add_edge_presence(
                source_id,
                target_id,
                Interval.closed(boundaries[bin_index], boundaries[bin_index + 1]),
            )

    return graph


def load_edge_event_graph(
    event_file: str | Path,
    *,
    event_duration: float = 0.0,
    skip_invalid: bool = False,
    has_header: bool | None = None,
    time_bins: int | None = None,
) -> DynamicGraph:
    """Load an edge-event graph from a ``src,dst,timestamp`` text or CSV file."""

    path = Path(event_file)
    events = parse_edge_events(
        path.read_text(encoding="utf-8").splitlines(),
        skip_invalid=skip_invalid,
        has_header=has_header,
    )
    return build_graph_from_edge_events(events, event_duration=event_duration, time_bins=time_bins)


def load_edge_events(
    event_file: str | Path,
    *,
    skip_invalid: bool = False,
    has_header: bool | None = None,
) -> list[EdgeEvent]:
    """Load timestamped edge events without constructing a graph."""

    path = Path(event_file)
    return parse_edge_events(
        path.read_text(encoding="utf-8").splitlines(),
        skip_invalid=skip_invalid,
        has_header=has_header,
    )


def _split_event_line(line: str) -> list[str]:
    stripped = line.strip()
    if "," in stripped:
        return [field.strip() for field in next(csv.reader([stripped]))]
    return stripped.split()


def _normalize_header_field(field: str) -> str:
    return field.strip().lstrip("\ufeff").lower().replace(" ", "").replace("-", "_")


def _time_bin_boundaries(start_time: float, end_time: float, time_bins: int) -> list[float]:
    if start_time == end_time:
        return [start_time for _ in range(time_bins + 1)]
    step = (end_time - start_time) / time_bins
    return [start_time + step * index for index in range(time_bins)] + [end_time]


def _time_bin_index(timestamp: float, start_time: float, end_time: float, time_bins: int) -> int:
    if start_time == end_time:
        return 0
    raw_index = int(((timestamp - start_time) / (end_time - start_time)) * time_bins)
    return min(max(raw_index, 0), time_bins - 1)


def _format_float(value: float) -> str:
    return f"{value:g}"
