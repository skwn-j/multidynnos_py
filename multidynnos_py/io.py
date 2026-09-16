"""Temporal input adapters and the timelighting interval-string JSON format.

Java compatibility: interval CSV follows ``ocotillo.run.customrun``. JSON
adapters and format detection are Python additions; they never classify integer
timestamps as snapshots. See PORTING.md for deliberate differences.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.request import urlopen

from .model import Edge, Interval, Node, PositionSegment, TemporalGraph, merge_intervals

_INTERVAL = re.compile(r"^\s*([\[(])\s*([^,]+),\s*([^\])]+)\s*([\])])\s*$")
_POINT = re.compile(r"\(([^,()]+),\s*([^,()]+)\)")
_TYPES = {"timesliced": "timesliced", "time_sliced": "timesliced",
          "time-sliced": "timesliced", "snapshot": "timesliced",
          "snapshots": "timesliced", "event_based": "event",
          "event-based": "event", "event": "event", "events": "event"}


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric time or coordinate: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Time and coordinate values must be finite: {value!r}")
    return result


def parse_interval(value: str | list | tuple | Mapping) -> Interval:
    """Parse brackets without losing open/closed endpoint information."""
    if isinstance(value, str):
        match = _INTERVAL.match(value)
        if not match:
            raise ValueError(f"Invalid interval: {value!r}")
        left, start, end, right = match.groups()
        return Interval(_number(start), _number(end), left == "[", right == "]")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Interval(_number(value[0]), _number(value[1]))
    if isinstance(value, Mapping):
        start = _number(value.get("start", value.get("time", value.get("timestamp"))))
        if "end" in value:
            end = _number(value["end"])
        elif "duration" in value:
            end = start + _number(value["duration"])
        else:
            raise ValueError("An interval needs end or duration, not only a timestamp")
        return Interval(start, end, bool(value.get("left_closed", True)),
                        bool(value.get("right_closed", True)))
    raise ValueError(f"Invalid interval: {value!r}")


def _evolution_items(value: Any):
    """Accept both the actual timelighting array and a direct interval map."""
    if isinstance(value, Mapping):
        return list(value.items())
    result = []
    for item in value:
        if isinstance(item, Mapping) and "string" in item:
            parts = item["string"].split(" : ", 1)
            if len(parts) != 2:
                raise ValueError("Evolution strings must contain 'interval : value'")
            result.append(tuple(parts))
        elif isinstance(item, Mapping) and "interval" in item:
            result.append((item["interval"], item.get("value", "true,true")))
        elif isinstance(item, str):
            result.append((item, "true,true"))
        else:
            result.append((item, "true,true"))
    return result


def _presence(record: Mapping, fallback: list[Interval] | None = None,
              event_duration: float | None = None) -> list[Interval]:
    if "presence" in record:
        result = []
        for key, value in _evolution_items(record["presence"]):
            if value is False or str(value).lower().replace(" ", "") in {"false", "false,false"}:
                continue
            if value is not True and str(value).lower().replace(" ", "") not in {"true", "true,true"}:
                raise ValueError("Presence values must be constant true or false")
            result.append(parse_interval(key))
        return merge_intervals(result)
    if "intervals" in record:
        return merge_intervals([parse_interval(v) for v in record["intervals"]])
    if "start" in record or "time" in record or "timestamp" in record:
        timed = dict(record)
        if "end" not in timed and "duration" not in timed:
            if event_duration is None:
                raise ValueError("Timestamp events require an explicit event_duration")
            timed["duration"] = event_duration
        return [parse_interval(timed)]
    if fallback is not None:
        return list(fallback)
    raise ValueError("An event record needs presence, intervals, or start/end (or duration)")


def _positions(record: Mapping, intervals: list[Interval]) -> list[PositionSegment]:
    raw = record.get("position", record.get("positions"))
    if raw is None and "x" in record and "y" in record:
        raw = [record["x"], record["y"]]
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(isinstance(x, (int, float)) for x in raw):
        point = (_number(raw[0]), _number(raw[1]))
        return [PositionSegment(iv, point, point) for iv in intervals]
    result = []
    for key, value in _evolution_items(raw):
        iv = parse_interval(key)
        if isinstance(value, str):
            points = _POINT.findall(value)
        else:
            points = value
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError(f"Position needs two xy endpoints: {value!r}")
        result.append(PositionSegment(iv, tuple(map(_number, points[0])), tuple(map(_number, points[1]))))
    return result


def _normal_type(value: str) -> str:
    try:
        return _TYPES[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown data_type {value!r}; use timesliced or event_based") from exc


def detect_data_type(data: Mapping | list) -> str:
    """Classify a schema, never time magnitudes or integrality."""
    if isinstance(data, Mapping):
        explicit = data.get("data_type", data.get("type", data.get("metadata", {}).get("data_type")))
        if explicit is not None:
            return _normal_type(explicit)
        if "snapshots" in data or "timeslices" in data:
            return "timesliced"
        if any(k in data for k in ("events", "graphnodes", "graphedges", "edges", "links")):
            return "event"
        nodes = data.get("nodes", [])
        if isinstance(nodes, Mapping):
            nodes = nodes.values()
        if any(isinstance(node, Mapping) and any(key in node for key in
               ("presence", "intervals", "start", "time", "timestamp", "position", "positions"))
               for node in nodes):
            return "event"
    elif data and all(isinstance(x, Mapping) for x in data):
        if all(any(k in x for k in ("edges", "links", "nodes")) and
               any(k in x for k in ("time", "timestamp", "start")) for x in data):
            return "timesliced"
        if all("source" in x and "target" in x for x in data):
            return "event"
    raise ValueError("Cannot determine temporal schema; supply snapshots or event intervals and data_type")


def _node_record(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {"id": str(value)}


def _edge_record(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return {"source": value[0], "target": value[1]}
    raise ValueError(f"Invalid edge: {value!r}")


def _edge_key(record: Mapping, directed: bool):
    source, target = str(record["source"]), str(record["target"])
    if source == target:
        raise ValueError(f"Self loops are not supported by the Java layout: {source}")
    if not directed and source > target:
        source, target = target, source
    return ("id", str(record["id"])) if "id" in record else ("pair", source, target)


def _add_edge(edges: dict, record: Mapping, intervals: list[Interval], directed: bool):
    key = _edge_key(record, directed)
    source, target = str(record["source"]), str(record["target"])
    if key not in edges:
        edges[key] = Edge(str(record.get("id", f"{len(edges)}e")), source, target, [])
    edge = edges[key]
    same_pair = (edge.source, edge.target) == (source, target)
    reverse_pair = not directed and (edge.target, edge.source) == (source, target)
    if not (same_pair or reverse_pair):
        raise ValueError(f"Edge id {edge.id!r} refers to different endpoint pairs")
    edge.presence.extend(intervals)


def apply_presence_mode(graph: TemporalGraph, mode: str, end: float | None = None) -> TemporalGraph:
    """Port Commons.mergePresenceFunctions, excluding rendering-only colors."""
    if mode not in {"plain", "keepAppearedNode", "keepAppearedEdges"}:
        raise ValueError("presence_mode must be plain, keepAppearedNode, or keepAppearedEdges")
    all_intervals = [iv for n in graph.nodes.values() for iv in n.presence]
    all_intervals += [iv for e in graph.edges for iv in e.presence]
    if not all_intervals:
        raise ValueError("The input graph has no active temporal intervals")
    end = max(iv.end for iv in all_intervals) if end is None else end
    for node in graph.nodes.values():
        node.presence = merge_intervals(node.presence)
        if mode != "plain" and node.presence:
            node.presence = [Interval(node.presence[0].start, end)]
    for edge in graph.edges:
        edge.presence = merge_intervals(edge.presence)
        if mode == "keepAppearedEdges" and edge.presence:
            edge.presence = [Interval(edge.presence[0].start, end)]
    graph.metadata["presence_mode"] = mode
    graph.metadata["load_mode"] = mode
    return graph


def _events(data: Mapping | list, event_duration: float | None) -> TemporalGraph:
    data = {"events": data} if isinstance(data, list) else data
    if "snapshots" in data or "timeslices" in data:
        raise ValueError("Grouped snapshots require data_type='timesliced', not 'event'")
    directed = bool(data.get("directed", data.get("metadata", {}).get("directed", False)))
    raw_edges = data.get("events", data.get("edges", data.get("links", data.get("graphedges", []))))
    edges: dict = {}
    for value in raw_edges:
        record = _edge_record(value)
        _add_edge(edges, record, _presence(record, event_duration=event_duration), directed)
    raw_nodes = data.get("nodes", data.get("graphnodes", []))
    if isinstance(raw_nodes, Mapping):
        raw_nodes = [dict(v, id=k) if isinstance(v, Mapping) else {"id": k} for k, v in raw_nodes.items()]
    bounds = [iv for edge in edges.values() for iv in edge.presence]
    for value in raw_nodes:
        record = _node_record(value)
        if any(k in record for k in ("presence", "intervals", "start", "time", "timestamp")):
            bounds.extend(_presence(record, event_duration=event_duration))
        else:
            bounds.extend(seg.interval for seg in _positions(record, []))
    if not bounds:
        raise ValueError("Event data must contain at least one explicit interval")
    global_presence = [Interval(min(iv.start for iv in bounds), max(iv.end for iv in bounds))]
    nodes = {}
    explicit_ids = set()
    for value in raw_nodes:
        record = _node_record(value)
        node_id = str(record["id"])
        existing_positions = _positions(record, [])
        fallback = merge_intervals(seg.interval for seg in existing_positions) or global_presence
        intervals = _presence(record, fallback, event_duration)
        if node_id not in nodes:
            nodes[node_id] = Node(node_id, [], [], record.get("label"))
        nodes[node_id].presence.extend(intervals)
        nodes[node_id].positions.extend(_positions(record, intervals))
        explicit_ids.add(node_id)
    # When no node lifetime is supplied, incident edge intervals provide the
    # observable lifetime. Explicit node records without lifetimes mean all time.
    for edge in edges.values():
        for node_id in (edge.source, edge.target):
            if node_id not in nodes:
                nodes[node_id] = Node(node_id, [], [])
            if node_id not in explicit_ids:
                nodes[node_id].presence.extend(edge.presence)
    metadata = dict(data.get("metadata", {}))
    metadata["detection_reason"] = "explicit event/interval schema"
    return TemporalGraph(nodes, list(edges.values()), "event", directed, metadata)


def _snapshots(data: Mapping | list) -> TemporalGraph:
    data = {"snapshots": data} if isinstance(data, list) else data
    snapshots = data.get("snapshots", data.get("timeslices"))
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("timesliced data requires a nonempty snapshots list")
    times = data.get("times")
    if times is not None and len(times) != len(snapshots):
        raise ValueError("times must contain one time per snapshot")
    normalized = []
    for i, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            snapshot = {"edges": snapshot}
        snapshot = dict(snapshot)
        if "time" not in snapshot:
            snapshot["time"] = snapshot.get("timestamp", times[i] if times is not None
                                            else snapshot.get("start", i))
        normalized.append(snapshot)
    normalized.sort(key=lambda s: _number(s["time"]))
    ts = [_number(s["time"]) for s in normalized]
    if len(ts) != len(set(ts)):
        raise ValueError("Snapshot times must be distinct")
    directed = bool(data.get("directed", data.get("metadata", {}).get("directed", False)))
    nodes, edges = {}, {}
    global_nodes = data.get("nodes", [])
    for i, snapshot in enumerate(normalized):
        if "start" in snapshot or "end" in snapshot or "duration" in snapshot:
            interval = parse_interval(snapshot)
        elif len(ts) == 1:
            width = _number(data.get("slice_duration", 1.0))
            if width <= 0:
                raise ValueError("slice_duration must be positive")
            interval = Interval(ts[i] - width / 2, ts[i] + width / 2, False, True)
        else:
            left = (ts[i - 1] + ts[i]) / 2 if i else ts[0] - (ts[1] - ts[0]) / 2
            right = (ts[i] + ts[i + 1]) / 2 if i + 1 < len(ts) else ts[-1] + (ts[-1] - ts[-2]) / 2
            interval = Interval(left, right, False, True)
        records = [_node_record(n) for n in snapshot.get("nodes", global_nodes)]
        snapshot_edges = [_edge_record(e) for e in snapshot.get("edges", snapshot.get("links", []))]
        listed = {str(n["id"]) for n in records}
        for edge in snapshot_edges:
            for endpoint in (str(edge["source"]), str(edge["target"])):
                if endpoint not in listed:
                    records.append({"id": endpoint})
                    listed.add(endpoint)
            _add_edge(edges, edge, [interval], directed)
        for record in records:
            node_id = str(record["id"])
            if node_id not in nodes:
                nodes[node_id] = Node(node_id, [], [], record.get("label"))
            nodes[node_id].presence.append(interval)
            nodes[node_id].positions.extend(_positions(record, [interval]))
    metadata = dict(data.get("metadata", {}))
    metadata.update({"snapshot_times": ts, "detection_reason": "explicit snapshot grouping",
                     "slice_intervals": "explicit bounds or adjacent time midpoints, right-closed"})
    return TemporalGraph(nodes, list(edges.values()), "timesliced", directed, metadata)


def _read_text(source: str | Path) -> str:
    value = str(source)
    if value.startswith(("https://", "http://")):
        with urlopen(value, timeout=60) as response:
            return response.read().decode("utf-8-sig")
    return Path(source).read_text(encoding="utf-8-sig")


def _csv_records(text: str, nodes: bool = False) -> list[dict]:
    rows = [row for row in csv.reader(io.StringIO(text)) if row and not row[0].lstrip().startswith("#")]
    if not rows:
        return []
    header = [part.strip().lower() for part in rows[0]]
    is_header = (nodes and header[0] in {"id", "node", "nodeid", "node_id"}) or (
        not nodes and header[0] in {"source", "sourceid", "source_id"})
    if is_header:
        aliases = {"node": "id", "nodeid": "id", "node_id": "id", "sourceid": "source",
                   "source_id": "source", "targetid": "target", "target_id": "target",
                   "starttime": "start", "start_time": "start"}
        keys = [aliases.get(k, k) for k in header]
        rows = rows[1:]
    else:
        width = len(rows[0])
        keys = (["id", "start", "duration"] if nodes else
                ["source", "target", "time"] if width == 3 else
                ["source", "target", "start", "duration"])
    result = []
    for lineno, row in enumerate(rows, 2 if is_header else 1):
        if len(row) != len(keys):
            raise ValueError(f"CSV line {lineno}: expected {len(keys)} columns, got {len(row)}")
        result.append(dict(zip(keys, (item.strip() for item in row))))
    return result


def load_graph(source: str | Path | Mapping | list, data_type: str = "auto", *,
               nodes_path: str | Path | None = None, event_duration: float | None = None,
               presence_mode: str = "plain") -> TemporalGraph:
    """Load snapshot/interval JSON, timelighting JSON, or event CSV.

    Three-column timestamp CSV needs ``event_duration`` in the same time units.
    Four-column CSV follows Java's source,target,start,duration format. A separate
    ``nodes_path`` uses Java's id,start,duration format. Local directories/zip
    files are interpreted as the Newcomb matrix dataset.
    """
    if event_duration is not None and (not math.isfinite(event_duration) or event_duration <= 0):
        raise ValueError("event_duration must be finite and positive")
    if isinstance(source, (str, Path)) and (Path(str(source)).is_dir() or str(source).lower().endswith(".zip")):
        from .datasets import load_newcomb
        graph = load_newcomb(source, presence_mode=presence_mode)
        if data_type != "auto" and _normal_type(data_type) != graph.data_type:
            raise ValueError("Newcomb matrices are timesliced data")
        return graph
    if isinstance(source, (Mapping, list)):
        data = source
    else:
        content = _read_text(source)
        if content.lstrip().startswith(("{", "[")):
            data = json.loads(content)
        else:
            data = {"events": _csv_records(content)}
            if nodes_path is not None:
                data["nodes"] = _csv_records(_read_text(nodes_path), nodes=True)
    detected = detect_data_type(data) if data_type == "auto" else _normal_type(data_type)
    serialized_trajectories = isinstance(data, Mapping) and "graphnodes" in data
    graph = (_snapshots(data) if detected == "timesliced" and not serialized_trajectories
             else _events(data, event_duration))
    graph.data_type = detected
    graph = apply_presence_mode(graph, presence_mode)
    if isinstance(source, (str, Path)):
        graph.metadata["source"] = str(source)
    graph.validate()
    return graph


def _format_number(number: float) -> str:
    return repr(_number(number))


def format_interval(interval: Interval) -> str:
    return ("[" if interval.left_closed else "(") + _format_number(interval.start) + "," + \
        _format_number(interval.end) + ("]" if interval.right_closed else ")")


def _pack_evolution(items: list[tuple[str, str]], interval_maps: bool):
    if interval_maps:
        return dict(items)
    return [{"string": f"{key} : {value}"} for key, value in items]


def to_timelighting(graph: TemporalGraph, *, interval_maps: bool = False,
                    include_metadata: bool = True) -> dict:
    """Serialize xy trajectories; interval endpoints supply the cube's time axis.

    The default array of ``{'string': 'interval : value'}`` is the exact schema
    of the referenced newcomb.json. ``interval_maps=True`` emits direct interval
    keys instead. Absent edge periods are never emitted.
    """
    graph.validate()
    graphnodes = []
    for node in graph.nodes.values():
        if not node.positions:
            raise ValueError(f"Node {node.id!r} has no trajectory; run the layout before export")
        items = []
        covered = []
        for segment in sorted(node.positions, key=lambda s: s.interval.start):
            for presence in node.presence:
                active = segment.interval.intersection(presence)
                if active is None:
                    continue
                start, end = segment.at(active.start), segment.at(active.end)
                coords = f"({_format_number(start[0])}, {_format_number(start[1])})," + \
                         f"({_format_number(end[0])}, {_format_number(end[1])})"
                items.append((format_interval(active), coords))
                covered.append(active)
        if merge_intervals(covered) != node.presence:
            raise ValueError(f"Trajectory of node {node.id!r} does not cover its full presence")
        graphnodes.append({"id": node.id, "position": _pack_evolution(items, interval_maps)})
    graphedges = []
    for edge in graph.edges:
        intervals = merge_intervals(edge.presence)
        if not intervals:
            continue
        graphedges.append({"id": edge.id, "source": edge.source, "target": edge.target,
                          "presence": _pack_evolution([(format_interval(iv), "true,true")
                                                      for iv in intervals], interval_maps)})
    result = {"graphnodes": graphnodes, "graphedges": graphedges}
    if include_metadata:
        result["metadata"] = {**graph.metadata, "data_type": graph.data_type, "directed": graph.directed,
                              "coordinate_system": "(x,y,time); time is the interval endpoint",
                              "format": "interval-map" if interval_maps else "timelighting"}
    return result


def save_graph(graph: TemporalGraph, path: str | Path, *, interval_maps: bool = False,
               include_metadata: bool = True) -> Path:
    path = Path(path)
    payload = to_timelighting(graph, interval_maps=interval_maps, include_metadata=include_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
