"""Display saved temporal layouts with the repository's HTML snapshot viewer."""

from __future__ import annotations

import html
import json
from math import fsum
from numbers import Integral
from pathlib import Path
import re

import numpy as np

from .io import load_graph
from .model import Interval, TemporalGraph, merge_intervals


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _snapshot(graph: TemporalGraph, window: Interval, number: int) -> dict:
    ids, xy = [], []
    for node in graph.nodes.values():
        active = [overlap for presence in node.presence
                  if (overlap := presence.intersection(window)) is not None]
        if not active:
            continue
        pieces = [(segment, overlap) for segment in node.positions for presence in active
                  if (overlap := segment.interval.intersection(presence)) is not None]
        if merge_intervals(overlap for _, overlap in pieces) != merge_intervals(active):
            raise ValueError(f"Node {node.id!r} has missing trajectory coverage in {window}")
        duration = fsum(overlap.duration for _, overlap in pieces)
        if duration > 0:
            position = np.sum([
                (segment.at(overlap.start) + segment.at(overlap.end)) * 0.5
                * (overlap.duration / duration)
                for segment, overlap in pieces if overlap.duration > 0
            ], axis=0)
        else:
            points = {overlap.start: segment.at(overlap.start) for segment, overlap in pieces}
            position = np.mean(list(points.values()), axis=0)
        ids.append(node.id)
        xy.append(position.tolist())
    indexes = {node_id: i for i, node_id in enumerate(ids)}
    edges = [[indexes[edge.source], indexes[edge.target]] for edge in graph.edges
             if edge.source in indexes and edge.target in indexes
             and any(presence.intersection(window) is not None for presence in edge.presence)]
    return {
        "number": number, "start": window.start, "end": window.end,
        "interval": str(window), "ids": ids, "xy": xy, "edges": edges,
        "nodes": len(ids), "edge_count": len(edges),
    }


def build_snapshot_data(graph: TemporalGraph, n_columns: int, visible_columns: int,
                        *, title: str = "Snapshots", source: str = "") -> dict:
    """Aggregate equal time windows, preserving gaps, boundaries, and stored XY."""
    count = _positive_count(n_columns, "N_COLUMNS")
    visible = _positive_count(visible_columns, "VISIBLE_COLUMNS")
    graph.validate()
    domain = graph.interval
    if domain.duration == 0:
        windows = [domain]
    else:
        boundaries = np.linspace(domain.start, domain.end, count + 1)
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("N_COLUMNS is too large for the time range's numeric precision")
        windows = [Interval(float(left), float(right), i == 0, True)
                   for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    snapshots = [_snapshot(graph, window, i + 1) for i, window in enumerate(windows)]
    points = np.asarray([point for node in graph.nodes.values() for segment in node.positions
                         for point in (segment.start, segment.end)], dtype=float)
    lower, upper = points.min(axis=0), points.max(axis=0)
    return {
        "title": title, "source": source, "directed": graph.directed,
        "snapshot_count": len(snapshots), "visible_columns": min(visible, len(snapshots)),
        "bounds": {"x": [float(lower[0]), float(upper[0])],
                   "y": [float(lower[1]), float(upper[1])]},
        "snapshots": snapshots,
    }


def render_viewer(data: dict, viewer_path: str | Path) -> str:
    """Embed a single dataset in the offline HTML template."""
    template = Path(viewer_path).read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # JSON remains inert even when node IDs or filenames contain HTML characters.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    document, count = re.subn(
        r'(<script id="data" type="application/json">).*?(</script>)',
        lambda match: match[1] + payload + match[2], template, flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("viewer.html must contain exactly one JSON data script")
    return document


def show_snapshots(json_path: str | Path, n_columns: int = 12, visible_columns: int = 4,
                   *, viewer_path: str | Path | None = None):
    """Save a standalone viewer and return an inline notebook iframe."""
    from IPython.display import IFrame

    template = (Path(viewer_path).expanduser().resolve() if viewer_path is not None
                else Path(__file__).resolve().parents[1] / "viewer.html")
    source = Path(json_path).expanduser()
    if not source.is_absolute():
        source = template.parent / source
    graph = load_graph(source, presence_mode="plain")
    data = build_snapshot_data(graph, n_columns, visible_columns,
                               title=source.name, source=str(json_path))
    document = render_viewer(data, template)
    output = template.parent / "output" / "snapshot_previews" / f"{source.stem}_viewer.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return IFrame(
        src="about:blank", width="100%", height=760,
        extras=['title="Temporal graph snapshots"', 'sandbox="allow-scripts"',
                'style="border:0;display:block"',
                f'srcdoc="{html.escape(document, quote=True)}"'],
    )
