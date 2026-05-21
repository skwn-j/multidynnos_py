"""Render static node-link snapshots from a generated layout JSON."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multidynnos_py.experiments.snapshot_windows import SnapshotWindow, compute_snapshot_windows

Position = tuple[float, float]
TrajectoryRecord = dict[str, float]

_SOURCE_HEADERS = {"src", "source", "sourceid", "source_id", "from", "u"}
_TARGET_HEADERS = {"dst", "dest", "destination", "target", "targetid", "target_id", "to", "v"}
_START_HEADERS = {"start", "start_time", "starttime", "begin", "begin_time"}
_END_HEADERS = {"end", "end_time", "endtime", "stop", "stop_time"}
_DURATION_HEADERS = {"duration", "dur", "dt"}
_TIME_HEADERS = {"time", "timestamp", "t", "ts"}
_WEIGHT_HEADERS = {"weight", "w"}
_ID_HEADERS = {"id", "edge_id", "edgeid"}


@dataclass(frozen=True, slots=True)
class Bounds:
    """Axis-aligned plotting bounds."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def padded(self, fraction: float = 0.05) -> Bounds:
        width = self.width
        height = self.height
        if width == 0.0:
            width = 1.0
        if height == 0.0:
            height = 1.0
        pad_x = width * fraction
        pad_y = height * fraction
        return Bounds(
            min_x=self.min_x - pad_x,
            min_y=self.min_y - pad_y,
            max_x=self.max_x + pad_x,
            max_y=self.max_y + pad_y,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """A static, point-event, or interval edge used for snapshot rendering."""

    source: str
    target: str
    start: float | None = None
    end: float | None = None
    time: float | None = None
    weight: float | None = None
    id: str | None = None

    @property
    def is_static(self) -> bool:
        return self.start is None and self.end is None and self.time is None

    @property
    def is_point(self) -> bool:
        return self.time is not None

    @property
    def is_interval(self) -> bool:
        return self.start is not None and self.end is not None

    def active_in(self, window: SnapshotWindow) -> bool:
        if self.is_static:
            return True
        if self.time is not None:
            return window.contains(self.time)
        if self.start is None or self.end is None:
            return False
        return window.overlaps_interval(self.start, self.end)


@dataclass(frozen=True, slots=True)
class SnapshotRenderResult:
    """Metadata for one rendered snapshot."""

    index: int
    window_start: float
    window_end: float
    window_midpoint: float
    window_start_iso: str
    window_end_iso: str
    window_midpoint_iso: str
    num_nodes: int
    num_edges: int
    figure_path: str
    skipped_edges: int = 0
    warning: str | None = None
    node_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_layout_json(path: str | Path) -> dict[str, Any]:
    """Load and validate a layout JSON file."""

    layout_path = Path(path)
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file does not exist: {layout_path}")
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    if "sample_times" not in data or not isinstance(data["sample_times"], list):
        raise ValueError("Layout JSON must contain a 'sample_times' list.")
    if "trajectories" not in data or not isinstance(data["trajectories"], dict):
        raise ValueError("Layout JSON must contain a 'trajectories' object.")
    return data


def infer_dataset_name(layout_path: str | Path, override: str | None = None) -> str:
    """Infer a dataset name from a layout filename unless an override is supplied."""

    if override:
        return override
    path = Path(layout_path)
    name = path.name
    if name == "layout.json" and path.parent.name:
        return path.parent.name
    if name.endswith("_multidynnos_layout.json"):
        return name[: -len("_multidynnos_layout.json")]
    if name.endswith("_layout.json"):
        return name[: -len("_layout.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return Path(layout_path).stem


def sample_node_positions_for_window(
    trajectories: dict[str, list[TrajectoryRecord]],
    window: SnapshotWindow,
    position_policy: str = "mean_in_window",
    active_node_policy: str = "in_window",
    allowed_nodes: set[str] | None = None,
) -> dict[str, Position]:
    """Return node positions for one temporal window."""

    if active_node_policy not in {"in_window", "all", "midpoint_available"}:
        raise ValueError(f"Unsupported active_node_policy: {active_node_policy}")
    if position_policy not in {"midpoint_interpolate", "mean_in_window", "nearest_midpoint"}:
        raise ValueError(f"Unsupported position_policy: {position_policy}")

    positions: dict[str, Position] = {}
    for node_id, trajectory in trajectories.items():
        samples = _sorted_samples(trajectory)
        if not samples:
            continue
        samples_in_window = [sample for sample in samples if window.contains(float(sample["time"]))]
        midpoint_position = _interpolate_at(samples, window.midpoint)

        if active_node_policy == "in_window" and not samples_in_window:
            continue
        if active_node_policy == "midpoint_available" and midpoint_position is None:
            continue

        position = _position_for_policy(samples, samples_in_window, window, position_policy, midpoint_position)
        if position is not None:
            positions[node_id] = position
    if allowed_nodes is not None:
        return {node_id: position for node_id, position in positions.items() if node_id in allowed_nodes}
    return positions


def load_edges_csv(path: str | Path) -> list[EdgeRecord]:
    """Load flexible edge CSV schemas for snapshot rendering."""

    edge_path = Path(path)
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge CSV does not exist: {edge_path}")

    rows = [row for row in csv.reader(edge_path.read_text(encoding="utf-8").splitlines()) if row]
    if not rows:
        return []

    first = [_normalize_header(field) for field in rows[0]]
    if _has_source_target_header(first):
        return [_edge_from_header_row(rows[0], row, row_number=index + 2) for index, row in enumerate(rows[1:])]
    return [_edge_from_unheadered_row(row, row_number=index + 1) for index, row in enumerate(rows)]


def active_edges_for_window(
    edges: Sequence[EdgeRecord],
    window: SnapshotWindow,
    available_nodes: set[str],
) -> list[EdgeRecord]:
    """Return edges active in a window whose endpoints have visible positions."""

    return [
        edge
        for edge in edges
        if edge.active_in(window) and edge.source in available_nodes and edge.target in available_nodes
    ]


def render_snapshot_node_link(
    positions: dict[str, Position],
    edges: Sequence[EdgeRecord],
    output_path: str | Path,
    title: str | None = None,
    global_bounds: Bounds | None = None,
    show_labels: bool = False,
    node_size: float = 40,
    edge_width: float = 0.8,
    dpi: int = 200,
    figure_size: tuple[float, float] = (6, 6),
) -> None:
    """Render one static node-link diagram using existing layout coordinates."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    if title:
        ax.set_title(title, fontsize=9)

    for edge in edges:
        source = positions.get(edge.source)
        target = positions.get(edge.target)
        if source is None or target is None:
            continue
        width = edge_width if edge.weight is None else edge_width * max(edge.weight, 0.1)
        ax.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color="#9aa0a6",
            linewidth=width,
            alpha=0.55,
            zorder=1,
        )

    if positions:
        node_ids = sorted(positions)
        xs = [positions[node_id][0] for node_id in node_ids]
        ys = [positions[node_id][1] for node_id in node_ids]
        ax.scatter(xs, ys, s=node_size, color="#2f6fed", edgecolors="white", linewidths=0.4, zorder=2)
        if show_labels:
            for node_id in node_ids:
                x, y = positions[node_id]
                ax.annotate(
                    node_id,
                    xy=(x, y),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=5,
                    ha="left",
                    va="bottom",
                    color="#202124",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
                    zorder=3,
                )

    bounds = global_bounds or compute_bounds_from_positions(positions)
    if bounds is not None:
        ax.set_xlim(bounds.min_x, bounds.max_x)
        ax.set_ylim(bounds.min_y, bounds.max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output)
    plt.close(fig)


def render_layout_snapshots(
    layout_path: str | Path,
    n_snapshots: int,
    edges_path: str | Path | None = None,
    snapshot_nodes_dir: str | Path | None = None,
    dataset_name: str | None = None,
    output_root: str | Path = "output",
    image_format: str = "png",
    allow_nodes_only: bool = False,
    show_labels: bool = False,
    active_node_policy: str = "in_window",
    position_policy: str = "mean_in_window",
    dpi: int = 200,
    figure_size: tuple[float, float] = (6, 6),
    title_format: str = "none",
    timezone: str = "UTC",
) -> list[SnapshotRenderResult]:
    """Render uniformly sampled static snapshots from one layout JSON."""

    if image_format not in {"png", "svg", "pdf"}:
        raise ValueError(f"Unsupported image format: {image_format}")
    layout = load_layout_json(layout_path)
    windows = compute_snapshot_windows(layout["sample_times"], n_snapshots)
    trajectories = layout["trajectories"]
    dataset = infer_dataset_name(layout_path, dataset_name)
    figures_dir = Path(output_root) / dataset / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    effective_snapshot_nodes_dir = _resolve_snapshot_nodes_dir(layout_path, snapshot_nodes_dir, output_root, dataset, n_snapshots)
    snapshot_node_sets = (
        _load_snapshot_node_sets(effective_snapshot_nodes_dir, n_snapshots)
        if effective_snapshot_nodes_dir is not None
        else None
    )

    edges = _edges_from_layout(layout)
    if edges_path is not None:
        edges = load_edges_csv(edges_path)
    node_only = False
    if not edges:
        if not allow_nodes_only:
            raise ValueError("No edge information found. Provide --edges or use --allow-nodes-only.")
        node_only = True

    global_bounds = compute_bounds_from_trajectories(trajectories).padded(0.05)
    results: list[SnapshotRenderResult] = []
    for window in windows:
        positions = sample_node_positions_for_window(
            trajectories,
            window,
            position_policy=position_policy,
            active_node_policy=active_node_policy,
            allowed_nodes=snapshot_node_sets[window.index] if snapshot_node_sets is not None else None,
        )
        active_edges = [] if node_only else active_edges_for_window(edges, window, set(positions))
        skipped_edges = 0 if node_only else _count_skipped_edges(edges, window, set(positions))
        figure_path = figures_dir / f"snapshot_{window.index:03d}.{image_format}"
        warning = "No active nodes in this snapshot." if not positions else None
        title = _title_for_window(window, title_format=title_format, timezone=timezone)

        render_snapshot_node_link(
            positions,
            active_edges,
            figure_path,
            title=title,
            global_bounds=global_bounds,
            show_labels=show_labels,
            dpi=dpi,
            figure_size=figure_size,
        )
        results.append(
            SnapshotRenderResult(
                index=window.index,
                window_start=window.start,
                window_end=window.end,
                window_midpoint=window.midpoint,
                window_start_iso=_to_iso(window.start, timezone),
                window_end_iso=_to_iso(window.end, timezone),
                window_midpoint_iso=_to_iso(window.midpoint, timezone),
                num_nodes=len(positions),
                num_edges=len(active_edges),
                figure_path=str(figure_path),
                skipped_edges=skipped_edges,
                warning=warning,
                node_only=node_only,
            )
        )

    manifest = {
        "dataset_name": dataset,
        "layout_path": str(layout_path),
        "edges_path": str(edges_path) if edges_path is not None else None,
        "snapshot_nodes_dir": str(effective_snapshot_nodes_dir) if effective_snapshot_nodes_dir is not None else None,
        "n_snapshots": n_snapshots,
        "node_only": node_only,
        "active_node_policy": active_node_policy,
        "position_policy": position_policy,
        "show_labels": show_labels,
        "time_range": {
            "start": min(float(time) for time in layout["sample_times"]),
            "end": max(float(time) for time in layout["sample_times"]),
            "start_iso": _to_iso(min(float(time) for time in layout["sample_times"]), timezone),
            "end_iso": _to_iso(max(float(time) for time in layout["sample_times"]), timezone),
        },
        "snapshots": [result.to_dict() for result in results],
    }
    (figures_dir / "snapshots_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def load_snapshot_csv_node_ids(path: str | Path) -> set[str]:
    """Return node IDs that occur as source or target in one headerless snapshot CSV."""

    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot CSV does not exist: {snapshot_path}")

    node_ids: set[str] = set()
    rows = csv.reader(snapshot_path.read_text(encoding="utf-8").splitlines())
    for row_number, row in enumerate(rows, start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) < 2:
            raise ValueError(
                f"Invalid snapshot CSV row {row_number} in {snapshot_path}: expected at least source,target."
            )
        source = row[0].strip()
        target = row[1].strip()
        if not source or not target:
            raise ValueError(
                f"Invalid snapshot CSV row {row_number} in {snapshot_path}: source and target cannot be empty."
            )
        node_ids.add(source)
        node_ids.add(target)
    return node_ids


def compute_bounds_from_positions(positions: dict[str, Position]) -> Bounds | None:
    if not positions:
        return None
    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys)).padded(0.05)


def compute_bounds_from_trajectories(trajectories: dict[str, list[TrajectoryRecord]]) -> Bounds:
    points = [
        (float(point["x"]), float(point["y"]))
        for trajectory in trajectories.values()
        for point in trajectory
    ]
    if not points:
        raise ValueError("Layout trajectories contain no points to render.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _position_for_policy(
    samples: list[TrajectoryRecord],
    samples_in_window: list[TrajectoryRecord],
    window: SnapshotWindow,
    position_policy: str,
    midpoint_position: Position | None,
) -> Position | None:
    if position_policy == "mean_in_window":
        mean_position = _mean_position(samples_in_window)
        return mean_position if mean_position is not None else _nearest_position(samples, window.midpoint)
    if position_policy == "nearest_midpoint":
        return _nearest_position(samples, window.midpoint)

    if midpoint_position is not None:
        return midpoint_position
    mean_position = _mean_position(samples_in_window)
    if mean_position is not None:
        return mean_position
    return _nearest_position(samples, window.midpoint)


def _sorted_samples(trajectory: list[TrajectoryRecord]) -> list[TrajectoryRecord]:
    return sorted(trajectory, key=lambda point: float(point["time"]))


def _interpolate_at(samples: list[TrajectoryRecord], time: float) -> Position | None:
    if not samples:
        return None
    if time < float(samples[0]["time"]) or time > float(samples[-1]["time"]):
        return None
    for point in samples:
        if float(point["time"]) == time:
            return float(point["x"]), float(point["y"])
    for left, right in zip(samples, samples[1:]):
        left_time = float(left["time"])
        right_time = float(right["time"])
        if left_time <= time <= right_time:
            if right_time == left_time:
                return float(left["x"]), float(left["y"])
            alpha = (time - left_time) / (right_time - left_time)
            return (
                float(left["x"]) + (float(right["x"]) - float(left["x"])) * alpha,
                float(left["y"]) + (float(right["y"]) - float(left["y"])) * alpha,
            )
    return None


def _mean_position(samples: list[TrajectoryRecord]) -> Position | None:
    if not samples:
        return None
    return (
        sum(float(sample["x"]) for sample in samples) / len(samples),
        sum(float(sample["y"]) for sample in samples) / len(samples),
    )


def _nearest_position(samples: list[TrajectoryRecord], time: float) -> Position | None:
    if not samples:
        return None
    nearest = min(samples, key=lambda sample: abs(float(sample["time"]) - time))
    return float(nearest["x"]), float(nearest["y"])


def _edges_from_layout(layout: dict[str, Any]) -> list[EdgeRecord]:
    raw_edges = layout.get("edges")
    if not raw_edges:
        return []
    if not isinstance(raw_edges, list):
        raise ValueError("Layout JSON 'edges' field must be a list when present.")
    edges: list[EdgeRecord] = []
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            raise ValueError(f"Layout edge {index} must be an object.")
        appearances = edge.get("appearances")
        if appearances:
            source = _required_value(edge, _SOURCE_HEADERS, f"layout edge {index}")
            target = _required_value(edge, _TARGET_HEADERS, f"layout edge {index}")
            edge_id = edge.get("id")
            for appearance_index, appearance in enumerate(appearances):
                if not isinstance(appearance, dict):
                    raise ValueError(f"Layout edge {index} appearance {appearance_index} must be an object.")
                start = float(appearance["start"])
                end = float(appearance.get("end", start + float(appearance.get("duration", 0.0))))
                edges.append(EdgeRecord(source=source, target=target, start=start, end=end, id=edge_id))
            continue
        edges.append(_edge_from_mapping(edge, row_repr=f"layout edge {index}"))
    return edges


def _edge_from_unheadered_row(row: list[str], *, row_number: int) -> EdgeRecord:
    fields = [field.strip() for field in row]
    if len(fields) == 2:
        return EdgeRecord(source=fields[0], target=fields[1])
    if len(fields) == 3:
        return EdgeRecord(source=fields[0], target=fields[1], time=float(fields[2]))
    if len(fields) == 4:
        start = float(fields[2])
        return EdgeRecord(source=fields[0], target=fields[1], start=start, end=start + float(fields[3]))
    raise ValueError(f"Unsupported edge CSV schema on row {row_number}: expected 2, 3, or 4 fields.")


def _edge_from_header_row(header: list[str], row: list[str], *, row_number: int) -> EdgeRecord:
    values = {_normalize_header(name): row[index].strip() if index < len(row) else "" for index, name in enumerate(header)}
    try:
        return _edge_from_mapping(values, row_repr=f"row {row_number}")
    except ValueError as exc:
        raise ValueError(f"Invalid edge CSV row {row_number}: {exc}") from exc


def _edge_from_mapping(mapping: dict[str, Any], *, row_repr: str) -> EdgeRecord:
    normalized = {_normalize_header(str(key)): value for key, value in mapping.items()}
    source = _required_value(normalized, _SOURCE_HEADERS, row_repr)
    target = _required_value(normalized, _TARGET_HEADERS, row_repr)
    edge_id = _optional_value(normalized, _ID_HEADERS)
    weight_value = _optional_value(normalized, _WEIGHT_HEADERS)
    weight = float(weight_value) if weight_value not in (None, "") else None
    time_value = _optional_value(normalized, _TIME_HEADERS)
    start_value = _optional_value(normalized, _START_HEADERS)
    end_value = _optional_value(normalized, _END_HEADERS)
    duration_value = _optional_value(normalized, _DURATION_HEADERS)

    if time_value not in (None, ""):
        return EdgeRecord(source=str(source), target=str(target), time=float(time_value), weight=weight, id=edge_id)
    if start_value not in (None, "") and duration_value not in (None, ""):
        start = float(start_value)
        return EdgeRecord(
            source=str(source),
            target=str(target),
            start=start,
            end=start + float(duration_value),
            weight=weight,
            id=edge_id,
        )
    if start_value not in (None, "") and end_value not in (None, ""):
        return EdgeRecord(
            source=str(source),
            target=str(target),
            start=float(start_value),
            end=float(end_value),
            weight=weight,
            id=edge_id,
        )
    if start_value in (None, "") and end_value in (None, "") and duration_value in (None, ""):
        return EdgeRecord(source=str(source), target=str(target), weight=weight, id=edge_id)
    raise ValueError(f"Unsupported temporal edge schema in {row_repr}.")


def _has_source_target_header(normalized_header: list[str]) -> bool:
    return _find_index(normalized_header, _SOURCE_HEADERS) is not None and _find_index(normalized_header, _TARGET_HEADERS) is not None


def _find_index(values: list[str], aliases: set[str]) -> int | None:
    for index, value in enumerate(values):
        if value in aliases:
            return index
    return None


def _required_value(mapping: dict[str, Any], aliases: set[str], row_repr: str) -> str:
    value = _optional_value(mapping, aliases)
    if value in (None, ""):
        raise ValueError(f"Missing required column in {row_repr}; accepted aliases are {sorted(aliases)}.")
    return str(value)


def _optional_value(mapping: dict[str, Any], aliases: set[str]) -> str | None:
    for alias in aliases:
        if alias in mapping and mapping[alias] is not None:
            return str(mapping[alias]).strip()
    return None


def _count_skipped_edges(edges: Sequence[EdgeRecord], window: SnapshotWindow, available_nodes: set[str]) -> int:
    return sum(
        1
        for edge in edges
        if edge.active_in(window) and (edge.source not in available_nodes or edge.target not in available_nodes)
    )


def _load_snapshot_node_sets(snapshot_nodes_dir: str | Path, n_snapshots: int) -> list[set[str]]:
    snapshot_dir = Path(snapshot_nodes_dir)
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot CSV directory does not exist: {snapshot_dir}")
    if not snapshot_dir.is_dir():
        raise ValueError(f"Snapshot CSV path must be a directory: {snapshot_dir}")

    node_sets: list[set[str]] = []
    for index in range(n_snapshots):
        node_sets.append(load_snapshot_csv_node_ids(snapshot_dir / f"snapshot_{index:03d}.csv"))
    return node_sets


def _resolve_snapshot_nodes_dir(
    layout_path: str | Path,
    snapshot_nodes_dir: str | Path | None,
    output_root: str | Path,
    dataset: str,
    n_snapshots: int,
) -> Path | None:
    if snapshot_nodes_dir is not None:
        return Path(snapshot_nodes_dir)

    candidates = [
        Path(layout_path).parent / "snapshots",
        Path(output_root) / dataset / "snapshots",
    ]
    for candidate in candidates:
        if _has_all_snapshot_csvs(candidate, n_snapshots):
            return candidate
    return None


def _has_all_snapshot_csvs(snapshot_dir: Path, n_snapshots: int) -> bool:
    if not snapshot_dir.is_dir():
        return False
    return all((snapshot_dir / f"snapshot_{index:03d}.csv").exists() for index in range(n_snapshots))


def _title_for_window(window: SnapshotWindow, *, title_format: str, timezone: str) -> str | None:
    if title_format == "none":
        return None
    if title_format == "datetime_range":
        return f"{_to_iso(window.start, timezone)} to {_to_iso(window.end, timezone)}"
    if title_format == "index_datetime":
        return f"Snapshot {window.index:03d} - {_to_iso(window.midpoint, timezone)}"
    raise ValueError(f"Unsupported title_format: {title_format}")


def _to_iso(timestamp: float, timezone: str) -> str:
    if timezone.upper() != "UTC":
        raise ValueError("Only timezone='UTC' is currently supported.")
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _normalize_header(field: str) -> str:
    return field.strip().lstrip("\ufeff").lower().replace(" ", "").replace("-", "_")
