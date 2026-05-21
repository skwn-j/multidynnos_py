from __future__ import annotations

import json
from pathlib import Path

import pytest

from multidynnos_py.experiments.snapshot_windows import SnapshotWindow, compute_snapshot_windows
from multidynnos_py.cli import main
from multidynnos_py.visualization.snapshot_rendering import (
    active_edges_for_window,
    infer_dataset_name,
    load_edges_csv,
    render_layout_snapshots,
    sample_node_positions_for_window,
)


def _write_tiny_layout(path: Path, *, include_edges: bool = False) -> None:
    payload = {
        "config": {},
        "sample_times": [0.0, 10.0],
        "trajectories": {
            "A": [
                {"time": 0.0, "x": 0.0, "y": 0.0},
                {"time": 10.0, "x": 10.0, "y": 0.0},
            ],
            "B": [
                {"time": 0.0, "x": 0.0, "y": 10.0},
                {"time": 10.0, "x": 10.0, "y": 10.0},
            ],
        },
    }
    if include_edges:
        payload["edges"] = [{"source": "A", "target": "B"}]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_infer_dataset_name() -> None:
    assert infer_dataset_name("rugby_multidynnos_layout.json") == "rugby"
    assert infer_dataset_name("rugby_layout.json") == "rugby"
    assert infer_dataset_name("output/rugby/layout.json") == "rugby"
    assert infer_dataset_name("layout.json", override="rugby") == "rugby"


def test_sample_node_positions_mean_in_window() -> None:
    trajectories = {
        "A": [
            {"time": 0.0, "x": 0.0, "y": 0.0},
            {"time": 2.0, "x": 2.0, "y": 4.0},
            {"time": 4.0, "x": 4.0, "y": 8.0},
        ],
        "B": [{"time": 10.0, "x": 10.0, "y": 10.0}],
    }
    positions = sample_node_positions_for_window(
        trajectories,
        SnapshotWindow(index=0, start=0.0, end=3.0),
        position_policy="mean_in_window",
    )

    assert positions == {"A": (1.0, 2.0)}


def test_midpoint_interpolation() -> None:
    trajectories = {
        "A": [
            {"time": 0.0, "x": 0.0, "y": 0.0},
            {"time": 10.0, "x": 10.0, "y": 20.0},
        ],
    }
    positions = sample_node_positions_for_window(
        trajectories,
        SnapshotWindow(index=0, start=4.0, end=6.0),
        position_policy="midpoint_interpolate",
        active_node_policy="midpoint_available",
    )

    assert positions == {"A": (5.0, 10.0)}


def test_edges_csv_interval_activation(tmp_path) -> None:
    edge_file = tmp_path / "edges.csv"
    edge_file.write_text("source,target,start,duration\nA,B,1,2\nB,C,7,1\n", encoding="utf-8")
    edges = load_edges_csv(edge_file)
    windows = compute_snapshot_windows([0.0, 10.0], 2)

    first_edges = active_edges_for_window(edges, windows[0], {"A", "B", "C"})
    second_edges = active_edges_for_window(edges, windows[1], {"A", "B", "C"})

    assert [(edge.source, edge.target) for edge in first_edges] == [("A", "B")]
    assert [(edge.source, edge.target) for edge in second_edges] == [("B", "C")]


def test_render_snapshots_creates_files(tmp_path) -> None:
    layout_path = tmp_path / "tiny_layout.json"
    edges_path = tmp_path / "edges.csv"
    _write_tiny_layout(layout_path)
    edges_path.write_text("source,target\nA,B\n", encoding="utf-8")

    results = render_layout_snapshots(
        layout_path,
        n_snapshots=2,
        edges_path=edges_path,
        output_root=tmp_path / "output",
        title_format="none",
    )
    figures_dir = tmp_path / "output" / "tiny" / "figures"
    manifest = json.loads((figures_dir / "snapshots_manifest.json").read_text(encoding="utf-8"))

    assert len(results) == 2
    assert (figures_dir / "snapshot_000.png").exists()
    assert (figures_dir / "snapshot_001.png").exists()
    assert manifest["n_snapshots"] == 2
    assert manifest["position_policy"] == "mean_in_window"
    assert manifest["active_node_policy"] == "in_window"
    assert manifest["show_labels"] is False
    assert manifest["snapshots"][0]["num_nodes"] == 2
    assert manifest["snapshots"][0]["num_edges"] == 1


def test_no_edges_requires_explicit_nodes_only(tmp_path) -> None:
    layout_path = tmp_path / "tiny_layout.json"
    _write_tiny_layout(layout_path)

    with pytest.raises(ValueError, match="No edge information found"):
        render_layout_snapshots(layout_path, n_snapshots=1, output_root=tmp_path / "output")

    results = render_layout_snapshots(
        layout_path,
        n_snapshots=1,
        output_root=tmp_path / "output",
        allow_nodes_only=True,
        title_format="none",
    )
    manifest_path = tmp_path / "output" / "tiny" / "figures" / "snapshots_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert results[0].node_only
    assert manifest["node_only"] is True
    assert (tmp_path / "output" / "tiny" / "figures" / "snapshot_000.png").exists()


def test_render_real_rugby_layout_with_event_edges(tmp_path) -> None:
    layout_path = Path("output/rugby/layout.json")
    edges_path = Path("raw/rugby.csv")
    if not layout_path.exists() or not edges_path.exists():
        pytest.skip("rugby layout or raw rugby CSV is not available in this workspace")

    results = render_layout_snapshots(
        layout_path,
        n_snapshots=2,
        edges_path=edges_path,
        dataset_name="rugby",
        output_root=tmp_path,
        title_format="none",
    )
    manifest_path = tmp_path / "rugby" / "figures" / "snapshots_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(results) == 2
    assert (tmp_path / "rugby" / "figures" / "snapshot_000.png").exists()
    assert (tmp_path / "rugby" / "figures" / "snapshot_001.png").exists()
    assert manifest["edges_path"] == str(edges_path)
    assert sum(snapshot["num_edges"] for snapshot in manifest["snapshots"]) > 0


def test_cli_render_snapshots_nodes_only(tmp_path, capsys) -> None:
    layout_path = tmp_path / "tiny_layout.json"
    _write_tiny_layout(layout_path)

    status = main(
        [
            "render-snapshots",
            "--layout",
            str(layout_path),
            "--num-snapshots",
            "2",
            "--allow-nodes-only",
            "--output-root",
            str(tmp_path / "output"),
            "--title-format",
            "none",
        ]
    )
    captured = capsys.readouterr()
    figures_dir = tmp_path / "output" / "tiny" / "figures"

    assert status == 0
    assert "Rendered 2 snapshots" in captured.out
    assert (figures_dir / "snapshot_000.png").exists()
    assert (figures_dir / "snapshot_001.png").exists()
