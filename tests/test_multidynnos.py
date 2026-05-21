from __future__ import annotations

import json

from multidynnos_py.cli import main
from multidynnos_py.data.io import build_graph_from_appearances, parse_edge_appearances, parse_node_appearances
from multidynnos_py.layout.dynnoslice import DynNoSliceConfig, DynamicLayout
from multidynnos_py.layout.multidynnos import MultiDynNoSConfig, run_multidynnos
from multidynnos_py.multilevel.coarsening import IndependentSetCoarsener, build_independent_set_hierarchy
from multidynnos_py.multilevel.flattener import StaticSumPresenceFlattener
from multidynnos_py.multilevel.placement import WeightedBarycenterPlacementStrategy


def _graph():
    return build_graph_from_appearances(
        parse_node_appearances(
            [
                "A,0,4",
                "B,0,4",
                "C,0,4",
                "D,0,4",
                "E,1,2",
            ]
        ),
        parse_edge_appearances(
            [
                "A,B,0,4",
                "B,C,0,4",
                "C,D,0,4",
                "D,E,1,2",
            ]
        ),
    )


def test_static_sum_presence_flattener_builds_weighted_summary() -> None:
    summary = StaticSumPresenceFlattener().flatten(_graph())

    assert summary.node_weights["A"] == 4.0
    assert summary.node_weights["E"] == 2.0
    assert summary.edge_weight("A", "B") == 4.0
    assert summary.neighbors("B") == {"A": 4.0, "C": 4.0}


def test_independent_set_coarsener_preserves_node_to_supernode_mapping() -> None:
    graph = _graph()
    step = IndependentSetCoarsener().coarsen(graph)

    assert set(step.fine_to_coarse) == set(graph.nodes)
    assert set().union(*[set(nodes) for nodes in step.coarse_to_fine.values()]) == set(graph.nodes)
    assert len(step.coarse_graph.nodes) < len(graph.nodes)
    assert all(supernode in step.coarse_graph.nodes for supernode in step.fine_to_coarse.values())


def test_build_hierarchy_refines_from_coarse_to_fine_levels() -> None:
    hierarchy = build_independent_set_hierarchy(_graph(), max_levels=4, min_coarse_nodes=1)

    assert hierarchy.levels[0] is not hierarchy.levels[-1]
    assert len(hierarchy.steps) == len(hierarchy.levels) - 1
    assert len(hierarchy.levels[-1].nodes) <= len(hierarchy.levels[0].nodes)


def test_weighted_barycenter_placement_expands_coarse_layout() -> None:
    graph = _graph()
    step = IndependentSetCoarsener().coarsen(graph)
    coarse_layout = run_multidynnos(
        step.coarse_graph,
        MultiDynNoSConfig(
            dynnoslice_config=DynNoSliceConfig(iterations=0, seed=3),
            max_levels=1,
            postprocess_passes=0,
        ),
    )
    expanded = WeightedBarycenterPlacementStrategy(seed=3).place(
        graph,
        coarse_layout,
        step.fine_to_coarse,
        config=DynNoSliceConfig(iterations=0, seed=3),
    )

    assert set(expanded.trajectories) == set(graph.nodes)
    assert expanded.sample_times == sorted(expanded.sample_times)
    assert all(point.to_dict() for trajectory in expanded.trajectories.values() for point in trajectory)


def test_run_multidynnos_returns_stable_json_serializable_layout() -> None:
    config = MultiDynNoSConfig(
        dynnoslice_config=DynNoSliceConfig(iterations=5, seed=9),
        max_levels=4,
        min_coarse_nodes=1,
        postprocess_passes=1,
    )

    first = run_multidynnos(_graph(), config)
    second = run_multidynnos(_graph(), config)

    assert isinstance(first, DynamicLayout)
    assert set(first.trajectories) == set(_graph().nodes)
    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True)


def test_cli_layout_method_multi_writes_json(tmp_path, capsys) -> None:
    node_file = tmp_path / "nodes.csv"
    edge_file = tmp_path / "edges.csv"
    output_file = tmp_path / "layout.json"
    node_file.write_text("A,0,2\nB,0,2\nC,0,2\n", encoding="utf-8")
    edge_file.write_text("A,B,0,2\nB,C,0,2\n", encoding="utf-8")

    status = main(
        [
            "layout",
            str(node_file),
            str(edge_file),
            "--method",
            "multi",
            "--iterations",
            "2",
            "--json",
            str(output_file),
        ]
    )
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert status == 0
    assert "Ran multi layout" in captured.out
    assert set(payload["trajectories"]) == {"A", "B", "C"}


def test_cli_layout_events_defaults_to_dataset_output_directory(tmp_path, monkeypatch, capsys) -> None:
    event_file = tmp_path / "rugby.csv"
    event_file.write_text("A,B,0\nB,C,1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    status = main(
        [
            "layout-events",
            str(event_file),
            "--method",
            "multi",
            "--iterations",
            "0",
            "--max-levels",
            "1",
            "--postprocess-passes",
            "0",
        ]
    )
    output_file = tmp_path / "output" / "rugby" / "layout.json"
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert status == 0
    assert output_file.exists()
    assert "Wrote layout to output/rugby/layout.json." in captured.out
    assert set(payload["trajectories"]) == {"A", "B", "C"}
    assert "edges" in payload


def test_cli_layout_events_visualize_renders_edge_snapshots(tmp_path, monkeypatch, capsys) -> None:
    event_file = tmp_path / "rugby.csv"
    event_file.write_text("A,B,0\nB,C,1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    status = main(
        [
            "layout-events",
            str(event_file),
            "--method",
            "multi",
            "--iterations",
            "0",
            "--max-levels",
            "1",
            "--postprocess-passes",
            "0",
            "--time-bins",
            "2",
            "--visualize",
            "2",
            "--show-labels",
        ]
    )
    manifest_file = tmp_path / "output" / "rugby" / "figures" / "snapshots_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert status == 0
    assert (tmp_path / "output" / "rugby" / "layout.json").exists()
    assert (tmp_path / "output" / "rugby" / "figures" / "snapshot_000.png").exists()
    assert (tmp_path / "output" / "rugby" / "figures" / "snapshot_001.png").exists()
    assert "Rendered 2 snapshots to output/rugby/figures." in captured.out
    assert manifest["node_only"] is False
    assert manifest["show_labels"] is True
    assert manifest["edges_path"] is None
    assert sum(snapshot["num_edges"] for snapshot in manifest["snapshots"]) == 2
    assert (tmp_path / "output" / "rugby" / "snapshots" / "snapshot_000.csv").read_text(encoding="utf-8") == "A,B,1\n"
    assert (tmp_path / "output" / "rugby" / "snapshots" / "snapshot_001.csv").read_text(encoding="utf-8") == "B,C,1\n"


def test_cli_layout_events_time_bins_limit_layout_sample_times(tmp_path, monkeypatch) -> None:
    event_file = tmp_path / "events.csv"
    event_file.write_text("A,B,0\nA,B,1\nB,C,2\nB,C,3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    status = main(
        [
            "layout-events",
            str(event_file),
            "--method",
            "multi",
            "--iterations",
            "0",
            "--max-levels",
            "1",
            "--postprocess-passes",
            "0",
            "--time-bins",
            "2",
        ]
    )
    payload = json.loads((tmp_path / "output" / "events" / "layout.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["sample_times"] == [0.0, 0.75, 1.5, 2.25, 3.0]
    assert len(payload["edges"][0]["appearances"]) <= 2
    assert (tmp_path / "output" / "events" / "snapshots" / "snapshot_000.csv").exists()
    assert (tmp_path / "output" / "events" / "snapshots" / "snapshot_001.csv").exists()
