from __future__ import annotations

import json

import pytest

from multidynnos_py.data.io import build_graph_from_appearances, parse_edge_appearances, parse_node_appearances
from multidynnos_py.layout.dynnoslice import DynNoSliceConfig, DynamicLayout, run_dynnoslice


def _tiny_graph():
    return build_graph_from_appearances(
        parse_node_appearances(["Alice,0,3", "Bob,0,3", "Carol,1,2"]),
        parse_edge_appearances(["Alice,Bob,0.5,1.5", "Bob,Carol,1.5,1"]),
    )


def test_run_dynnoslice_returns_stable_json_serializable_layout() -> None:
    graph = _tiny_graph()
    config = DynNoSliceConfig(tau=1.0, delta=5.0, iterations=10, learning_rate=0.05, seed=7)

    layout = run_dynnoslice(graph, config)
    encoded = json.dumps(layout.to_dict(), sort_keys=True)

    assert isinstance(layout, DynamicLayout)
    assert "Alice" in layout.trajectories
    assert layout.sample_times == sorted(layout.sample_times)
    assert encoded
    assert all(point.time in layout.sample_times for point in layout.trajectories["Alice"])


def test_run_dynnoslice_is_deterministic_for_same_seed() -> None:
    graph = _tiny_graph()
    config = DynNoSliceConfig(iterations=15, seed=11)

    first = run_dynnoslice(graph, config).to_dict()
    second = run_dynnoslice(graph, config).to_dict()

    assert first == second


def test_different_seed_changes_initial_layout_path() -> None:
    graph = _tiny_graph()

    first = run_dynnoslice(graph, DynNoSliceConfig(iterations=0, seed=1)).to_dict()
    second = run_dynnoslice(graph, DynNoSliceConfig(iterations=0, seed=2)).to_dict()

    assert first["trajectories"] != second["trajectories"]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DynNoSliceConfig(delta=0)

    with pytest.raises(ValueError):
        DynNoSliceConfig(iterations=-1)

