from __future__ import annotations

import json

from multidynnos_py.experiments.snapshots import export_small_multiples_data, sample_snapshots
from multidynnos_py.layout.dynnoslice import DynNoSliceConfig, DynamicLayout, TrajectoryPoint


def _layout() -> DynamicLayout:
    return DynamicLayout(
        trajectories={
            "A": [
                TrajectoryPoint(time=0.0, x=0.0, y=0.0),
                TrajectoryPoint(time=1.0, x=2.0, y=0.0),
                TrajectoryPoint(time=2.0, x=4.0, y=0.0),
            ],
            "B": [
                TrajectoryPoint(time=1.0, x=0.0, y=2.0),
                TrajectoryPoint(time=2.0, x=0.0, y=4.0),
            ],
        },
        sample_times=[0.0, 1.0, 2.0],
        config=DynNoSliceConfig(iterations=0),
    )


def test_sample_snapshots_interpolates_and_omits_absent_nodes() -> None:
    snapshots = sample_snapshots(_layout(), [0.5, 1.5, 3.0])

    assert snapshots[0].time == 0.5
    assert set(snapshots[0].nodes) == {"A"}
    assert snapshots[0].nodes["A"].x == 1.0
    assert snapshots[0].nodes["A"].y == 0.0

    assert set(snapshots[1].nodes) == {"A", "B"}
    assert snapshots[1].nodes["A"].x == 3.0
    assert snapshots[1].nodes["B"].y == 3.0

    assert snapshots[2].nodes == {}


def test_export_small_multiples_data_writes_json(tmp_path) -> None:
    out_path = tmp_path / "small_multiples.json"

    data = export_small_multiples_data(_layout(), (time for time in [0.0, 1.0]), out_path)
    restored = json.loads(out_path.read_text(encoding="utf-8"))

    assert restored == data
    assert restored["times"] == [0.0, 1.0]
    assert [snapshot["time"] for snapshot in restored["snapshots"]] == [0.0, 1.0]
    assert "layout_bounding_box" in restored
    assert "crowding" in restored["snapshots"][0]
