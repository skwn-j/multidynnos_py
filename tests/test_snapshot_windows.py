from __future__ import annotations

import pytest

from multidynnos_py.experiments.snapshot_windows import compute_snapshot_windows


def test_compute_snapshot_windows_even_split() -> None:
    windows = compute_snapshot_windows([0.0, 10.0], 5)

    assert [(window.start, window.end, window.include_end) for window in windows] == [
        (0.0, 2.0, False),
        (2.0, 4.0, False),
        (4.0, 6.0, False),
        (6.0, 8.0, False),
        (8.0, 10.0, True),
    ]
    assert windows[0].contains(0.0)
    assert not windows[0].contains(2.0)
    assert windows[-1].contains(10.0)


def test_compute_snapshot_windows_validates_inputs() -> None:
    with pytest.raises(ValueError):
        compute_snapshot_windows([0.0, 1.0], 0)

    with pytest.raises(ValueError):
        compute_snapshot_windows([], 3)
