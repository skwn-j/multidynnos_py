from __future__ import annotations

import pytest

from multidynnos_py.data.temporal import Interval, presence_at


def test_closed_interval_contains_endpoints() -> None:
    interval = Interval.closed(1.0, 3.0)

    assert interval.contains(1.0)
    assert interval.contains(2.0)
    assert interval.contains(3.0)
    assert not interval.contains(0.999)
    assert not interval.contains(3.001)


def test_open_interval_excludes_endpoints() -> None:
    interval = Interval.open(1.0, 3.0)

    assert not interval.contains(1.0)
    assert interval.contains(2.0)
    assert not interval.contains(3.0)


def test_interval_from_start_duration() -> None:
    interval = Interval.from_start_duration(2.5, 1.5)

    assert interval.start == 2.5
    assert interval.end == 4.0
    assert interval.duration == 1.5


def test_negative_duration_rejected() -> None:
    with pytest.raises(ValueError):
        Interval.from_start_duration(1.0, -0.1)


def test_presence_at_uses_any_interval() -> None:
    intervals = [Interval.closed(0.0, 1.0), Interval.closed(3.0, 4.0)]

    assert presence_at(intervals, 0.5)
    assert not presence_at(intervals, 2.0)
    assert presence_at(intervals, 3.5)


def test_interval_overlap_respects_open_closed_boundaries() -> None:
    closed = Interval.closed(0.0, 1.0)
    right_closed = Interval.left_open_right_closed(1.0, 2.0)
    open_interval = Interval.open(1.0, 2.0)

    assert not closed.overlaps(right_closed)
    assert not closed.overlaps(open_interval)
    assert Interval.closed(1.0, 2.0).overlaps(closed)
