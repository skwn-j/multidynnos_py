"""ARCOL-derived additive force, reference normalization, and final box fit.

ARCOL section 3.2 / Algorithm 1 applies a centroid-preserving XY rescaling
after a normal layout iteration. Its fourth-root correction targets the ratio
of standard deviations, not the bounding-box ratio. Source:
https://arxiv.org/abs/2603.29618

The dynamic solver converts the reference normalization displacement into a
unit-weight XY force before the common 3D movement constraints. It does not
apply the reference normalization directly to the moving cube. This additive
integration is an intentional extension of the published ARCOL algorithm.

The public argument ``aspect_ratio=N`` means width:height = 1:N. Consequently
the paper's width/height target is 1/N. N=1 still applies the paper's correction.
Only active XY coordinates are written; time and inactive points are untouched.
Equal weighting of active mirror points is this dynamic port's sampling choice.
The public C++ implementation also skips variance below 1e-9 or absolute
width/height error below 1e-3; these guards apply to the normalization-derived
force as well, but not to the separate final bounding-box fit.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


MIN_VARIANCE = 1e-9
ASPECT_RATIO_TOLERANCE = 1e-3


def validate_aspect_ratio(aspect_ratio: float | None) -> None:
    """Validate the optional target height/width ratio; None disables it."""
    if aspect_ratio is None:
        return
    if (isinstance(aspect_ratio, (bool, np.bool_))
            or not isinstance(aspect_ratio, Real)
            or not math.isfinite(aspect_ratio) or aspect_ratio <= 0):
        raise ValueError("aspect_ratio must be a finite positive number or None")


def _active_coordinates(positions, active=None, *, writable=False):
    if writable and not isinstance(positions, np.ndarray):
        raise ValueError("In-place aspect-ratio adjustment requires a NumPy array")
    array = np.asarray(positions)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("positions must be a two-dimensional array with XY columns")
    if writable and (not np.issubdtype(array.dtype, np.floating) or not array.flags.writeable):
        raise ValueError("positions must be a writable floating-point array")
    if active is None:
        indices = np.arange(len(array), dtype=np.int64)
    else:
        indices = np.asarray(active)
        if indices.ndim != 1 or (len(indices) and not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("active must be a one-dimensional integer index array")
        indices = indices.astype(np.int64, copy=False)
        if len(indices) and (indices.min() < 0 or indices.max() >= len(array)):
            raise ValueError("active contains an index outside positions")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("active must not contain duplicate point indices")
    xy = np.asarray(array[indices, :2], dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("Active XY coordinates must be finite")
    return array, indices, xy


def _exp_metric(value: float) -> float | None:
    """A metric that cannot be represented finitely is reported as None."""
    if value == -math.inf:
        return 0.0
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) and result > 0 else None


def _summary(xy: np.ndarray):
    """Scaled arithmetic avoids overflow in means and squared deviations."""
    count = len(xy)
    if not count:
        return {
            "point_count": 0, "width": 0.0, "height": 0.0,
            "height_over_width": None, "std_height_over_width": None,
            "centroid": None,
        }, None
    scales = np.max(np.abs(xy), axis=0)
    unit = np.divide(xy, scales, out=np.zeros_like(xy), where=scales > 0)
    unit_mean = unit.mean(axis=0)
    centroid = unit_mean * scales
    centered_unit = unit - unit_mean
    variance_unit = np.mean(centered_unit * centered_unit, axis=0)
    ranges_unit = unit.max(axis=0) - unit.min(axis=0)

    def log_extent(unit_value, scale):
        return math.log(float(unit_value)) + math.log(float(scale)) if unit_value > 0 and scale > 0 else -math.inf

    log_std = np.array([log_extent(math.sqrt(float(variance_unit[i])), scales[i]) for i in range(2)])
    log_range = np.array([log_extent(ranges_unit[i], scales[i]) for i in range(2)])

    def ratio(logs):
        if logs[0] == -math.inf:
            return None
        return _exp_metric(float(logs[1] - logs[0]))

    # Direct subtraction makes ordinary bounding-box measurements transparent;
    # the logarithmic representation is retained for stable scaling factors.
    with np.errstate(over="ignore", invalid="ignore"):
        ranges = xy.max(axis=0) - xy.min(axis=0)
    measured = {
        "point_count": count,
        "width": float(ranges[0]) if np.isfinite(ranges[0]) else None,
        "height": float(ranges[1]) if np.isfinite(ranges[1]) else None,
        "height_over_width": ratio(log_range),
        "std_height_over_width": ratio(log_std),
        "centroid": [float(v) for v in centroid],
    }
    return measured, (centroid, centered_unit, scales, log_std, log_range)


def _within_absolute_tolerance(log_current: float, log_target: float) -> bool:
    """Check |current-target| < 1e-3 without overflowing either ratio."""
    current, target = _exp_metric(log_current), _exp_metric(log_target)
    if current is not None and target is not None:
        # Retain the C++ subtraction for representable ratios, including its
        # strict floating-point boundary (e.g. .501 - .5 is just above .001).
        return abs(current - target) < ASPECT_RATIO_TOLERANCE
    separation = abs(log_current - log_target)
    if separation == 0:
        return True
    log_difference = max(log_current, log_target) + math.log(-math.expm1(-separation))
    return log_difference < math.log(ASPECT_RATIO_TOLERANCE)


def measure_aspect_ratio(positions, active=None) -> dict:
    """Measure active XY points; ratios with zero width are None.

    Standard deviations use population variances and equal point weights.
    Unrepresentable non-finite metrics are None rather than JSON Infinity.
    No coordinates are modified.
    """
    _, _, xy = _active_coordinates(positions, active)
    return _summary(xy)[0]


def _adjust(positions, active, aspect_ratio, *, bounding_box: bool) -> dict:
    validate_aspect_ratio(aspect_ratio)
    array, indices, xy = _active_coordinates(positions, active, writable=True)
    before, detail = _summary(xy)
    stats = {
        "applied": False,
        "reason": None,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "max_xy_displacement": 0.0,
        "before": before,
        "after": dict(before),
    }
    if aspect_ratio is None:
        stats["reason"] = "disabled"
        return stats
    if len(indices) < 2:
        stats["reason"] = "no_active_points" if not len(indices) else "insufficient_points"
        return stats
    centroid, centered_unit, scales, log_std, log_range = detail
    log_spread = log_range if bounding_box else log_std
    for axis, label in enumerate(("x", "y")):
        if log_spread[axis] == -math.inf:
            stats["reason"] = ("zero_extent_" if bounding_box else "zero_variance_") + label
            return stats
    if not bounding_box:
        for axis, label in enumerate(("x", "y")):
            if 2 * log_std[axis] < math.log(MIN_VARIANCE):
                stats["reason"] = "variance_below_threshold_" + label
                return stats
        # C++ uses WIDTH/HEIGHT absolute error, not relative error and not the
        # public API's reciprocal HEIGHT/WIDTH convention.
        if _within_absolute_tolerance(float(log_std[0] - log_std[1]), -math.log(aspect_ratio)):
            stats["reason"] = "within_ratio_tolerance"
            return stats
    else:
        # scaleCoreToTargetAR in the published code scales about the point
        # bounding-box centre; its centre differs from the population mean.
        centroid = xy.min(axis=0) * .5 + xy.max(axis=0) * .5
        centered_unit = np.divide(xy, scales) - centroid / scales
    # In height/width convention: sx = (current_ratio / N) ** (1/4).
    # Bounding-box fitting uses the full, square-root correction instead.
    power = 0.5 if bounding_box else 0.25
    log_scale = power * (float(log_spread[1] - log_spread[0]) - math.log(aspect_ratio))
    if log_scale == 0.0:
        stats["reason"] = "already_at_target"
        return stats
    sx, sy = _exp_metric(log_scale), _exp_metric(-log_scale)
    if sx is None or sy is None or sx == 0 or sy == 0:
        stats["reason"] = "unrepresentable_scale"
        return stats
    # Form scaled deviations without squaring large coordinates. Multiplying
    # the unit deviations first also avoids an unnecessary intermediate range
    # overflow when opposite-sign finite coordinates straddle zero.
    factors = np.array([sx, sy])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        adjusted = centroid + (centered_unit * factors) * scales
        converted = adjusted.astype(array.dtype)
    if not np.isfinite(converted).all():
        stats["reason"] = "unrepresentable_coordinates"
        return stats
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = np.hypot(converted[:, 0] - xy[:, 0], converted[:, 1] - xy[:, 1])
    if not np.isfinite(displacement).all():
        stats["reason"] = "unrepresentable_displacement"
        return stats
    # The only write: all other columns (including Z) and rows stay bitwise
    # unchanged, even for a non-contiguous view of an underlying array.
    array[indices, :2] = converted
    stats.update({
        "applied": True,
        "reason": "applied",
        "scale_x": sx,
        "scale_y": sy,
        "max_xy_displacement": float(displacement.max()),
        "after": _summary(np.asarray(converted, dtype=np.float64))[0],
    })
    return stats


def normalize_aspect_ratio(positions, active, aspect_ratio) -> dict:
    """Apply ARCOL's fourth-root variance correction in place, including N=1.

    For target width:height=1:N, the new standard-deviation height/width ratio
    is the geometric mean of its previous value and N when a correction is
    applied. Bounding-box equality is not guaranteed. The public C++ guards
    skip variance below 1e-9 and absolute width/height error below 1e-3.
    Empty, single-point and zero-variance inputs also skip with a reason;
    no jitter or time-axis transformation is introduced.
    """
    return _adjust(positions, active, aspect_ratio, bounding_box=False)


def aspect_ratio_force(positions, active, aspect_ratio) -> tuple[np.ndarray, dict]:
    """Return ARCOL's proposed XY displacement as an additive unit-step force.

    The current iteration's coordinates determine the centroid, variances,
    and quarter-root scale. No input coordinates are modified. All inactive
    rows and non-spatial columns (including Z) have zero force. The solver adds
    this force to its four original forces before acceleration, cooling, the
    3D movement cap, and time constraints. Thus zero AR Z force does not imply
    unchanged Z movement after the common 3D constraints.

    ``before``/``after`` and ``max_xy_displacement`` in the returned reference
    statistics describe the hypothetical unconstrained normalization, not an
    applied solver movement. ``max_force_magnitude`` names its force magnitude.
    """
    array, indices, xy = _active_coordinates(positions, active)
    proposed = xy.copy()
    stats = normalize_aspect_ratio(proposed, None, aspect_ratio)
    forces = np.zeros(array.shape, dtype=np.float64)
    forces[indices, :2] = proposed - xy
    stats["max_force_magnitude"] = stats["max_xy_displacement"]
    return forces, stats


def fit_aspect_ratio(positions, active, aspect_ratio) -> dict:
    """Fit the active point bounding box to width:height=1:N in place.

    This optional final affine fit is separate from the paper's variance
    normalization. The square-root correction preserves bounding-box area,
    bounding-box centre, time, and inactive points, as the published code's
    scaleCoreToTargetAR does. Fixed-size node glyphs are not included in the
    measured box. Degenerate axes are skipped rather than invented.
    """
    return _adjust(positions, active, aspect_ratio, bounding_box=True)
