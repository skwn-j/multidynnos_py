"""Optional parallel implementation of the existing edge-repulsion force.

The serial task pass preserves the reference seed boxes, trajectory exclusions,
and ``nodesDone`` order. Each parallel block accumulates into a private force
array, followed by a serial reduction in a fixed block order. Thread count does
not affect that order. Floating-point grouping differs from the NumPy backend;
the geometry and force equations do not. No fast-math transformations are used.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import math

import numpy as np


_MAX_BLOCKS = 32
_PRIVATE_FORCE_BUDGET = 64 * 1024 * 1024


def numba_available() -> bool:
    """Check the optional dependency without importing or compiling Numba."""
    return importlib.util.find_spec("numba") is not None


@lru_cache(maxsize=1)
def _kernels():
    try:
        import numba
        from numba import njit, prange
    except ImportError as exc:
        raise RuntimeError("The Numba backend requires the optional 'numba' package") from exc

    @njit(cache=True, fastmath=False)
    def tasks_for_boxes(positions, sources, targets, delta):
        count = len(sources)
        lower = np.empty((count, 3), dtype=np.float64)
        upper = np.empty((count, 3), dtype=np.float64)
        for segment in range(count):
            for axis in range(3):
                a = positions[sources[segment], axis]
                b = positions[targets[segment], axis]
                lower[segment, axis] = min(a, b) - 0.2
                upper[segment, axis] = max(a, b) + 0.2
        done = np.zeros(len(positions), dtype=np.bool_)
        # A segment with identical source/target indices generates two tasks
        # in the NumPy implementation, before either endpoint is marked done.
        tasks = np.empty((2 * count, 3), dtype=np.int64)
        task_count = 0
        for seed in range(count):
            if done[sources[seed]] and done[targets[seed]]:
                continue
            for first in range(count):
                inside = True
                for axis in range(3):
                    if (lower[first, axis] < lower[seed, axis] - 4 * delta
                            or upper[first, axis] > upper[seed, axis] + 4 * delta):
                        inside = False
                        break
                if not inside or (done[sources[first]] and done[targets[first]]):
                    continue
                for side in range(2):
                    point = sources[first] if side == 0 else targets[first]
                    if not done[point]:
                        tasks[task_count, 0] = point
                        tasks[task_count, 1] = seed
                        tasks[task_count, 2] = first
                        task_count += 1
                done[sources[first]] = True
                done[targets[first]] = True
        return lower, upper, tasks[:task_count]

    @njit(cache=True, fastmath=False, error_model="numpy", inline="always")
    def interaction(positions, node, source, target, delta, exponent, epsilon):
        px, py, pz = positions[node, 0], positions[node, 1], positions[node, 2]
        cx, cy, cz = positions[source, 0], positions[source, 1], positions[source, 2]
        ex, ey, ez = positions[target, 0], positions[target, 1], positions[target, 2]
        if ((abs(cx - px) <= epsilon and abs(cy - py) <= epsilon and abs(cz - pz) <= epsilon)
                or (abs(ex - px) <= epsilon and abs(ey - py) <= epsilon and abs(ez - pz) <= epsilon)):
            return False, 0.0, 0.0, 0.0, 0.0, 0.0
        dx, dy, dz = ex - cx, ey - cy, ez - cz
        squared = dx * dx + dy * dy + dz * dz
        degenerate = abs(dx) <= epsilon and abs(dy) <= epsilon and abs(dz) <= epsilon
        projection = 0.0 if degenerate else ((px - cx) * dx + (py - cy) * dy + (pz - cz) * dz) / squared
        included = (not degenerate and projection >= 0.0
                    and projection <= 1.0 + epsilon / max(np.sqrt(squared), epsilon))
        factor = projection if included else min(max(projection, 0.0), 1.0)
        vx, vy, vz = cx + factor * dx - px, cy + factor * dy - py, cz + factor * dz - pz
        distance = np.sqrt(vx * vx + vy * vy + vz * vz)
        if distance == 0.0:
            # Match np.argmin(abs(segment)), including its first-index tie rule.
            axis = 0
            if abs(dy) < abs(dx):
                axis = 1
            if abs(dz) < (abs(dx) if axis == 0 else abs(dy)):
                axis = 2
            if axis == 0:
                vx, vy, vz = 0.0, dz, -dy
            elif axis == 1:
                vx, vy, vz = -dz, 0.0, dx
            else:
                vx, vy, vz = dy, -dx, 0.0
            normal = np.sqrt(vx * vx + vy * vy + vz * vz)
            if normal != 0.0:
                vx, vy, vz = vx / normal, vy / normal, vz / normal
            else:
                vx, vy, vz = 1.0, 0.0, 0.0
            distance = epsilon
        norm = np.sqrt(vx * vx + vy * vy + vz * vz)
        strength = (delta / distance) ** exponent
        bx, by, bz = (vx / norm) * strength, (vy / norm) * strength, (vz / norm) * strength
        balance = abs(projection)
        weight_source = 1.0 - balance if included else 1.0
        weight_target = balance if included else 1.0
        return True, bx, by, bz, weight_source, weight_target

    @njit(cache=True, parallel=True, fastmath=False, error_model="numpy")
    def calculate(positions, sources, targets, owners, lower, upper, tasks,
                  delta, exponent, epsilon, blocks):
        private = np.zeros((blocks, len(positions), 3), dtype=np.float64)
        for block in prange(blocks):
            begin = len(tasks) * block // blocks
            end = len(tasks) * (block + 1) // blocks
            for task_index in range(begin, end):
                point, seed, first = tasks[task_index, 0], tasks[task_index, 1], tasks[task_index, 2]
                sx, sy, sz = 0.0, 0.0, 0.0
                for second in range(len(sources)):
                    if owners[second] == owners[first]:
                        continue
                    outside = False
                    for axis in range(3):
                        if (upper[second, axis] < lower[seed, axis] - 9 * delta
                                or lower[second, axis] > upper[seed, axis] + 9 * delta):
                            outside = True
                            break
                    if outside:
                        continue
                    source, target = sources[second], targets[second]
                    valid, bx, by, bz, ws, wt = interaction(
                        positions, point, source, target, delta, exponent, epsilon)
                    if not valid:
                        continue
                    sx += bx
                    sy += by
                    sz += bz
                    private[block, source, 0] += bx * ws
                    private[block, source, 1] += by * ws
                    private[block, source, 2] += bz * ws
                    private[block, target, 0] += bx * wt
                    private[block, target, 1] += by * wt
                    private[block, target, 2] += bz * wt
                private[block, point, 0] -= sx
                private[block, point, 1] -= sy
                private[block, point, 2] -= sz
        forces = np.zeros((len(positions), 3), dtype=np.float64)
        # This is deliberately a serial reduction in block order.
        for block in range(blocks):
            for point in range(len(positions)):
                for axis in range(3):
                    forces[point, axis] += private[block, point, axis]
        return forces

    return numba, tasks_for_boxes, calculate


def edge_repulsion_numba(
    positions: np.ndarray, sources: np.ndarray, targets: np.ndarray,
    owners: np.ndarray, delta: float, temperature: float,
    epsilon: float = 1e-4, num_threads: int | None = None,
) -> np.ndarray:
    """Compute the NumPy backend's force with deterministic parallel blocks.

    Inputs are the arrays returned by ``SpaceTimeCube.segments()`` and its
    positions. They remain unchanged. The optional thread override is restored
    after the call. Private force buffers use at most 64 MiB, except when one
    output-sized buffer alone exceeds that budget; no pairwise matrix is built.
    """
    if num_threads is not None and (isinstance(num_threads, bool)
                                    or not isinstance(num_threads, int) or num_threads < 1):
        raise ValueError("num_threads must be a positive integer")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (point_count, 3)")
    arrays = []
    for name, values in (("sources", sources), ("targets", targets), ("owners", owners)):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        arrays.append(values.astype(np.int64, copy=False))
    sources, targets, owners = arrays
    if len(sources) != len(targets) or len(sources) != len(owners):
        raise ValueError("sources, targets and owners must have equal lengths")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    for name, value in (("delta", delta), ("epsilon", epsilon)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if len(sources) and (min(sources.min(), targets.min()) < 0
                         or max(sources.max(), targets.max()) >= len(positions)):
        raise ValueError("segment endpoint indices must refer to existing positions")
    if not len(sources):
        return np.zeros_like(positions)
    numba, build_tasks, calculate = _kernels()
    previous_threads = numba.get_num_threads()
    if num_threads is not None:
        numba.set_num_threads(num_threads)
    try:
        lower, upper, tasks = build_tasks(positions, sources, targets, delta)
        blocks = min(_MAX_BLOCKS, max(1, len(tasks)),
                     max(1, _PRIVATE_FORCE_BUDGET // max(1, len(positions) * 3 * 8)))
        return calculate(positions, sources, targets, owners, lower, upper, tasks,
                         delta, 3 - 2 * temperature, epsilon, blocks)
    finally:
        if num_threads is not None:
            numba.set_num_threads(previous_threads)
