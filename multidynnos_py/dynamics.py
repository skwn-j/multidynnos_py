"""DynNoSlice's force system on a bend-explicit space-time cube.

Port of Ocotillo's DyModularForce, DyModularPreMovement,
DyModularPostProcessing, ModularConstraint, and SpaceTimeCubeSynchroniser.
Copyright 2014–2016 Paolo Simonetto; 2021 Alessio Arleo.
Original and this port are available under the Apache License, Version 2.0.

The temporal graph keeps time in its original units. Only this temporary mirror
uses ``z = time * tau``. Trajectory endpoints never change their time; interior
bends may move in time, subject to EnsureTimeCorrectness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .model import Interval, PositionSegment, TemporalGraph, position_at


EPSILON = 0.0001  # GeomXD.defaultEpsilon
SAFETY_MOVEMENT_FACTOR = 0.9  # ModularFdl.safetyMovementFactor
NUMBA_MIN_SEGMENTS = 256  # Avoid JIT startup for small default runs.


@dataclass
class MirrorLine:
    node_id: str
    interval: Interval
    points: list[int]


@dataclass
class MirrorConnection:
    source: MirrorLine
    target: MirrorLine
    start: float
    end: float
    left_closed: bool
    right_closed: bool


class SpaceTimeCube:
    """The two Java mirror graphs flattened into indexed coordinate arrays."""

    def __init__(self, graph: TemporalGraph, tau: float, seed: int = 0):
        self.graph = graph
        self.tau = tau
        self.lines: list[MirrorLine] = []
        self.node_lines: dict[str, list[MirrorLine]] = {}
        self.connections: list[MirrorConnection] = []
        coordinates = []
        for node in graph.nodes.values():
            lines = self.node_lines.setdefault(node.id, [])
            for appearance in sorted(node.presence, key=lambda p: (p.start, p.end)):
                start = position_at(node, appearance.start)
                end = position_at(node, appearance.end)
                points = [(*start, tau * appearance.start)]
                for function in sorted(node.positions, key=lambda s: (s.interval.start, s.interval.end)):
                    t = function.interval.end
                    if appearance.start < t < appearance.end:
                        # Java takes each function's rightValue, not valueAt(t).
                        if points[-1][2] != tau * t:
                            points.append((*function.end, tau * t))
                points.append((*end, tau * appearance.end))
                indices = list(range(len(coordinates), len(coordinates) + len(points)))
                coordinates.extend(points)
                line = MirrorLine(node.id, appearance, indices)
                self.lines.append(line)
                lines.append(line)
        self.positions = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        for edge in graph.edges:
            for appearance in edge.presence:
                # Normally presence validation guarantees a single containing
                # line. Splitting also supports an input crossing presence gaps.
                for source in self.node_lines.get(edge.source, []):
                    for target in self.node_lines.get(edge.target, []):
                        interval = _intersection(appearance, source.interval)
                        if interval is not None:
                            interval = _intersection(interval, target.interval)
                        if interval is not None:
                            self.connections.append(MirrorConnection(
                                source, target, tau * interval.start,
                                tau * interval.end, interval.left_closed,
                                interval.right_closed,
                            ))

    def active_points(self) -> np.ndarray:
        return np.asarray([point for line in self.lines for point in line.points], dtype=int)

    def segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        endpoints = []
        owners = []
        for owner, lines in enumerate(self.node_lines.values()):
            for line in lines:
                endpoints.extend(zip(line.points[:-1], line.points[1:]))
                owners.extend([owner] * (len(line.points) - 1))
        pairs = np.asarray(endpoints, dtype=int).reshape((-1, 2))
        return pairs[:, 0], pairs[:, 1], np.asarray(owners, dtype=int)

    def update_original(self) -> None:
        for node in self.graph.nodes.values():
            functions = []
            for line in self.node_lines[node.id]:
                for i, (a, b) in enumerate(zip(line.points[:-1], line.points[1:])):
                    start, end = self.positions[[a, b]]
                    left = line.interval.start if i == 0 else float(start[2] / self.tau)
                    right = (line.interval.end if i == len(line.points) - 2
                             else float(end[2] / self.tau))
                    interval = Interval(
                        left, right,
                        line.interval.left_closed if i == 0 else False,
                        line.interval.right_closed if i == len(line.points) - 2 else True,
                    )
                    functions.append(PositionSegment(interval, tuple(start[:2]), tuple(end[:2])))
            node.positions = functions


def _intersection(a: Interval, b: Interval) -> Interval | None:
    left, right = max(a.start, b.start), min(a.end, b.end)
    lc = (a.left_closed if a.start == left else True) and (b.left_closed if b.start == left else True)
    rc = (a.right_closed if a.end == right else True) and (b.right_closed if b.end == right else True)
    if left < right or (left == right and lc and rc):
        return Interval(left, right, lc, rc)
    return None


def time_straightening(cube: SpaceTimeCube, delta: float) -> np.ndarray:
    """TimeStraightning, including its all-pairs angular component."""
    forces = np.zeros_like(cube.positions)
    desired_distance = delta / 5.0
    for lines in cube.node_lines.values():
        ids = [point for line in lines for point in line.points]
        if len(ids) < 2:
            continue
        pos = cube.positions[ids]
        bend_ids = {point for line in lines for point in line.points[1:-1]}
        vectors = np.zeros_like(pos)
        vectors[0, :2] = (pos[1, :2] - pos[0, :2]) / 2
        vectors[-1, :2] = (pos[-2, :2] - pos[-1, :2]) / 2
        for i in range(1, len(ids) - 1):
            before, current, after = pos[i - 1:i + 2]
            if ids[i] in bend_ids:
                midpoint = (before + after) / 2
            else:
                width = after[2] - before[2]
                factor = (current[2] - before[2]) / width if width else 0.5
                midpoint = before + (after - before) * factor
            vectors[i] = (midpoint - current) * (2.0 / 3.0)
        magnitude = np.linalg.norm(vectors, axis=1)
        forces[ids] += vectors * (magnitude / desired_distance**2)[:, None]
        for i, source in enumerate(ids[:-1]):
            vector = pos[i + 1:] - pos[i]
            planar = np.linalg.norm(vector[:, :2], axis=1)
            valid = ((np.abs(vector[:, 2]) > EPSILON)
                     & np.any(np.abs(vector[:, :2]) > EPSILON, axis=1))
            angles = np.maximum(np.arctan2(np.abs(vector[:, 2]), planar), 0.01)
            pair_force = vector[:, :2] * ((math.pi / 2 - angles) / angles)[:, None]
            pair_force[~valid] = 0
            forces[source, :2] += pair_force.sum(axis=0)
            forces[np.asarray(ids[i + 1:]), :2] -= pair_force
    return forces


def gravity(cube: SpaceTimeCube, centre: np.ndarray) -> np.ndarray:
    """Unit attraction toward the fixed centre from the first iteration."""
    forces = np.zeros_like(cube.positions)
    ids = cube.active_points()
    vectors = centre - cube.positions[ids, :2]
    norms = np.linalg.norm(vectors, axis=1)
    np.divide(vectors, norms[:, None], out=vectors, where=norms[:, None] > 0)
    forces[ids, :2] = vectors
    return forces


def connection_attraction(cube: SpaceTimeCube, delta: float, temperature: float) -> np.ndarray:
    """ConnectionAttraction, integrating at both ends of each overlap."""
    forces = np.zeros_like(cube.positions)
    exponent = 2 + 2 * temperature
    for connection in cube.connections:
        for a, b in zip(connection.source.points[:-1], connection.source.points[1:]):
            pa, pb = cube.positions[[a, b]]
            left_a = max(pa[2], connection.start)
            right_a = min(pb[2], connection.end)
            if left_a > right_a:
                continue
            for c, d in zip(connection.target.points[:-1], connection.target.points[1:]):
                pc, pd = cube.positions[[c, d]]
                left = max(left_a, pc[2])
                right = min(right_a, pd[2])
                if left > right:
                    continue
                if left == right and ((left == connection.start and not connection.left_closed)
                                      or (left == connection.end and not connection.right_closed)):
                    continue
                aw, bw = pb[2] - pa[2], pd[2] - pc[2]
                ar = (right - left) / aw if aw else 1.0
                br = (right - left) / bw if bw else 1.0
                for z in (left, right):
                    ab = (z - pa[2]) / aw if aw else 0.5
                    bb = (z - pc[2]) / bw if bw else 0.5
                    vector = (pc + (pd - pc) * bb - pa - (pb - pa) * ab)[:2]
                    distance = np.linalg.norm(vector)
                    if distance <= EPSILON:
                        continue
                    force = vector / distance * (distance / delta)**exponent
                    forces[a, :2] += force * ar * (1 - ab)
                    forces[b, :2] += force * ar * ab
                    forces[c, :2] -= force * br * (1 - bb)
                    forces[d, :2] -= force * br * bb
    return forces


def _node_edge_repulsion(
    forces: np.ndarray, positions: np.ndarray, node: int,
    sources: np.ndarray, targets: np.ndarray, delta: float, exponent: float,
) -> None:
    """Vectorized Java applyNodeEdgeRepulsion; preserve endpoint reactions."""
    if not len(sources):
        return
    point = positions[node]
    c, d = positions[sources], positions[targets]
    noncoincident = ~(np.all(np.abs(c - point) <= EPSILON, axis=1)
                     | np.all(np.abs(d - point) <= EPSILON, axis=1))
    sources, targets, c, d = (x[noncoincident] for x in (sources, targets, c, d))
    if not len(sources):
        return
    segment = d - c
    squared_length = np.sum(segment**2, axis=1)
    degenerate = np.all(np.abs(segment) <= EPSILON, axis=1)
    projection = np.divide(np.sum((point - c) * segment, axis=1), squared_length,
                           out=np.zeros_like(squared_length), where=~degenerate)
    # GeomXD.isPointInSegment permits epsilon overshoot at the far endpoint.
    included = (~degenerate & (projection >= 0)
                & (projection <= 1 + EPSILON / np.maximum(np.sqrt(squared_length), EPSILON)))
    closest_factor = np.where(included, projection, np.clip(projection, 0, 1))
    closest = c + closest_factor[:, None] * segment
    difference = closest - point
    distance = np.linalg.norm(difference, axis=1)
    # A point exactly on another line makes Java's 0 * Infinity become NaN.
    # Choose a deterministic perpendicular in that singular case.
    singular = distance == 0
    if np.any(singular):
        for row in np.flatnonzero(singular):
            axis = np.zeros(3)
            axis[int(np.argmin(np.abs(segment[row])))] = 1
            normal = np.cross(segment[row], axis)
            norm = np.linalg.norm(normal)
            difference[row] = normal / norm if norm else np.array([1.0, 0.0, 0.0])
        distance[singular] = EPSILON
    unit = difference / np.linalg.norm(difference, axis=1)[:, None]
    base = unit * ((delta / distance)**exponent)[:, None]
    balance = np.abs(projection)
    weight_c = np.where(included, 1 - balance, 1)
    weight_d = np.where(included, balance, 1)
    forces[node] -= base.sum(axis=0)
    np.add.at(forces, sources, base * weight_c[:, None])
    np.add.at(forces, targets, base * weight_d[:, None])


def _validate_repulsion_options(backend: str, num_threads: int | None) -> None:
    if backend not in {"auto", "numpy", "numba"}:
        raise ValueError("repulsion_backend must be 'auto', 'numpy', or 'numba'")
    if num_threads is not None and (isinstance(num_threads, bool)
                                    or not isinstance(num_threads, int) or num_threads < 1):
        raise ValueError("num_threads must be a positive integer")


def _resolve_repulsion_backend(backend: str, segment_count: int) -> str:
    if backend == "numpy" or (backend == "auto" and segment_count < NUMBA_MIN_SEGMENTS):
        return "numpy"
    from .acceleration import numba_available
    if numba_available():
        return "numba"
    if backend == "auto":
        return "numpy"
    raise RuntimeError("Numba is not installed; run: pip install -e './mutidynnos_py[fast]'")


def edge_repulsion(cube: SpaceTimeCube, delta: float, temperature: float, *,
                   backend: str = "auto", num_threads: int | None = None) -> np.ndarray:
    """Use the reference NumPy force or its optional parallel Numba equivalent.

    Auto selects Numba when installed and there are at least 256 mirror
    segments. Explicit ``numpy`` preserves the previous reduction ordering.
    """
    _validate_repulsion_options(backend, num_threads)
    if backend == "numpy":
        return _edge_repulsion_numpy(cube, delta, temperature)
    segments = cube.segments()
    selected = _resolve_repulsion_backend(backend, len(segments[0]))
    if selected == "numpy":
        return _edge_repulsion_numpy(cube, delta, temperature)
    from .acceleration import edge_repulsion_numba
    return edge_repulsion_numba(cube.positions, *segments, delta, temperature,
                                epsilon=EPSILON, num_threads=num_threads)


def _edge_repulsion_numpy(cube: SpaceTimeCube, delta: float, temperature: float) -> np.ndarray:
    """EdgeRepulsion with the upstream 4δ/9δ box grouping and nodesDone."""
    forces = np.zeros_like(cube.positions)
    sources, targets, owners = cube.segments()
    if not len(sources):
        return forces
    pos = cube.positions
    # ElementLocatorAbst.computeBox expands by StdAttribute.edgeWidth (0.2).
    lower = np.minimum(pos[sources], pos[targets]) - 0.2
    upper = np.maximum(pos[sources], pos[targets]) + 0.2
    done = np.zeros(len(pos), dtype=bool)
    exponent = 3 - 2 * temperature
    for seed in range(len(sources)):
        if done[sources[seed]] and done[targets[seed]]:
            continue
        inner = np.flatnonzero(np.all(lower >= lower[seed] - 4 * delta, axis=1)
                               & np.all(upper <= upper[seed] + 4 * delta, axis=1))
        outer = np.flatnonzero(np.all(upper >= lower[seed] - 9 * delta, axis=1)
                               & np.all(lower <= upper[seed] + 9 * delta, axis=1))
        for first in inner:
            if done[sources[first]] and done[targets[first]]:
                continue
            second = outer[owners[outer] != owners[first]]
            for point in (sources[first], targets[first]):
                if not done[point]:
                    _node_edge_repulsion(forces, pos, point, sources[second], targets[second], delta, exponent)
            done[sources[first]] = True
            done[targets[first]] = True
    return forces


def movement_acceleration(
    forces: np.ndarray, previous: dict[int, np.ndarray], max_movement: float,
    active: np.ndarray,
) -> np.ndarray:
    """MovementAcceleration stores a limit vector, not the applied movement."""
    limits = np.full(len(forces), np.inf)
    for node in active:
        force = forces[node]
        # Upstream deliberately tests the planar force despite 3D geometry.
        if np.all(np.abs(force[:2]) <= EPSILON):
            previous.pop(int(node), None)
            continue
        magnitude = np.linalg.norm(force)
        old = previous.get(int(node))
        if old is None:
            limit = max_movement / 5
        else:
            old_magnitude = np.linalg.norm(old)
            if old_magnitude == 0:
                limit = 0.0
            else:
                angle = math.acos(float(np.clip(np.dot(force, old) / (magnitude * old_magnitude), -1, 1)))
                if angle < math.pi / 3:
                    limit = min(old_magnitude * (1 + 2 * (1 - angle / (math.pi / 3))), max_movement)
                elif angle < math.pi / 2:
                    limit = old_magnitude
                else:
                    limit = old_magnitude / (1 + 4 * (angle / (math.pi / 2) - 1))
        limits[node] = limit
        previous[int(node)] = force / magnitude * limit
    return limits


def ensure_time_correctness(cube: SpaceTimeCube, movements: np.ndarray, preserve_time: bool = False) -> None:
    """Preserve endpoint times and prevent bends overtaking their neighbors."""
    if preserve_time:
        movements[:, 2] = 0
    for line in cube.lines:
        ids = np.asarray(line.points)
        movements[ids[[0, -1]], 2] = 0
        pos, move = cube.positions[ids], movements[ids]
        factors = np.ones(len(ids))
        half = SAFETY_MOVEMENT_FACTOR * np.diff(pos[:, 2]) / 2
        forward, backward = move[:-1, 2], -move[1:, 2]
        mask = forward > half
        factors[:-1][mask] = np.minimum(factors[:-1][mask], half[mask] / forward[mask])
        mask = backward > half
        factors[1:][mask] = np.minimum(factors[1:][mask], half[mask] / backward[mask])
        movements[ids[1:-1]] *= factors[1:-1, None]


def flexible_trajectories(cube: SpaceTimeCube, contract_distance: float, expand_distance: float) -> tuple[int, int]:
    """Expand each old segment at most once, then contract bends in order."""
    inserted, removed = 0, 0
    new_positions = list(cube.positions)
    for line in cube.lines:
        expanded = [line.points[0]]
        for a, b in zip(line.points[:-1], line.points[1:]):
            start, end = cube.positions[[a, b]]
            if (np.linalg.norm(end - start) > expand_distance
                    and abs(end[2] - start[2]) > expand_distance / 2):
                expanded.append(len(new_positions))
                new_positions.append((start + end) / 2)
                inserted += 1
            expanded.append(b)
        line.points = expanded
    cube.positions = np.asarray(new_positions, dtype=float).reshape((-1, 3))
    for line in cube.lines:
        for bend in line.points[1:-1].copy():
            i = line.points.index(bend)
            a, b, c = cube.positions[line.points[i - 1:i + 2]]
            if (np.linalg.norm(c - a) < contract_distance
                    or np.linalg.norm(b - a) < contract_distance / 5
                    or np.linalg.norm(c - b) < contract_distance / 5):
                line.points.remove(bend)
                removed += 1
    return inserted, removed


def refine(
    graph: TemporalGraph, tau: float, delta: float = 5.0,
    max_movement: float | None = None, iterations: int = 75,
    flexible: bool = False, flexible_interval: int = 30, seed: int = 0,
    preserve_time: bool = False,
    repulsion_backend: str = "auto", num_threads: int | None = None,
    aspect_ratio: float | None = None,
) -> dict:
    """Run the four upstream forces and constraints, mutating node positions.

    ``preserve_time`` implements ForbidTimeShitfing and freezes existing bend
    times as well as endpoints. It is independent of the input data format.
    ``flexible`` enables FlexibleTimeTrajectories with distances 1.5δ and 2δ.
    ``repulsion_backend`` selects only the repulsion implementation. Auto is
    resolved once per refinement level, using its initial segment count.
    ``aspect_ratio`` is target height/width. ARCOL's proposed XY normalization
    displacement is added as a unit-weight force before the common movement
    constraints. Its Z component is zero; combined 3D movement can still change
    interior bend times under EnsureTimeCorrectness.
    """
    for name, value in (("tau", tau), ("delta", delta)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
    max_movement = 2 * delta if max_movement is None else max_movement
    if not math.isfinite(max_movement) or max_movement < 0:
        raise ValueError("max_movement must be finite and nonnegative")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not isinstance(flexible_interval, int) or flexible_interval < 1:
        raise ValueError("flexible_interval must be a positive integer")
    _validate_repulsion_options(repulsion_backend, num_threads)
    from .aspect_ratio import validate_aspect_ratio, aspect_ratio_force, measure_aspect_ratio
    validate_aspect_ratio(aspect_ratio)
    started = time.perf_counter()
    cube = SpaceTimeCube(graph, tau, seed)
    selected_backend = _resolve_repulsion_backend(
        repulsion_backend, sum(len(line.points) - 1 for line in cube.lines))
    effective_threads = None
    if selected_backend == "numba":
        import numba
        effective_threads = numba.get_num_threads() if num_threads is None else num_threads
        if effective_threads > numba.config.NUMBA_NUM_THREADS:
            raise ValueError(f"num_threads cannot exceed Numba's configured maximum "
                             f"({numba.config.NUMBA_NUM_THREADS})")
    active = cube.active_points()
    initial_points = len(active)
    if not np.all(np.isfinite(cube.positions)):
        raise ValueError("The space-time cube contains non-finite coordinates")
    centre = cube.positions[active, :2].mean(axis=0) if len(active) else np.zeros(2)
    previous: dict[int, np.ndarray] = {}
    inserted = removed = 0
    actual_max_movement = 0.0
    ar_trace = []
    ar_initial = measure_aspect_ratio(cube.positions, active) if aspect_ratio is not None else None
    for iteration in range(iterations):
        if not len(active):
            break
        temperature = (iterations - iteration) / iterations
        forces = time_straightening(cube, delta)
        forces += gravity(cube, centre)
        forces += connection_attraction(cube, delta, temperature)
        forces += edge_repulsion(cube, delta, temperature, backend=selected_backend,
                                  num_threads=num_threads)
        correction = None

        ## Where ARCOL force is applied
        if aspect_ratio is not None:
            ar_force, correction = aspect_ratio_force(cube.positions, active, aspect_ratio)
            forces += ar_force
        
        if not np.all(np.isfinite(forces[active])):
            raise FloatingPointError("Layout forces overflowed; rescale input coordinates or time")
        limits = movement_acceleration(forces, previous, max_movement, active)
        limits = np.minimum(limits, max_movement * temperature) * SAFETY_MOVEMENT_FACTOR
        magnitude = np.linalg.norm(forces, axis=1)
        scale = np.minimum(np.divide(limits, magnitude, out=np.zeros_like(limits),
                                     where=magnitude > EPSILON), 1.0)
        scale[limits <= EPSILON] = 0
        movements = forces * scale[:, None]
        ensure_time_correctness(cube, movements, preserve_time)
        step_max_movement = float(np.linalg.norm(movements[active], axis=1).max())
        actual_max_movement = max(actual_max_movement, step_max_movement)
        if aspect_ratio is not None:
            step_max_xy_movement = float(np.linalg.norm(movements[active, :2], axis=1).max())
        cube.positions += movements
        if flexible and temperature > 0.2 and iteration % flexible_interval == 0:
            add, delete = flexible_trajectories(cube, 1.5 * delta, 2 * delta)
            inserted += add
            removed += delete
            active = cube.active_points()
        if aspect_ratio is not None:
            actual_geometry = measure_aspect_ratio(cube.positions, active)
            ar_trace.append({
                "iteration": iteration + 1,
                "active_points": len(active), "applied": correction["applied"],
                "force_active_points": correction["before"]["point_count"],
                "reason": correction["reason"],
                "scale_x": correction["scale_x"], "scale_y": correction["scale_y"],
                "max_force_magnitude": correction["max_force_magnitude"],
                "cooling_temperature": temperature,
                "max_movement_cap": float(max_movement * temperature * SAFETY_MOVEMENT_FACTOR),
                "max_total_movement": step_max_movement,
                "max_xy_displacement": step_max_xy_movement,
                "std_height_over_width_before": correction["before"]["std_height_over_width"],
                "std_height_over_width_after": actual_geometry["std_height_over_width"],
                "bbox_height_over_width_before": correction["before"]["height_over_width"],
                "bbox_height_over_width_after": actual_geometry["height_over_width"],
                "before": correction["before"], "after": actual_geometry,
            })
    cube.update_original()
    statistics = {
        "iterations": iterations if initial_points else 0,
        "initial_mirror_points": initial_points,
        "final_mirror_points": len(active),
        "bends_inserted": inserted,
        "bends_removed": removed,
        "max_applied_movement": actual_max_movement,
        "repulsion_backend": selected_backend,
        "repulsion_threads": effective_threads,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if aspect_ratio is not None:
        statistics["aspect_ratio"] = {
            "target_height_over_width": float(aspect_ratio),
            "method": "ARCOL quarter-root variance correction adapted as an additive force",
            "integration": "additive_force",
            "force_weight": 1.0,
            "sampling": "uniform active mirror points",
            "stage": "summed with original forces before movement constraints",
            "direct_time_adjustment": False,
            "time_effect": "indirect through combined 3D movement and time constraints",
            "applied_semantics": "AR force component enabled before the common constraints",
            "max_xy_displacement_semantics": "actual combined XY movement after all constraints",
            "before": ar_initial, "after": measure_aspect_ratio(cube.positions, active),
            "steps": ar_trace,
        }
    return statistics
