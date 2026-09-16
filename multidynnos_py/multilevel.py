"""MultiDynNoS's GRIP coarsening, placement, and layout driver.

Ported from EngAAlex/MultiDynNos, revision
068aa79680b7d670d2338493bc9c88f4ffbd3db6 (Apache-2.0).
The intermediate hierarchy uses separate graph objects as node-ID namespaces;
the caller's IDs therefore survive unmodified. See PORTING.md for deviations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import logging
import math
import shutil
import subprocess
from typing import Iterable

import numpy as np

from .model import Edge, Interval, Node, PositionSegment, TemporalGraph, merge_intervals

logger = logging.getLogger(__name__)


class JavaRandom:
    """java.util.Random's 48-bit generator, including exact nextDouble bits."""

    def __init__(self, seed: int):
        self.state = (int(seed) ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def _next(self, bits: int) -> int:
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.state >> (48 - bits)

    def random(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) / float(1 << 53)


def duration_weight(presence: Iterable[Interval]) -> float:
    """StaticSumPresenceFlattener: sum end-start, including zero-length events."""
    return sum(interval.end - interval.start for interval in presence)


@dataclass
class HierarchyLevel:
    graph: TemporalGraph
    node_weights: dict[str, float]
    edge_weights: dict[str, float]
    # On a coarse level, groups and leaders describe its immediately finer level.
    groups: dict[str, list[str]] = field(default_factory=dict)
    leaders: dict[str, str] = field(default_factory=dict)


def flatten(graph: TemporalGraph) -> HierarchyLevel:
    """Attach presence-duration weights without changing the dynamic graph."""
    return HierarchyLevel(graph,
                          {key: duration_weight(node.presence) for key, node in graph.nodes.items()},
                          {edge.id: duration_weight(edge.presence) for edge in graph.edges})


def coarsen(graph: TemporalGraph) -> list[HierarchyLevel]:
    """Return the GRIP hierarchy, finest first, preserving outgoing-edge rules.

    Java's outEdges() is directional even when the graph is drawn undirected.
    The betweenEdge() lookup, conversely, combines either orientation.
    """
    hierarchy = [flatten(copy.deepcopy(graph))]
    while hierarchy[-1].graph.nodes:
        fine = hierarchy[-1]
        outgoing: dict[str, list[Edge]] = {key: [] for key in fine.graph.nodes}
        for edge in fine.graph.edges:
            outgoing[edge.source].append(edge)
        scores = {key: fine.node_weights[key] + sum(fine.edge_weights[e.id]
                                                   for e in outgoing[key])
                  for key in fine.graph.nodes}
        # Stable input order replaces Java HashMap iteration when scores tie.
        ordered = sorted(fine.graph.nodes, key=lambda key: -scores[key])
        available = set(ordered)
        groups: dict[str, list[str]] = {}
        leaders: dict[str, str] = {}
        for key in ordered:
            if key not in available:
                continue
            available.remove(key)
            groups[key] = [key]
            leaders[key] = key
            for edge in outgoing[key]:
                other = edge.target
                if other in available:
                    available.remove(other)
                    groups[key].append(other)
                    leaders[other] = key

        count = len(groups)
        previous_count = len(fine.graph.nodes)
        if count >= previous_count or (len(hierarchy) > 1 and count / previous_count > 0.95):
            break

        nodes = {leader: Node(leader, merge_intervals(i for member in members
                                                      for i in fine.graph.nodes[member].presence))
                 for leader, members in groups.items()}
        node_weights = {leader: sum(fine.node_weights[member] for member in members)
                        for leader, members in groups.items()}
        edges: list[Edge] = []
        edge_weights: dict[str, float] = {}
        between: dict[frozenset[str], Edge] = {}
        # Iterate source vertices before outgoing edges, exactly as generateEdges.
        for source in fine.graph.nodes:
            for edge in outgoing[source]:
                source_leader, target_leader = leaders[edge.source], leaders[edge.target]
                if source_leader == target_leader:
                    continue
                pair = frozenset((source_leader, target_leader))
                if pair not in between:
                    created = Edge(f"coarse_edge_{len(edges)}", source_leader, target_leader,
                                   copy.deepcopy(edge.presence))
                    between[pair] = created
                    edges.append(created)
                    edge_weights[created.id] = fine.edge_weights[edge.id]
                else:
                    created = between[pair]
                    created.presence = merge_intervals((*created.presence, *edge.presence))
                    edge_weights[created.id] += fine.edge_weights[edge.id]
        coarse_graph = TemporalGraph(nodes=nodes, edges=edges,
                                     data_type=graph.data_type, directed=graph.directed)
        hierarchy.append(HierarchyLevel(coarse_graph, node_weights, edge_weights, groups, leaders))
    return hierarchy


def estimate_tau(graph: TemporalGraph, mode: str | None = None) -> float:
    """MultiDynNoS paper, section 3.7.1, equation (2), DOI 10.1111/cgf.14615.

    Divide mean true-interval duration by the sum of the global node and edge
    time spans. Both node and edge intervals participate, regardless of load
    mode. ``mode`` remains accepted for API compatibility and provenance only;
    presence preprocessing must happen before this calculation.

    This intentionally corrects DyGraph.autocomputeTau's edge-extrema updates
    and mode-dependent exclusions. Empty/degenerate inputs retain tau=1 as a
    documented fallback, outside the paper's positive-span formula.
    """
    mode = mode or graph.metadata.get("presence_mode", graph.metadata.get("load_mode", "plain"))
    node_intervals = [interval for node in graph.nodes.values() for interval in node.presence]
    edge_intervals = [interval for edge in graph.edges for interval in edge.presence]

    def summarize(intervals: list[Interval]) -> tuple[int, float, float]:
        try:
            duration = math.fsum(interval.duration for interval in intervals)
        except OverflowError:
            duration = math.inf
        span = (max(interval.end for interval in intervals)
                - min(interval.start for interval in intervals)) if intervals else 0.0
        return len(intervals), duration, span

    nodes = summarize(node_intervals)
    edges = summarize(edge_intervals)
    count = nodes[0] + edges[0]
    span = nodes[2] + edges[2]
    duration = nodes[1] + edges[1]
    value = duration / count / span if count and span > 0.0 else float("nan")
    valid = math.isfinite(value) and value > 0.0

    def statistics(summary: tuple[int, float, float]) -> dict:
        return {"interval_count": summary[0],
                "total_duration": summary[1] if math.isfinite(summary[1]) else None,
                "time_span": summary[2] if math.isfinite(summary[2]) else None}

    graph.metadata["tau_estimation"] = {
        "method": "MultiDynNoS AutoTau equation (2)",
        "paper_doi": "10.1111/cgf.14615",
        "load_mode": mode,
        "preserves_upstream_edge_span_bug": False,
        "iteration_order": "order-independent",
        "nodes": statistics(nodes),
        "edges": statistics(edges),
        "degenerate_fallback": not valid,
    }
    return float(value) if valid else 1.0


def _constant_position(node: Node, point: np.ndarray | tuple[float, float]) -> None:
    coordinates = tuple(float(value) for value in point)
    node.positions = [PositionSegment(interval, coordinates, coordinates)
                      for interval in node.presence]


def _graphviz_bounding_box(result: dict) -> dict:
    """Keep Graphviz's graph box separate from the node-center measurements."""
    raw = result.get("bb")
    box = {"raw": raw, "coordinate_unit": "points", "bounds": None,
           "width": None, "height": None, "height_over_width": None,
           "status": "not_reported" if raw is None else "invalid"}
    if raw is None:
        return box
    try:
        values = [float(value) for value in raw.split(",")]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            return box
        width, height = values[2] - values[0], values[3] - values[1]
        if min(width, height) < 0 or not all(math.isfinite(v) for v in (width, height)):
            return box
        ratio = height / width if width else None
        box.update(bounds=values, width=width, height=height,
                   height_over_width=ratio if ratio is None or math.isfinite(ratio) else None,
                   status="reported")
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    return box


def static_layout(graph: TemporalGraph, seed: int = 0, engine: str = "sfdp",
                  level: int | None = None, initial_aspect_ratio: float | None = None) -> dict:
    """Use the original Graphviz initializer and its inches coordinate units.

    Flattening here deliberately recomputes durations of merged edge presences;
    these differ from summed coarsening weights when lower edges overlap.
    ``initial_aspect_ratio`` is Graphviz's requested height/width ratio, passed
    only as ``-Gratio=N``. It does not fit the node-center box afterward.
    Returned measurements describe this initial static graph, before refinement.
    """
    from .aspect_ratio import measure_aspect_ratio, validate_aspect_ratio
    try:
        validate_aspect_ratio(initial_aspect_ratio)
    except ValueError as error:
        raise ValueError("initial_aspect_ratio must be a finite positive number or None") from error
    if engine not in {"sfdp", "fdp"}:
        raise ValueError("Graphviz engine must be 'sfdp' or 'fdp'")
    stats = {
        "method": "graphviz",
        "enabled": initial_aspect_ratio is not None,
        "requested_height_over_width": (float(initial_aspect_ratio)
                                        if initial_aspect_ratio is not None else None),
        "engine": engine, "level": level, "node_count": len(graph.nodes),
        "command": [], "status": "empty_graph",
        "node_center_geometry": measure_aspect_ratio(np.empty((0, 2))),
        "node_center_coordinate_unit": "inches",
        "graphviz_bounding_box": _graphviz_bounding_box({}),
        "exact_node_center_ratio_guaranteed": False,
        "scope": "initial static graph only, before dynamic refinement and final aspect-ratio fitting",
        "ratio_semantics": "Graphviz graph ratio and measured node-center bbox/std ratios are distinct; no extra center fitting is applied",
    }
    if not graph.nodes:
        return stats
    executable = shutil.which(engine)
    if executable is None:
        raise RuntimeError(f"Graphviz {engine!r} executable was not found; install Graphviz")
    aliases = {key: f"n{index}" for index, key in enumerate(graph.nodes)}
    lines = ['graph G {', 'node [width="1.00", height="1.00", pos="0.00,0.00"];']
    for key, alias in aliases.items():
        label = key if level is None else f"{key}__{level}"
        lines.append(f"{alias} [label={json.dumps(label, ensure_ascii=False)}];")
    for edge in graph.edges:
        weight = duration_weight(edge.presence)
        lines.append(f'{aliases[edge.source]} -- {aliases[edge.target]} [weight="{weight}"];')
    lines.append('}')
    command = [executable, "-Tjson"]
    if seed:
        command.append(f"-Gstart={int(seed)}")
    if initial_aspect_ratio is not None:
        command.append(f"-Gratio={float(initial_aspect_ratio):.17g}")
    completed = subprocess.run(command, input="\n".join(lines), text=True,
                               capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Graphviz {engine} failed: {completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout)
        positions = {item["name"]: tuple(float(value) / 72.0
                                         for value in item["pos"].split(",")[:2])
                     for item in result.get("objects", []) if "pos" in item}
        for key, node in graph.nodes.items():
            _constant_position(node, positions[aliases[key]])
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Could not read positions returned by Graphviz {engine}") from error
    points = np.asarray([positions[aliases[key]] for key in graph.nodes], dtype=float)
    stats.update(command=command, status="completed",
                 node_center_geometry=measure_aspect_ratio(points),
                 graphviz_bounding_box=_graphviz_bounding_box(result))
    return stats


def _expand_initial_ratio(points: np.ndarray, target: float | None) -> dict:
    """Apply Graphviz's numeric-ratio expansion rule to a node-center box.

    Expand only the deficient axis about the box center. This preserves XY
    orientation and does not run ARCOL or change time coordinates. Unlike
    Graphviz's drawing box, this box excludes node glyphs, labels, and margins.
    """
    from .aspect_ratio import measure_aspect_ratio
    before = measure_aspect_ratio(points)
    stats = {"applied": False, "reason": "disabled", "scale_x": 1.0,
             "scale_y": 1.0, "center": None, "before": before,
             "after": dict(before)}
    if target is None:
        return stats
    if not len(points):
        stats["reason"] = "empty_graph"
        return stats
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = upper - lower
    center = lower + extent / 2.0
    stats["center"] = center.tolist()
    if np.any(extent <= 0):
        stats["reason"] = "degenerate_node_center_box"
        return stats
    current = float(extent[1] / extent[0])
    scale_x, scale_y = max(1.0, current / target), max(1.0, target / current)
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = (points - center) * (scale_x, scale_y) + center
    if not np.isfinite(adjusted).all():
        raise ValueError("initial aspect ratio produces non-finite coordinates")
    points[:] = adjusted
    applied = scale_x != 1.0 or scale_y != 1.0
    stats.update(applied=applied, reason="expanded" if applied else "already_at_target",
                 scale_x=float(scale_x), scale_y=float(scale_y),
                 after=measure_aspect_ratio(points))
    return stats


def spring_layout(graph: TemporalGraph, seed: int = 0, level: int | None = None,
                  initial_aspect_ratio: float | None = None) -> dict:
    """Initialize XY using NetworkX spring_layout on the duration-weighted graph.

    Use NetworkX defaults (including scale=1 and 50 iterations), with a seeded
    2D layout. Parallel/oppositely directed edges contribute summed duration
    weights to an undirected graph. Preserve isolated nodes and time intervals.
    """
    import networkx as nx
    from .aspect_ratio import measure_aspect_ratio, validate_aspect_ratio
    try:
        validate_aspect_ratio(initial_aspect_ratio)
    except ValueError as error:
        raise ValueError("initial_aspect_ratio must be a finite positive number or None") from error
    effective_seed = int(seed) % (2 ** 32)
    flat = nx.Graph()
    flat.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        weight = duration_weight(edge.presence)
        if flat.has_edge(edge.source, edge.target):
            flat[edge.source][edge.target]["weight"] += weight
        else:
            flat.add_edge(edge.source, edge.target, weight=weight)
    positions = nx.spring_layout(flat, dim=2, seed=effective_seed, weight="weight")
    points = np.asarray([positions[key] for key in graph.nodes], dtype=float).reshape(-1, 2)
    adjustment = _expand_initial_ratio(points, initial_aspect_ratio)
    for key, point in zip(graph.nodes, points):
        _constant_position(graph.nodes[key], point)
    return {
        "method": "networkx.spring_layout", "networkx_version": nx.__version__,
        "enabled": initial_aspect_ratio is not None,
        "requested_height_over_width": (float(initial_aspect_ratio)
                                        if initial_aspect_ratio is not None else None),
        "seed": effective_seed, "seed_semantics": "seed modulo 2**32",
        "level": level, "node_count": len(graph.nodes),
        "status": "completed" if graph.nodes else "empty_graph", "command": [],
        "parameters": {"dim": 2, "weight": "weight", "iterations": 50, "scale": 1},
        "node_center_geometry": measure_aspect_ratio(points),
        "node_center_coordinate_unit": "NetworkX normalized layout units",
        "adjustment": adjustment,
        "exact_node_center_ratio_guaranteed": (initial_aspect_ratio is not None
                                               and adjustment["reason"] in {"expanded", "already_at_target"}),
        "scope": "initial static graph only, before dynamic refinement and final aspect-ratio fitting",
        "ratio_semantics": "expand only the insufficient axis around the node-center bounding-box center; degenerate boxes are unchanged",
    }


def _initialize_layout(graph: TemporalGraph, *, graphviz: bool, seed: int,
                       engine: str, level: int | None = None,
                       initial_aspect_ratio: float | None = None) -> dict:
    if graphviz:
        return static_layout(graph, seed=seed, engine=engine, level=level,
                             initial_aspect_ratio=initial_aspect_ratio)
    return spring_layout(graph, seed=seed, level=level,
                         initial_aspect_ratio=initial_aspect_ratio)


def scatter_nodes(graph: TemporalGraph, distance: float, seed: int = 73) -> None:
    """Commons.scatterNodes: random.nextDouble()*distance for x, then y."""
    random = JavaRandom(seed)
    for node in graph.nodes.values():
        _constant_position(node, (random.random() * distance, random.random() * distance))


def _last_position(node: Node) -> np.ndarray:
    return np.asarray(node.positions[-1].end if node.positions else (0.0, 0.0), dtype=float)


def barycenter_position(lower: str, fine: TemporalGraph, coarse: HierarchyLevel,
                        random: JavaRandom, *, fuzziness: float = 0.05,
                        optimal_distance: float = 10.0) -> np.ndarray:
    """WeightedBarycenterPlacementStrategy with distinct neighbors per cluster."""
    leader = coarse.leaders[lower]
    own_position = _last_position(coarse.graph.nodes[leader])
    neighbors: dict[str, set[str]] = {}
    for edge in fine.edges:
        if lower not in (edge.source, edge.target):
            continue
        other = edge.target if lower == edge.source else edge.source
        other_leader = coarse.leaders[other]
        if other_leader != leader:
            neighbors.setdefault(other_leader, set()).add(other)
    if not neighbors:
        angle = random.random() * 2.0 * math.pi
        result = own_position + optimal_distance * np.asarray((math.cos(angle), math.sin(angle)))
    else:
        own_weight = len(coarse.groups[leader])
        result = own_position * own_weight
        total_weight = own_weight
        for other_leader, members in neighbors.items():
            result = result + _last_position(coarse.graph.nodes[other_leader]) * len(members)
            total_weight += len(members)
        result = result / total_weight
    for dimension in range(2):
        magnitude = random.random() * fuzziness
        sign = 1.0 if random.random() > 0.5 else -1.0
        result[dimension] += magnitude * sign
    return result


def place_vertices(fine: TemporalGraph, coarse: HierarchyLevel, random: JavaRandom,
                   bend_transfer: bool = False) -> None:
    """Transfer coarse positions to one finer level, optionally preserving bends.

    The optional Java bend-transfer path aliases left and right Coordinates.
    Consequently each nonleader segment is constant at the accumulated right
    endpoint, with jumps between segments. This quirk is deliberately retained.
    """
    for leader, members in coarse.groups.items():
        upper = coarse.graph.nodes[leader]
        for lower in members:
            node = fine.nodes[lower]
            if bend_transfer and lower == leader:
                node.positions = copy.deepcopy(upper.positions)
                continue
            point = barycenter_position(lower, fine, coarse, random)
            if not bend_transfer or not upper.positions:
                _constant_position(node, point)
                continue
            segments = []
            previous_right: np.ndarray | None = None
            for segment in upper.positions:
                left, right = np.asarray(segment.start), np.asarray(segment.end)
                if previous_right is not None:
                    point = point + left - previous_right
                point = point + right - left
                coordinates = tuple(float(value) for value in point)
                segments.append(PositionSegment(segment.interval, coordinates, coordinates))
                previous_right = right
            node.positions = segments


def cooling_schedule(depth: int, delta: float = 5.0,
                     iterations: int = 75) -> list[tuple[float, int]]:
    """Cumulative cooling; values remain at their previous level below minima."""
    movement, rounds = 2.0 * delta, float(iterations)
    schedule = []
    for index in range(depth):
        if index:
            factor = 1.0 - 0.07 * (index + 1)
            candidate = movement * factor
            if candidate > 3.0:
                movement = candidate
            candidate = rounds * factor
            if candidate > 20.0:
                rounds = candidate
        schedule.append((movement, math.ceil(rounds)))
    return schedule


def _finish_aspect_ratio(graph: TemporalGraph, target: float | None, fit: str) -> None:
    """Measure the complete cube and optionally enforce its final XY box ratio.

    Linear trajectories attain coordinate extrema at their endpoints. This
    final affine transform therefore fits every interpolated position too.
    """
    if target is None:
        return
    from .aspect_ratio import fit_aspect_ratio, measure_aspect_ratio
    from .dynamics import SpaceTimeCube
    cube = SpaceTimeCube(graph, tau=1.0)
    active = cube.active_points()
    before = measure_aspect_ratio(cube.positions, active)
    adjustment = None
    if fit == "exact":
        adjustment = fit_aspect_ratio(cube.positions, active, target)
        if adjustment["applied"]:
            cube.update_original()
    after = measure_aspect_ratio(cube.positions, active)
    actual_ratio = after["height_over_width"]
    graph.metadata["layout"]["aspect_ratio_result"] = {
        "target_height_over_width": float(target),
        "scope": "global XY bounding box of the complete space-time cube",
        "before_final_fit": before,
        "after_final_fit": after,
        "bbox_ratio_matches_target": (actual_ratio is not None
                                       and math.isclose(actual_ratio, target, rel_tol=1e-9, abs_tol=0)),
        "fit": fit, "adjustment": adjustment,
    }


def _validate_initial_ratio(initial_ratio: bool, aspect_ratio: float | None,
                            algorithm: str) -> None:
    """Reject unsupported initialization requests before graph or process work."""
    from .aspect_ratio import validate_aspect_ratio
    if not isinstance(initial_ratio, bool):
        raise ValueError("initial_ratio must be a boolean flag")
    validate_aspect_ratio(aspect_ratio)
    if initial_ratio and aspect_ratio is None:
        raise ValueError("initial_ratio (--initial-ratio) requires aspect_ratio (--aspect-ratio N)")


def layout(graph: TemporalGraph, delta: float = 5.0, tau: float | None = None,
           seed: int = 0, bend_transfer: bool = False, iterations: int = 75,
           algorithm: str = "multi", graphviz_engine: str = "sfdp",
           repulsion_backend: str = "auto", num_threads: int | None = None,
           aspect_ratio: float | None = None, aspect_ratio_fit: str = "exact",
           initial_ratio: bool = False, graphviz: bool = False) -> TemporalGraph:
    """Draw a copy of a temporal graph with a selectable static initializer.

    Multi-level uses GRIP, weighted barycenters, cumulative
    cooling, and flexible trajectories only at the finest level (every 30
    iterations). Single-level runs DynNoSlice with flexibility every iteration.
    NetworkX spring_layout initializes XY by default; graphviz=True selects
    Graphviz SFDP (or graphviz_engine). Single-level also uses this initializer,
    replacing any supplied positions. Aspect ratio N means width:height = 1:N.
    With N set, an ARCOL-derived force joins the original forces before the
    shared 3D movement constraints. An optional final box fit follows refinement.
    ``initial_ratio=True`` additionally adjusts initialization to that N, using
    Graphviz ratio=N or the same expand-only rule on Spring's node-center box.
    On static/sfdp runs it only sets the initializer ratio; no ARCOL force or
    final fit is applied. The legacy sfdp alias also respects graphviz selection.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite positive number")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if algorithm not in {"multi", "single", "static", "sfdp"}:
        raise ValueError("algorithm must be 'multi', 'single', or 'static'")
    if not isinstance(graphviz, bool):
        raise ValueError("graphviz must be a boolean flag")
    from .dynamics import _validate_repulsion_options, refine
    _validate_repulsion_options(repulsion_backend, num_threads)
    _validate_initial_ratio(initial_ratio, aspect_ratio, algorithm)
    if aspect_ratio_fit not in {"exact", "none"}:
        raise ValueError("aspect_ratio_fit must be 'exact' or 'none'")
    if aspect_ratio is not None and algorithm in {"static", "sfdp"} and not initial_ratio:
        raise ValueError("aspect_ratio requires the 'multi' or 'single' dynamic algorithm")
    static_only = algorithm in {"static", "sfdp"}
    drawn = copy.deepcopy(graph)
    drawn.validate()
    automatic_tau = tau is None
    if tau is None:
        tau = estimate_tau(drawn)
    else:
        # An input JSON can contain metadata from an earlier automatic run.
        drawn.metadata.pop("tau_estimation", None)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a finite positive number")
    metadata = drawn.metadata
    metadata["layout"] = {"algorithm": algorithm, "delta": float(delta), "tau": float(tau),
                           "tau_source": "automatic" if automatic_tau else "explicit",
                           "seed": int(seed), "bend_transfer": bend_transfer,
                           "graphviz": graphviz, "graphviz_engine": graphviz_engine,
                           "initial_ratio": initial_ratio,
                           "repulsion_backend": repulsion_backend,
                           "num_threads": num_threads,
                           "upstream_revision": "068aa79680b7d670d2338493bc9c88f4ffbd3db6"}
    # Coordinate consumers need tau independently of detailed layout statistics.
    metadata["tau"] = float(tau)
    if aspect_ratio is not None and not static_only:
        metadata["layout"].update(
            aspect_ratio=float(aspect_ratio), aspect_ratio_fit=aspect_ratio_fit,
            aspect_ratio_integration="additive_force",
            aspect_ratio_force_weight=1.0,
            aspect_ratio_reference={
                "paper_doi": "10.1111/cgf.70437",
                "source_download": "https://osf.io/download/3z4mh/",
                "source_sha256": "337135e7639c24638f0430e2d975e558658a9607c51e3f841a35fead414af64b",
            },
        )
    if not drawn.nodes:
        metadata["layout"].update(hierarchy_sizes=[], levels=[])
        metadata["layout"]["initialization"] = _initialize_layout(
            drawn, graphviz=graphviz, seed=seed, engine=graphviz_engine,
            initial_aspect_ratio=aspect_ratio if initial_ratio else None)
        if not static_only:
            _finish_aspect_ratio(drawn, aspect_ratio, aspect_ratio_fit)
        return drawn
    if static_only:
        metadata["layout"]["initialization"] = _initialize_layout(
            drawn, graphviz=graphviz, seed=seed, engine=graphviz_engine, level=0,
            initial_aspect_ratio=aspect_ratio if initial_ratio else None)
        metadata["layout"].update(hierarchy_sizes=[len(drawn.nodes)], levels=[])
        return drawn

    if algorithm == "single":
        metadata["layout"]["initialization"] = _initialize_layout(
            drawn, graphviz=graphviz, seed=seed, engine=graphviz_engine, level=0,
            initial_aspect_ratio=aspect_ratio if initial_ratio else None)
        stats = refine(drawn, tau=tau, delta=delta, max_movement=2.0 * delta,
                       iterations=iterations, flexible=True, flexible_interval=1, seed=seed,
                       repulsion_backend=repulsion_backend, num_threads=num_threads,
                       aspect_ratio=aspect_ratio)
        metadata["layout"].update(hierarchy_sizes=[len(drawn.nodes)], levels=[stats])
        _finish_aspect_ratio(drawn, aspect_ratio, aspect_ratio_fit)
        return drawn

    hierarchy = coarsen(drawn)
    logger.info("GRIP hierarchy: %s", " -> ".join(str(len(level.graph.nodes)) for level in hierarchy))
    metadata["layout"]["initialization"] = _initialize_layout(
        hierarchy[-1].graph, graphviz=graphviz, seed=seed, engine=graphviz_engine, level=len(hierarchy) - 1,
        initial_aspect_ratio=aspect_ratio if initial_ratio else None)
    schedule = cooling_schedule(len(hierarchy), delta, iterations)
    random = JavaRandom(seed)
    level_stats = []
    for round_index, level_index in enumerate(range(len(hierarchy) - 1, -1, -1)):
        level = hierarchy[level_index]
        if round_index:
            place_vertices(level.graph, hierarchy[level_index + 1], random, bend_transfer)
        movement, rounds = schedule[round_index]
        logger.info("Drawing level %s: %s nodes, %s edges, %s iterations",
                    level_index, len(level.graph.nodes), len(level.graph.edges), rounds)
        if round_index == 0 and len(level.graph.nodes) <= 1:
            stats = {"iterations": 0, "skipped": "coarsest graph has at most one node"}
        else:
            stats = refine(level.graph, tau=tau, delta=delta, max_movement=movement,
                           iterations=rounds, flexible=(level_index == 0),
                           flexible_interval=30, seed=seed + round_index,
                           repulsion_backend=repulsion_backend, num_threads=num_threads,
                           aspect_ratio=aspect_ratio)
        level_stats.append({"level": level_index, "nodes": len(level.graph.nodes),
                            "edges": len(level.graph.edges), "max_movement": movement,
                            "requested_iterations": rounds, "statistics": stats})
    drawn = hierarchy[0].graph
    drawn.metadata = metadata
    metadata["layout"].update(hierarchy_sizes=[len(level.graph.nodes) for level in hierarchy],
                               levels=level_stats)
    _finish_aspect_ratio(drawn, aspect_ratio, aspect_ratio_fit)
    return drawn
