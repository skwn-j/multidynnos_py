# Implementation Notes

## Minimal DynNoSlice-Style Layout

The Python implementation in `multidynnos_py.layout` is intentionally a minimal single-level dynamic layout. It is inspired by Java `DynNoSliceRun` and the force classes under `ocotillo.dygraph.layout.fdl.modular`, but it does not yet port the Java space-time-cube mirror graph, bend-explicit graph synchronizer, spatial locator, or exact force formulas.

Implemented behavior:

- Each active node is represented as one sampled `(x, y)` point per sampled time.
- Sample times are deterministic: all finite node and edge interval starts, ends, and midpoints.
- Initial positions are deterministic for a given seed.
- Iterative forces conceptually match the Java defaults:
  - gravity pulls nodes toward the current time-slice centroid,
  - connection attraction treats active edges as springs with preferred distance `delta`,
  - repulsion pushes active nodes apart within each sampled time slice,
  - temporal smoothing straightens each node trajectory across sampled times.
- Movement is capped per iteration to keep tiny examples stable.
- Results are exposed through `run_dynnoslice(graph, config) -> DynamicLayout` and can be serialized with `DynamicLayout.to_dict()`.

Key differences from Java:

- Java DynNoSlice converts the dynamic graph to a 3D space-time cube where time is the z-axis. This Python version keeps sampled 2D positions and uses `tau` only as the temporal smoothing weight.
- Java edge attraction operates between trajectory segments over edge-presence intervals. This Python version applies attraction only at sampled times where the edge is active.
- Java edge repulsion operates on space-time trajectory segments with a spatial locator. This Python version applies pairwise node repulsion inside each sampled time slice.
- Java temporal straightening includes angle and smoothing components in 3D. This Python version moves each sampled point toward a linear interpolation between neighboring sampled points.
- Java flexible trajectories add and remove bends dynamically. This minimal version uses a fixed sample set.
- Java uses `ModularFdl` with thermostats and movement acceleration constraints. This version uses a simple learning rate and max-step clamp.

These differences keep the Phase 4 implementation small, deterministic, and testable while preserving the high-level purpose of DynNoSlice: balance graph readability at active times with smooth node movement over time.

## Minimal MultiDynNoS-Style Multilevel Pipeline

The Python implementation in `multidynnos_py.layout.multidynnos` is a first working multilevel pipeline modeled on the Java `MultiDynNoSliceRun` configuration, with simpler data structures and deterministic fallbacks.

Implemented behavior:

- `StaticSumPresenceFlattener` converts a dynamic graph to weighted static node and edge summaries by summing temporal presence durations.
- `IndependentSetCoarsener` builds a greedy maximal independent-set hierarchy and records both `fine_to_coarse` and `coarse_to_fine` mappings for every transition.
- The coarsest level is initialized from a static layout. If `use_sfdp=True` and GraphViz `sfdp` is available, the flattened static graph is sent to `sfdp -Tplain`; otherwise a deterministic circular layout is used.
- Refinement proceeds from coarse to fine levels. Expanded nodes are placed with `WeightedBarycenterPlacementStrategy`, which combines the parent supernode position with weighted neighbor-parent barycenters and a deterministic small offset for siblings.
- Each level is refined with the existing `run_dynnoslice` force loop. `run_dynnoslice` now accepts an optional `initial_layout` while preserving the original `run_dynnoslice(graph, config)` API.
- A light fixed-sample trajectory smoothing pass emulates the Java flexible trajectory post-processing stage.
- The public API is `run_multidynnos(graph, config) -> DynamicLayout`, and the CLI exposes `multidynnos-py layout ... --method single|multi`.

Key simplifications from Java:

- The Java independent-set coarsener has richer graph-bundle bookkeeping. This version preserves node-to-supernode mappings but does not yet preserve detailed edge lineage.
- The Java static flattener and GraphViz integration carry more rendering and attribute metadata. This version uses only summed weights and parses GraphViz plain positions when available.
- Java weighted barycenter placement accounts for the full graph hierarchy and richer placement constraints. This version uses parent positions, weighted flattened neighbors, and deterministic sibling offsets.
- Java flexible trajectories can add and remove bends. This version keeps the same sampled time set and smooths only interior sampled points.
- Java uses the full DynNoSlice engine at every level. This version reuses the minimal Python DynNoSlice approximation described above.
