# Porting Plan

Goal: port MultiDynNoS from Java to a Python research package without carrying over GUI-heavy Ocotillo infrastructure that is not needed for algorithmic experiments.

## Scope

In scope:

- Custom dynamic graph input from node and edge appearance CSV files.
- Core dynamic graph model: temporal presence, temporal positions, snapshots, tau calculation.
- Single-level DynNoSlice layout.
- MultiDynNoS multilevel layout.
- GraphViz `fdp`/`sfdp` integration.
- Research metrics: stress, movement, crowding, hierarchy/coarsening stats.
- Optional reproduction helpers for selected built-in datasets.

Out of scope for the MVP:

- Swing GUI, QuickView, animation windows, canvas controls.
- SVG/image rendering and GMap/OCO serialization.
- Full Ocotillo graph library API compatibility.
- Static layout utilities not used by the DynNoSlice or MultiDynNoS pipelines.

## Fidelity Strategy

Port the algorithms in two layers:

1. Behavior-compatible core: data model, interval semantics, STC transform, forces, constraints, coarsening, placement, and metrics.
2. Pythonic shell: dataclasses, type hints, small configuration objects, standard logging, testable functions instead of Java-style builders everywhere.

The first implementation should favor correctness over speed. After reproducing small Java outputs or metric ranges, optimize interval queries and edge-repulsion spatial lookup.

## Phase Plan

### Phase 1: Core Model

Implement:

- `Coordinates` and `Interval` with open/closed boundary semantics.
- `FunctionConst`, `FunctionRect`, `Evolution`, and interpolation.
- Static `Graph`, `Node`, `Edge`, standard attributes.
- `DyGraph` with dynamic attributes and `snapshot_at(time)`.
- Basic `EvolutionAnalyser.get_intervals_with_value`.

Verification:

- Unit tests for interval containment/intersection/sample.
- Unit tests for piecewise evolution lookup, insert order, default value.
- Snapshot tests on a tiny dynamic graph.

### Phase 2: Input And Baseline Static Layout

Implement:

- `parse_node_appearances(lines)` and `parse_edge_appearances(lines)`.
- `create_dynamic_graph(node_appearances, edge_appearances, delta=5.0)`.
- `StaticSumPresenceFlattener`.
- GraphViz adapter for `fdp`/`sfdp`: write DOT, run executable, parse node `pos`.

Verification:

- Tiny custom graph fixture matching README format.
- Validate that flattened node/edge weights equal summed presence durations.
- GraphViz smoke test should be skipped cleanly if `sfdp` is unavailable.

### Phase 3: Space-Time Cube And Single-Level DynNoSlice

Implement:

- `SpaceTimeCubeSynchroniser`.
- `BendExplicitGraphSynchroniser`.
- `ModularFdl` iteration loop.
- Dynamic forces: time straightening, gravity, connection attraction, edge repulsion.
- Constraints: decreasing max movement, movement acceleration.
- Pre-movement: ensure time correctness.
- Post-processing: flexible time trajectories.
- `dynnoslice_layout(graph, delta=5.0, tau=..., iterations=100)`.

Verification:

- Mirror graph construction on one node with two position functions.
- Mirror connections on one dynamic edge.
- Iteration smoke test: finite coordinates, time order preserved, original graph updated.

### Phase 4: MultiDynNoS

Implement:

- `GraphCoarsener` base and node ID translation.
- `IndependentSet` coarsener, including Walshaw variant.
- `SolarMerger` coarsener.
- Presence merging and coarse edge generation.
- `WeightedBarycenterPlacementStrategy` and identity placement.
- Optional bend transfer.
- `MultiLevelDynNoSlice.run()`.

Verification:

- Coarsening shrink tests on small graphs.
- Presence-merge tests for overlapping and adjacent intervals.
- End-to-end multilevel smoke test on a small dynamic graph.

### Phase 5: Metrics And Experiments

Implement:

- Static graph stress metric.
- Average snapshot metric.
- STC average node movement.
- Crowding metric.
- Ideal scaling search.
- Discretization by snapshot times.
- Optional experiment runner for Java's benchmark combinations.

Verification:

- Stress on simple path/triangle fixtures.
- Movement on known piecewise paths.
- Crowding on two moving nodes with known overlap transitions.

### Phase 6: Dataset Parsers, CLI, Packaging

Implement only after the algorithmic path is stable:

- Selected dataset parsers from `ocotillo.samples.parsers`.
- CLI equivalent for `animate`, `showcube`, `metrics`, and `dump`, minus GUI rendering.
- Package metadata and documentation.

## Java Behaviors To Preserve Or Deliberately Change

- Custom node lines are parsed as `<node_id>,<start_time>,<duration>`.
- Custom edge lines are parsed as `<source_id>,<target_id>,<start_time>,<duration>`.
- The Java parser uses `String.split(",")` with no trimming and prints errors per bad line. Python should parse this format but can offer clearer errors.
- `Run.checkNodeAppearanceCorrectness` and `Run.checkEdgeAppearanceCorrectness` exist but are not called by the constructor. Python should expose validation and decide whether to make it default.
- Java edge correctness requires source ID lexicographically before target ID and forbids loops, but this is not enforced in the actual custom load path.
- Java custom graph loading appears to need special care around tau: after creating a custom graph, `Run` still asks `requestedDataSet` for tau unless CLI tau is supplied. Python should compute tau from the created graph by default.
- `DyGraph.autocomputeTau` should be ported with tests. The edge time-span update in Java looks suspicious, so preserve behavior only if matching Java output is the goal; otherwise document a corrected formula.
- Randomness is mixed: initial scatter uses `Random(73)`, while multilevel placement uses `Math.random()`. Python should accept an explicit RNG for reproducible research.
- Keep the misspelled Java concept `ForbidTimeShitfing` as a compatibility alias, but expose a corrected Python name.

## Dependency Plan

No Python libraries were installed during this analysis. When implementation starts, install external Python libraries only into the `aspgd` conda environment, as requested.

Suggested minimal dependencies:

- Required initially: Python standard library only, plus external GraphViz executable for actual `sfdp`/`fdp` layouts.
- Useful after core tests: `numpy` for vector arrays and stable numeric operations.
- Optional for metrics/performance: `networkx` for shortest paths and graph utilities, or keep a small Floyd-Warshall implementation for parity.
- Optional for DOT parsing: `pydot` if the focused parser becomes brittle. A small parser is enough for GraphViz `pos` output in the MVP.

## Recommended Python Architecture

Use this package structure:

```text
multidynnos/
  core/
  io/
  graphviz.py
  layout/
  multilevel/
  metrics.py
  discretize.py
  experiments.py
  cli.py
tests/
```

Keep the public API small:

```python
from multidynnos.io.custom import load_custom_graph
from multidynnos.layout.dynnoslice import dynnoslice_layout
from multidynnos.multilevel.pipeline import multidynnos_layout
from multidynnos.metrics import compute_research_metrics
```

The core package should not depend on GraphViz, plotting, or dataset-specific code. The layout packages may depend on the core model. The multilevel package may depend on layout and graphviz. Metrics may depend on layout only for STC synchronizer objects.

## First Three Implementation Steps

1. Implement the core dynamic graph model and custom parser.
   Build `Interval`, `Coordinates`, `Evolution`, `Graph`, `DyGraph`, and `load_custom_graph`, then test snapshots and presence intervals.

2. Implement static flattening plus GraphViz integration.
   Add `StaticSumPresenceFlattener` and a focused `sfdp` subprocess adapter so the coarsest-level initialization can work.

3. Implement single-level DynNoSlice before multilevel.
   Build the STC synchronizer, modular FDL loop, default forces/constraints/post-processing, and a small end-to-end layout smoke test.

## Testing Strategy

- Start with pure unit tests that do not require GraphViz.
- Mark GraphViz-dependent tests as integration tests.
- Create tiny dynamic graph fixtures by hand; avoid large repository datasets until the core is stable.
- Once Java can be built, generate golden CSV outputs from small Java runs and compare Python metric ranges rather than exact coordinates, because Java placement includes nondeterministic `Math.random()`.
- Test interval boundary cases explicitly. Open/closed intervals affect node/edge presence and metric snapshots.

## Files To Ignore During Port

Ignore or defer:

- `source/ocotillo/gui/**`
- `source/ocotillo/gui/quickview/**`
- `source/ocotillo/graph/rendering/**`
- `source/ocotillo/dygraph/rendering/Animation.java`
- `source/ocotillo/Gui.java`
- `source/ocotillo/serialization/oco/**`
- `source/ocotillo/serialization/SvgReader.java`
- `source/ocotillo/graph/layout/other/gmap/**`
- `source/ocotillo/graph/layout/fdl/defragmenter/**`
- Most of `source/ocotillo/various/**`

Keep optional:

- `source/ocotillo/export/GMLOutputWriter.java` only if slice dumping is needed.
- `source/ocotillo/samples/parsers/**` only for benchmark reproduction.

