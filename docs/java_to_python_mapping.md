# Java to Python Mapping

This document maps the Java repository `EngAAlex/MultiDynNos` to a research-oriented Python package. The Java source was inspected read-only from `/tmp/MultiDynNos`; no Java files were edited.

## Repository Shape

MultiDynNoS is a Java 11/Maven project whose main class is `ocotillo.DefaultRun`. It vendors a substantial subset of the Ocotillo graph library, then adds MultiDynNoS-specific packages under `ocotillo.multilevel` and `ocotillo.graph.multilevel.layout`.

Build/runtime dependencies in `pom.xml`:

- `commons-exec`: runs external GraphViz commands.
- `commons-cli`: present for CLI support, though most CLI parsing is manual.
- `commons-csv`: dataset parsing support.
- `lombok`: equality/hash helpers on some value classes.
- `ejml`: matrix support, mostly outside the core MultiDynNoS path.
- `GraphViz`: external `fdp`/`sfdp` executable, configured by `resources/multidynnos.properties`.

## Package Responsibilities

| Java package | Key classes | Responsibility | Python port decision |
| --- | --- | --- | --- |
| `ocotillo` | `DefaultRun`, `Experiment`, `Gui` | CLI entry point, experiment orchestration, GUI launcher. | Port `DefaultRun` behavior as a thin CLI later. Port `Experiment` metrics logic selectively. Ignore `Gui`. |
| `ocotillo.run` | `Run`, `DynNoSliceRun`, `MultiDynNoSliceRun`, `SFDPRun` | Runtime setup, custom file loading, delta/tau/options, dispatch to layout algorithms. | Port as `pipelines` plus optional CLI. Keep behavior but avoid Java's null dataset coupling. |
| `ocotillo.run.customrun` | `NodeAppearance`, `EdgeAppearance` | Parses custom CSV node and edge appearance files. | Faithfully port, with optional stricter validation behind a flag. |
| `ocotillo.graph` | `Graph`, `Node`, `Edge`, attributes, `StdAttribute` | Static graph core: element IDs, adjacency, typed graph/node/edge attributes. | Port the data model, but simplify Java generics and observer machinery. |
| `ocotillo.graph.extra` | `GraphMetric`, `BendExplicitGraphSynchroniser`, lookup helpers | Static metrics and bend-explicit mirror graph support. | Port `GraphMetric` and `BendExplicitGraphSynchroniser`; skip lookup helpers unless locator performance needs them. |
| `ocotillo.graph.layout` | `Layout2D`, `LayoutXD` | Small static layout wrappers. | Ignore initially. |
| `ocotillo.graph.layout.fdl.modular` | `ModularFdl`, `ModularForce`, `ModularConstraint`, `ModularThermostat`, `ModularStatistics` | Generic force-directed iteration loop over a static mirror graph. | Faithfully port the iteration loop, constraints, thermostat, stats, and base extension points. |
| `ocotillo.graph.layout.fdl.sfdp` | `SfdpExecutor` | Writes DOT, runs GraphViz `fdp`/`sfdp`, reads positions back. | Port as a focused GraphViz adapter. |
| `ocotillo.graph.layout.fdl.defragmenter` | `Defragmenter`, `ClusterPlacer`, `NodePlacer` | Static defragmentation utilities. | Ignore for core MultiDynNoS. |
| `ocotillo.graph.layout.locator` | `ElementLocator`, `ElementLocatorAbst` | Spatial lookup API for force calculations. | Port only the query API needed by edge repulsion. |
| `ocotillo.graph.layout.locator.intervaltree` | `IntervalTreeLocator` | Spatial index implementation for nodes/edges. | Simplify at first with brute-force or grid lookup; add an optimized locator later. |
| `ocotillo.graph.layout.or.repulsion` | `RepulsionOverlapRemover` | Static overlap removal. | Ignore initially. |
| `ocotillo.graph.layout.other.gmap` | `GmapExecutor` | External GMap layout integration. | Ignore. |
| `ocotillo.graph.multilevel.layout` | `MultiLevelDynNoSlice`, `DynamicLayoutParameter` | Main multilevel layout driver and per-level cooled parameters. | Faithfully port. This is the MultiDynNoS core. |
| `ocotillo.graph.rendering` | renderers, `HeatMap`, `ViewAngle`, `YuvColor` | Swing/2D/3D rendering helpers. | Ignore except minimal color/value helpers if needed by datasets. |
| `ocotillo.graph.rendering.svg` | `SvgDocument`, `SvgExporter`, `SvgElement` | SVG output/rendering. | Ignore. |
| `ocotillo.graph.rendering.image` | `ImageExporter` | Image export. | Ignore. |
| `ocotillo.dygraph` | `DyGraph`, `Evolution`, `Function`, `FunctionConst`, `FunctionRect`, dynamic attributes | Dynamic graph model, piecewise temporal attributes, snapshots, tau computation. | Faithfully port core classes with Pythonic dataclasses. |
| `ocotillo.dygraph.extra` | `SpaceTimeCubeSynchroniser`, `DyGraphDiscretiser`, `DyGraphMetric`, `StcGraphMetric`, `EvolutionAnalyser`, `DyClustering`, `DyFlattening` | Space-time cube transform, discretization, dynamic metrics, clustering. | Port synchronizer, discretizer, metrics, evolution analysis. Defer clustering and extra flattening. |
| `ocotillo.dygraph.layout.fdl.modular` | `DyModularFdl`, `DyModularForce`, `DyModularPreMovement`, `DyModularPostProcessing`, `DyModularMetric` | Dynamic layout implemented by converting to a 3D mirror graph and running modular FDL. | Faithfully port. This is the DynNoSlice core. |
| `ocotillo.dygraph.rendering` | `Animation` | Animation time control. | Ignore. |
| `ocotillo.multilevel` | `MultilevelMetrics` | Metrics for preprocess, coarsening, placement, hierarchy depth. | Port lightweight stats only. |
| `ocotillo.multilevel.coarsening` | `GraphCoarsener`, `IndependentSet`, `SolarMerger` | Builds dynamic graph hierarchy by grouping nodes and merging presences/edges. | Faithfully port `GraphCoarsener` and default `IndependentSet`; port `SolarMerger` for experiments. |
| `ocotillo.multilevel.placement` | `MultilevelNodePlacementStrategy`, `WeightedBarycenterPlacementStrategy` | Transfers positions from coarse to fine levels, with optional bend transfer. | Faithfully port default weighted barycenter and identity placement. |
| `ocotillo.multilevel.flattener` | `DyGraphFlattener.StaticSumPresenceFlattener` | Converts a dynamic graph to a weighted static graph by summing presence durations. | Faithfully port. |
| `ocotillo.multilevel.cooling` | `MultiLevelCoolingStrategy` | Parameter cooling across hierarchy levels. | Port identity, linear, inverse exponential strategies. |
| `ocotillo.multilevel.options` | `MultiLevelDrawingOption` | Activates post/pre-processing options by hierarchy level. | Port minimal option mechanism. |
| `ocotillo.multilevel.logger` | `Logger` | Singleton console logger. | Replace with Python `logging`. |
| `ocotillo.geometry` | `Coordinates`, `Interval`, `Geom*`, `Box`, `Polygon`, `Circle` | Numeric geometry, intervals, vector math, collision/segment relations. | Port `Coordinates`, `Interval`, 2D/3D vector operations, boxes. Defer full polygon/circle support. |
| `ocotillo.structures` | `IntervalTree`, `MultidimIntervalTree`, red-black tree | Interval storage and spatial indexing support. | Use sorted lists first; add interval tree only if performance requires it. |
| `ocotillo.serialization` | `ParserTools`, `SvgReader` | File helpers and SVG parsing. | Port simple file helpers only if needed. |
| `ocotillo.serialization.dot` | `DotReader`, `DotWriter`, converters | DOT serialization for GraphViz round trips. | Port a narrow DOT writer/reader for `pos`, `width`, `height`, `weight`. |
| `ocotillo.serialization.oco` | `OcoReader`, `OcoWriter`, converters | Ocotillo native serialization. | Ignore. |
| `ocotillo.export` | `GMLOutputWriter` | GML output for slice dumps. | Optional, not core. |
| `ocotillo.gui` | `GraphCanvas`, listeners, controls | Swing UI controls. | Ignore. |
| `ocotillo.gui.quickview` | `QuickView`, `DyQuickView` | Swing graph views. | Ignore. |
| `ocotillo.samples` | `GraphSamples`, `DyGraphSamples` | Sample graph factories. | Ignore unless needed for tests. |
| `ocotillo.samples.parsers` | `Commons`, dataset parsers | Preloaded dataset readers and normalization. | Port `Commons` helpers and selected parsers only when reproducing experiments. |
| `ocotillo.various` | color collections, point clustering | Misc rendering/clustering helpers. | Ignore except optional color palettes for dataset parity. |

## Class-Level Porting Map

| Java class/group | Python module target | Port priority | Notes |
| --- | --- | --- | --- |
| `Node`, `Edge`, `Graph`, attributes, `StdAttribute` | `multidynnos.core.graph` | Required | Use strings for IDs, adjacency dicts, and attribute dicts. Avoid Java observer/bulk notification unless needed. |
| `Interval`, `Coordinates`, `Geom.e2D/e3D` | `multidynnos.core.geometry` | Required | Boundary semantics matter for dynamic presence and snapshots. |
| `Evolution`, `FunctionConst`, `FunctionRect`, `Interpolation` | `multidynnos.core.evolution` | Required | Piecewise time functions are the core abstraction. |
| `DyGraph`, `DyNodeAttribute`, `DyEdgeAttribute`, `DyGraphAttribute` | `multidynnos.core.dynamic_graph` | Required | A dynamic attribute is an `Evolution[T]` per node/edge/graph value. |
| `EvolutionAnalyser` | `multidynnos.core.evolution_analysis` | Required | Needed to extract true-presence intervals and merge presences. |
| `SpaceTimeCubeSynchroniser` | `multidynnos.layout.space_time_cube` | Required | Creates mirror lines and mirror connections and writes final 2D trajectories back. |
| `BendExplicitGraphSynchroniser` | `multidynnos.layout.bends` | Required | Required by `ModularFdl` to expose bends as movable graph nodes. |
| `ModularFdl` and base modular elements | `multidynnos.layout.modular` | Required | Generic iteration loop: forces, constraints, movement, pre/post steps, stats. |
| `DyModularFdl` | `multidynnos.layout.dynnoslice` | Required | Builds STC mirror graph, delegates to `ModularFdl`, then updates original dynamic graph. |
| `DyModularForce` | `multidynnos.layout.forces` | Required | Port `TimeStraightning`, `Gravity`, `ConnectionAttraction`, `EdgeRepulsion`; `MentalMapPreservation` is optional because the default pipeline does not use it. |
| `DyModularPreMovement` | `multidynnos.layout.pre_movement` | Required | Port `EnsureTimeCorrectness` and `ForbidTimeShitfing` behavior. Use a corrected Python name plus compatibility alias. |
| `DyModularPostProcessing` | `multidynnos.layout.postprocess` | Required | Port `FlexibleTimeTrajectories` for bend expansion/contraction. |
| `SfdpExecutor`, DOT converters | `multidynnos.graphviz` | Required | Needed for MultiDynNoS initial coarse layout and SFDP baseline. |
| `DyGraphFlattener.StaticSumPresenceFlattener` | `multidynnos.multilevel.flattener` | Required | Produces weighted static graph and aggregated weights. |
| `GraphCoarsener`, `IndependentSet`, `SolarMerger` | `multidynnos.multilevel.coarsening` | Required | Default run uses `IndependentSet`; experiment grid also uses Walshaw and SolarMerger. |
| `MultilevelNodePlacementStrategy`, `WeightedBarycenterPlacementStrategy` | `multidynnos.multilevel.placement` | Required | Transfers coordinates from coarser layouts to finer levels. |
| `DynamicLayoutParameter`, `MultiLevelCoolingStrategy`, `MultiLevelDrawingOption` | `multidynnos.multilevel.options` | Required | Controls per-level max movement, iteration count, and trajectory post-processing. |
| `MultiLevelDynNoSlice` | `multidynnos.multilevel.pipeline` | Required | Main multilevel algorithm. |
| `GraphMetric`, `DyGraphMetric`, `StcGraphMetric`, `MultilevelMetrics` | `multidynnos.metrics` | Required for research use | Stress, movement, crowding, runtime/hierarchy stats. |
| `DyGraphDiscretiser` | `multidynnos.discretize` | Required for metrics and discrete experiments | Preserve snap-time graph attribute behavior or expose structured metadata. |
| `NodeAppearance`, `EdgeAppearance`, `Run.createDynamicGraph` | `multidynnos.io.custom` | Required | Custom CSV input path. |
| Dataset parsers | `multidynnos.io.datasets` | Optional phase | Useful for reproduction but not needed for the first algorithmic port. |
| GUI/rendering/export/OCO/GMap/defragmenter | none or optional extras | Not core | Omit from the research package MVP. |

## Proposed Python Package Layout

```text
multidynnos/
  core/
    geometry.py          # Coordinates, Interval, 2D/3D geometry helpers
    graph.py             # Node, Edge, Graph, attributes, StdAttribute
    evolution.py         # Evolution, FunctionConst, FunctionRect, interpolation
    dynamic_graph.py     # DyGraph and dynamic attribute helpers
    evolution_analysis.py
  io/
    custom.py            # node/edge appearance CSV parser
    datasets.py          # optional preloaded datasets
  graphviz.py            # fdp/sfdp subprocess adapter
  layout/
    bends.py             # bend-explicit mirror graph
    space_time_cube.py   # STC synchronizer
    modular.py           # generic force-directed engine
    dynnoslice.py        # single-level dynamic layout builder
    forces.py
    constraints.py
    pre_movement.py
    postprocess.py
  multilevel/
    flattener.py
    coarsening.py
    placement.py
    options.py
    pipeline.py          # MultiDynNoS
  metrics.py
  discretize.py
  experiments.py         # optional benchmark orchestration
  cli.py                 # optional command-line wrapper
```

## Data Structure Mapping

| Java model | Python shape |
| --- | --- |
| `Element`, `Node`, `Edge` objects with string IDs | Frozen or simple dataclasses keyed by `id`; edges store `source_id` and `target_id`. |
| `GraphWithElements` maps and adjacency sets | `nodes: dict[str, Node]`, `edges: dict[str, Edge]`, `out_edges/in_edges: dict[str, set[str]]`. |
| `GraphWithAttributes` with generic Java attributes | `graph_attrs`, `node_attrs`, `edge_attrs` dictionaries mapping attribute names to default plus per-element values. |
| `StdAttribute` enum | Python `Enum` or string constants. Keep names exactly for compatibility. |
| `Evolution<T>` with `IntervalTree<Function<T>>` | `Evolution[T]` with sorted list of functions for MVP; replace with interval tree if profiling requires. |
| `FunctionConst`, `FunctionRect` | Dataclasses with `interval`, `left_value`, `right_value`, `value_at(t)`. |
| `DyGraph.snapshotAt(t)` | Method returning static `Graph` containing nodes/edges whose `dyPresence` is true at `t`, with non-dynamic attrs snapped. |
| `SpaceTimeCubeSynchroniser.MirrorLine` | Dataclass for one node appearance interval with mirror source, target, bends, and conversion between time and z. |
| `SpaceTimeCubeSynchroniser.MirrorConnection` | Dataclass for one edge appearance interval linking source/target mirror lines. |
| `ModularFdl` | Iterative engine over a mirror graph with pluggable forces/constraints and movement arrays. |

## Simplifications For A Research Python Package

- Replace Java's deeply generic attribute hierarchy with typed dictionaries and small helper methods.
- Replace Swing rendering and QuickView classes with optional plotting utilities outside the core algorithm.
- Replace the full DOT/OCO serialization stack with a targeted GraphViz round trip for node coordinates and weights.
- Use sorted interval lists initially. MultiDynNoS spends most time in layout forces; interval tree complexity can wait until profiling shows it matters.
- Make randomness injectable. Java uses a fixed seed for initial scatter (`73`) but uses `Math.random()` in multilevel placement, so deterministic research runs need an explicit RNG.
- Keep GraphViz as an external executable dependency. Python wrappers are optional conveniences, but the Java behavior is subprocess-based.
- Preserve algorithmic defaults first, then expose Pythonic configuration objects.

