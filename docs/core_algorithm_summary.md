# Core Algorithm Summary

This document summarizes the Java architecture that matters for a Python port of DynNoSlice and MultiDynNoS.

## Core Dynamic Graph Model

The static graph model lives in `ocotillo.graph`.

- `GraphWithElements` stores nodes and edges by string ID and maintains incoming/outgoing adjacency maps.
- `Node` and `Edge` are immutable ID-bearing elements. `Edge` stores source and target nodes.
- `GraphWithAttributes` adds graph, node, and edge attributes.
- `Graph` is the concrete static graph type.
- `StdAttribute` defines common attributes: `nodePosition`, `nodeSize`, `edgePoints`, `color`, `label`, `weight`, and dynamic-only `dyPresence`.

The dynamic model lives in `ocotillo.dygraph`.

- `DyGraph` is a graph whose graph/node/edge attributes store `Evolution<T>` values.
- `Evolution<T>` is a piecewise temporal value with a default value and a set of interval functions.
- `FunctionConst<T>` represents a constant value on one interval.
- `FunctionRect<T>` represents a left-to-right interpolated value on one interval.
- `Interval` supports open/closed bounds, infinite bounds, containment, intersection, union, and sampling.
- `DyGraph.snapshotAt(time)` builds a static graph with nodes and edges whose `dyPresence` evaluates to true, then snapshots all non-dynamic attributes.

For layout, the most important dynamic attributes are:

- Node `dyPresence`: when a node exists.
- Edge `dyPresence`: when an edge exists.
- Node `nodePosition`: a temporal 2D trajectory.
- Node/edge `weight`: used by static flattening and coarsening.

Tau is the time-to-space scale used in the space-time cube. The Java code can use dataset-provided tau, CLI tau, or `DyGraph.autocomputeTau(mode)`.

## Custom Node And Edge File Parsing

Custom input is handled by `Run`, `NodeAppearance`, and `EdgeAppearance`.

Node file format:

```text
<Node ID>,<Start Time>,<Duration>
Alice,1,5
Bob,2,4.6
```

Edge file format:

```text
<Source Node ID>,<Target Node ID>,<Start Time>,<Duration>
Alice,Bob,2.5,1
Bob,Carol,2.1,0.6
```

Parsing behavior:

- Each line is split by comma.
- Node lines produce `NodeAppearance(id, startTime, duration)`.
- Edge lines produce `EdgeAppearance(sourceId, targetId, startTime, duration)`.
- Bad lines are reported to stderr and skipped.
- Tokens are not trimmed in Java.

Graph construction behavior:

- Create each new node with label, false presence, and default position `(0, 0)`.
- Insert a closed true-presence interval `[start, start + duration]` for each node appearance.
- For each edge appearance, find source and target nodes, create one edge between the pair if missing, and insert a closed true-presence interval.
- Scatter initial node positions using Java `Random(73)` and a graph diameter estimate `sqrt(node_count * delta)`.

Validation methods exist but are not invoked in the Java constructor:

- Node appearances should be sorted and non-overlapping per node.
- Edge appearances should not be loops, should have source ID before target ID, and should be sorted/non-overlapping per edge.

Python should expose validation explicitly and compute tau from the custom graph when no CLI tau is supplied.

## GraphViz `fdp`/`sfdp` Usage

GraphViz integration is in `SfdpExecutor`.

Pipeline:

1. Read `resources/multidynnos.properties`.
2. Use `graphviz.prefix` plus either `fdp` or `sfdp`.
3. Check availability by running `<command> -V`.
4. Write the static graph to DOT.
5. Send DOT to GraphViz stdin.
6. Read DOT output from stdout.
7. Parse node `pos` attributes.
8. Copy generated positions back to the input graph.

DOT conversion is focused on:

- Node `nodePosition` <-> DOT `pos`.
- Node `nodeSize` <-> DOT `width,height`.
- Edge `weight` -> DOT `weight`.

GraphViz is used in two places:

- `SFDPRun`: flatten the whole dynamic graph, run static layout, copy static coordinates to every node trajectory.
- `MultiLevelDynNoSlice.nodesFirstPlacement()`: flatten the coarsest graph, lay it out statically, then transfer coordinates into the coarsest dynamic graph.

## Single-Level DynNoSlice Pipeline

Entry point: `DynNoSliceRun.run()`.

Configuration:

- `delta = 5.0` by default.
- `tau = 1.0`, dataset manual tau, CLI tau, or auto-computed tau.
- `iterations = 100`.

Algorithm builder:

```text
DyModularFdl(dygraph, tau)
  forces:
    TimeStraightning(delta)
    Gravity()
    ConnectionAttraction(delta)
    EdgeRepulsion(delta)
  constraints:
    DecreasingMaxMovement(2 * delta)
    MovementAcceleration(2 * delta, Geom.e3D)
  postprocessing:
    FlexibleTimeTrajectories(1.5 * delta, 2.0 * delta, Geom.e3D)
```

Execution flow:

1. `DyModularFdl` builds a `SpaceTimeCubeSynchroniser`.
2. The synchronizer converts each dynamic node appearance into a 3D mirror line.
3. Dynamic edge appearances become mirror connections between source and target node mirror lines.
4. The mirror graph is passed to `ModularFdl`.
5. `ModularFdl` creates a bend-explicit mirror graph so polyline bends become movable nodes.
6. Each iteration resets forces/constraints, rebuilds the locator, updates temperature, computes forces, applies constraints, runs pre-movement, moves nodes, runs post-processing, and syncs back.
7. At the end, `SpaceTimeCubeSynchroniser.updateOriginal()` writes mirror line polylines back as dynamic 2D `nodePosition` functions.

### Space-Time Cube Details

`SpaceTimeCubeSynchroniser` maps time to z-coordinate using `z = time * tau`.

For each node true-presence interval:

- Create mirror source and target nodes.
- Set their `(x, y, z)` from dynamic node position at interval bounds.
- Add bends for intermediate node-position function boundaries inside the appearance interval.
- Store a `MirrorLine` for interval queries.

For each edge true-presence interval:

- Create a `MirrorConnection`.
- Find the source and target node mirror lines containing the edge interval midpoint.
- Store the connection interval in time and mirror-space coordinates.

Updating original:

- For each original node, gather its mirror lines.
- Convert source, bends, and target positions back to 2D piecewise linear `FunctionRect.Coordinates` functions.

### Forces And Constraints

Default forces:

- `TimeStraightning`: smooths each node trajectory and discourages segments from becoming too horizontal in space-time.
- `Gravity`: pulls mirror nodes toward the initial 2D center.
- `ConnectionAttraction`: attracts source and target node trajectories while an original dynamic edge is present.
- `EdgeRepulsion`: repels different node trajectories/segments to reduce overlap and crowding.

Default constraints:

- `DecreasingMaxMovement`: global max movement decreases linearly with temperature.
- `MovementAcceleration`: limits movement when current force direction changes sharply from the previous movement.

Default pre-movement:

- `EnsureTimeCorrectness`: prevents endpoint z movement and keeps bend z-order valid.

Default post-processing:

- `FlexibleTimeTrajectories`: expands long trajectory segments by adding bends and contracts short bend chains by removing bends. This lets trajectories gain or lose detail during optimization.

## MultiDynNoS Multilevel Pipeline

Entry point: `MultiDynNoSliceRun.run()`.

Default Java configuration:

```text
MultiLevelDynNoSlice(dygraph, tau, delta)
  coarsener: IndependentSet()
  placement: WeightedBarycenterPlacementStrategy(bendTransfer)
  flattener: StaticSumPresenceFlattener()
  layout parameter minimums: LIMITED
  layer postprocessing: FlexibleTimeTrajectoriesPostProcessing(
      active_from_level=0,
      refresh_interval=TRAJECTORY_OPTIMIZATION_INTERVAL=30
  )
  single-level static layout: sfdp
  log option: true
```

Default dynamic parameters:

- Desired distance: `delta`.
- Initial max movement: `2 * delta`, cooled linearly with slope `-0.07`, lower bounded to `3`.
- Contract distance: `1.5 * delta`.
- Expand distance: `2 * delta`.
- Max iterations: `75`, cooled linearly with slope `-0.07`, lower bounded to `20`.

Execution flow:

1. Preprocess: pass the original graph to the coarsener.
2. Coarsening: build a hierarchy from finest to coarsest.
3. Initial placement: flatten the coarsest dynamic graph, run GraphViz `sfdp` or `fdp`, transfer static coordinates to the coarsest dynamic graph.
4. Start from the coarsest dynamic graph.
5. Run a single-level dynamic layout on that level if it has more than one node.
6. For each finer level:
   - Cool dynamic layout parameters.
   - Place fine-level vertices from the already laid-out coarser level.
   - Run single-level dynamic layout with cooled movement and iteration parameters.
   - Record stats.
7. Return the final finest-level dynamic graph.

### Coarsening

`GraphCoarsener.setGraph(original)` copies the original graph into hierarchy level `0`, translating node IDs as `<original_id>__0`.

At each new level:

- `computeNewVertexSet(lastLevel)` chooses groups and creates one node per group.
- `mergeNodePresenceAndWeight` OR-merges grouped node presence intervals and sums node weights.
- `generateEdges` maps lower-level edges to upper-level group leaders, merges parallel coarse edges, sums edge weights, and OR-merges edge presences.
- Stop if node count does not decrease, or if after the first level the new count is more than 95 percent of the previous count.

Default `IndependentSet`:

- Compute node and edge weights by summing presence durations.
- Sort nodes by node weight plus outgoing edge weights, descending.
- Pick the highest-weight remaining node as group leader.
- Group it with remaining neighbors reached by outgoing edges.

`WalshawIndependentSet` variant:

- Same as `IndependentSet`, but groups only one selected neighbor.

`SolarMerger`:

- Selects a sun node, groups immediate neighbors as planets, and neighbors of those neighbors as moons.
- Stores status on `SolarMerger_Status`.

### Placement

`transferCoordinatesFromStaticGraph` initializes coarsest dynamic node positions from GraphViz static positions.

`WeightedBarycenterPlacementStrategy` places each finer node:

- Find its upper-level group leader.
- Look at neighboring finer nodes that belong to other groups.
- Compute a weighted barycenter of neighboring group coordinates and own group coordinate.
- If there are no external groups, place the node on a random radius around its group leader.
- Add small random fuzziness.

Optional bend transfer:

- If the lower node is the upper-level leader's homolog, copy the upper evolution.
- Otherwise compute an initial coordinate and replay the upper trajectory deltas into a new evolution.

## Metrics

Metrics are orchestrated by `Experiment`.

### Stress

`GraphMetric.StressMetric` computes static stress:

```text
sum over connected node pairs ((a - b) / a)^2
```

Where:

- `a` is graph-theoretic shortest-path distance.
- `b` is Euclidean layout distance divided by the scaling factor.
- Disconnected pairs are ignored.

Dynamic stress uses `DyGraphMetric.AverageSnapshotMetricCalculation`:

- `StressOn`: average stress over the selected snapshot count.
- `StressOff`: average stress over `snap_count + (snap_count - 1) * 10` samples.

Before reporting, `Experiment.computeIdealScaling` tries scaling values `1.1^i` for `i = -20..20`, selects the best stress, then divides positions by that scaling. Output reports `1 / scaling`.

### Movement

`StcGraphMetric.AverageNodeMovement2D`:

- For each original node, collect all mirror-line bends and extremities.
- Sum 2D distances along the trajectory.
- Divide by original node count.

### Crowding

`StcGraphMetric.Crowding`:

- Sample 600 times over the suggested interval.
- For each snapshot, compare every node pair.
- A crowding event is counted when a pair newly overlaps.
- Overlap threshold is half the sum of the nodes' max size dimensions.

### Multilevel Stats

`MultilevelMetrics` records:

- Preprocess time.
- Coarsening time.
- Placement time.
- Hierarchy depth.
- Layout time placeholder.

Metrics mode in `DefaultRun` runs combinations of:

- Small datasets: Bunt, Newcomb, InfoVis, Rugby, Dialogs.
- Large datasets: Reality Mining, MOOC, College, RampInfectionMap.
- Layouts: single DynNoSlice, MultiDynNoS, flattened SFDP, Visone imports.
- MultiDynNoS variants: `wi_id`, `sm_sp`, `iset_grip`; static layouts `fdp` and `sfdp`.

## GUI And Rendering Classes To Ignore

These are not needed for a research Python port:

- `ocotillo.Gui`
- `ocotillo.gui.*`
- `ocotillo.gui.quickview.*`
- `ocotillo.dygraph.rendering.Animation`
- `ocotillo.graph.rendering.*`
- `ocotillo.graph.rendering.svg.*`
- `ocotillo.graph.rendering.image.*`

Optional output-only pieces:

- `GMLOutputWriter` if plot-slice export is needed.
- Dataset color/fade helpers if reproducing visuals matters.

## Classes That Must Be Faithfully Ported

Core:

- `Graph`, `Node`, `Edge`, `StdAttribute`
- `DyGraph`, `Evolution`, `Function`, `FunctionConst`, `FunctionRect`, `Interpolation`
- `Interval`, `Coordinates`, core `Geom` helpers

Single-level layout:

- `SpaceTimeCubeSynchroniser`
- `BendExplicitGraphSynchroniser`
- `ModularFdl`, `ModularForce`, `ModularConstraint`, `ModularThermostat`, `ModularStatistics`
- `DyModularFdl`
- `DyModularForce.TimeStraightning`, `Gravity`, `ConnectionAttraction`, `EdgeRepulsion`
- `DyModularPreMovement.EnsureTimeCorrectness`
- `DyModularPostProcessing.FlexibleTimeTrajectories`

Multilevel:

- `MultiLevelDynNoSlice`
- `DynamicLayoutParameter`
- `GraphCoarsener`
- `IndependentSet`
- `SolarMerger`
- `StaticSumPresenceFlattener`
- `MultilevelNodePlacementStrategy`
- `WeightedBarycenterPlacementStrategy`
- `MultiLevelCoolingStrategy`
- `MultiLevelDrawingOption`

Research metrics:

- `GraphMetric.StressMetric`
- `DyGraphMetric.AverageSnapshotMetricCalculation`
- `StcGraphMetric.AverageNodeMovement2D`
- `StcGraphMetric.Crowding`
- `DyGraphDiscretiser`

## Parts That Can Be Simplified

- Java builders can become Python functions or config dataclasses.
- Attribute generics can become typed dictionaries.
- Interval trees can initially be sorted lists.
- Spatial locator can initially be brute-force; optimize edge repulsion later.
- Dataset parsers can be optional plugins or examples.
- Full DOT parser/writer can be reduced to the subset needed for GraphViz layouts.
- Rendering, GUI, OCO, SVG, GMap, and defragmentation can be omitted from the core package.

