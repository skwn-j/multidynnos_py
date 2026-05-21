# multidynnos-py

Python research port scaffolding for MultiDynNoS dynamic graph experiments.

The package currently includes:

- Temporal intervals and presence schedules.
- Dynamic graph construction from the original MultiDynNoS custom input format.
- Edge-event input for text/CSV rows with source, target, and timestamp columns.
- Deterministic edge IDs for input edges that do not include IDs.
- `presence_at(t)`, `active_nodes(t)`, and `active_edges(t)`.
- CSV and JSON export of parsed graphs.
- A minimal single-level DynNoSlice-style dynamic layout.
- Snapshot, aspect-ratio scaling, and layout metrics for flat small-multiple studies.

## Custom Input Format

Node file:

```text
<Node ID>,<Start Time>,<Duration>
Alice,1,5
Bob,2,4.6
```

Edge file:

```text
<Source Node ID>,<Target Node ID>,<Start Time>,<Duration>
Alice,Bob,2.5,1
Bob,Carol,2.1,0.6
```

## Edge-Event Input

Single-file interaction logs are also supported. Rows may be whitespace-delimited or CSV:

```text
1 2 1082040961
3 4 1082155839
```

CSV files may include headers, including reordered columns:

```csv
timestamp,source,target
1082040961,1,2
1082155839,3,4
```

```python
from multidynnos_py import load_edge_event_graph

graph = load_edge_event_graph("tests/test_data.txt")
csv_graph = load_edge_event_graph("events.csv", has_header=True)
```

## CLI

```bash
multidynnos-py parse nodes.csv edges.csv --json graph.json
multidynnos-py parse nodes.csv edges.csv --nodes-csv out_nodes.csv --edges-csv out_edges.csv
multidynnos-py parse nodes.csv edges.csv --at 2.5
multidynnos-py layout nodes.csv edges.csv --method single --json single_layout.json
multidynnos-py layout nodes.csv edges.csv --method multi --json multi_layout.json
multidynnos-py layout-events raw/rugby.csv --method multi
multidynnos-py layout-events raw/rugby.csv --method multi --visualize 8
multidynnos-py layout-events raw/rugby.csv --method multi --visualize 8 --show-labels
multidynnos-py layout-events raw/collegemsg.csv --method multi --time-bins 100 --visualize 8
```

When working in the project conda environment:

```bash
conda run -n aspgd multidynnos-py parse nodes.csv edges.csv --json graph.json
conda run -n aspgd multidynnos-py layout nodes.csv edges.csv --method multi --json layout.json
conda run -n aspgd python -m multidynnos_py.cli layout-events raw/rugby.csv --method multi
conda run -n aspgd python -m multidynnos_py.cli layout-events raw/rugby.csv --method multi --visualize 8
conda run -n aspgd python -m multidynnos_py.cli layout-events raw/collegemsg.csv --method multi --time-bins 100 --visualize 8
conda run -n aspgd python -m pytest
```

When a layout command is run without `--json`, the layout is written to `output/<dataset-name>/layout.json`.
When `--visualize N` is supplied to `layout` or `layout-events`, the command also renders `N` edge-aware snapshots to `output/<dataset-name>/figures/`.
For large event datasets, `--time-bins N` aggregates raw event timestamps into `N` uniform temporal bins before layout; if omitted, the original event timestamps are used.
When `--time-bins N` is used, the binned static graph for each bin is also written as headerless directed `src,dst,count` CSV files:

```text
output/<dataset-name>/snapshots/
  snapshot_000.csv
  snapshot_001.csv
```

## Layout And Small-Multiple Export

```python
from pathlib import Path

from multidynnos_py import (
    DynNoSliceConfig,
    MultiDynNoSConfig,
    export_small_multiples_data,
    load_custom_graph,
    run_multidynnos,
)

graph = load_custom_graph("nodes.csv", "edges.csv")
layout = run_multidynnos(
    graph,
    MultiDynNoSConfig(dynnoslice_config=DynNoSliceConfig(iterations=50, seed=73)),
)
export_small_multiples_data(layout, times=[0.0, 1.0, 2.0], out_path=Path("small_multiples.json"))
```

Aspect-ratio utilities can be used directly:

```python
from multidynnos_py import affine_scale_layout, compute_aspect_ratio

wide_layout = affine_scale_layout(layout, target_aspect_ratio=4.0)
ratio = compute_aspect_ratio(wide_layout)
```

Available research metrics:

```python
from multidynnos_py import crowding_metric, movement_metric, sample_snapshots

snapshots = sample_snapshots(layout, [0.0, 1.0, 2.0])
movement = movement_metric(layout)
crowding = crowding_metric(snapshots[0])
```

## Rendering Static Snapshots From A Layout JSON

Render uniformly spaced static node-link snapshots from an existing layout JSON:

```bash
python -m multidynnos_py.cli render-snapshots \
  --layout output/rugby/layout.json \
  --num-snapshots 8 \
  --edges raw/rugby.csv \
  --show-labels \
  --output-root output
```

By default, each snapshot uses `--position-policy mean_in_window`, so node positions are the mean of that node's layout samples inside the temporal bin.

If the layout JSON has no edge data and no edge CSV is available, render nodes only explicitly:

```bash
python -m multidynnos_py.cli render-snapshots \
  --layout output/rugby/layout.json \
  --num-snapshots 8 \
  --allow-nodes-only \
  --output-root output
```

Figures and metadata are written to:

```text
output/<dataset-name>/figures/
  snapshot_000.png
  snapshot_001.png
  snapshots_manifest.json
```
