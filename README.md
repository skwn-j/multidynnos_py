# MultiDynNoS Python

A Python implementation of MultiDynNoS for drawing temporal graphs as continuous
node trajectories. It reads event or timesliced input, computes a multilevel
layout, and exports a space-time cube as TimeLighting JSON. It also supports
ARCOL-based aspect ratio adjustment and optional Numba acceleration for repulsion.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── NOTICE
├── requirements.txt
├── multidynnos_py/       # Core implementation, CLI, packaging, and small input examples
├── raw/                 # Input datasets
├── samples/             # Published layout results and reproduction commands
├── viewer.html          # Interactive HTML snapshot viewer
└── view_snapshots.ipynb  # Visualization of saved results
```

Generated results are saved under `output/`, which is excluded from Git.
Local experiments, tests, evaluation code, and archived material are excluded
from the public repository.

## Installation

The pinned `requirements.txt` targets the `aspgd` environment, tested with Python
3.11.15. Use Python 3.11 to reproduce it. It includes the core implementation,
Numba acceleration, notebook tools, and dependencies used by local analysis and
tests. Initial XY coordinates use NetworkX `spring_layout` by default.
To select Graphviz initialization with `--graphviz`, install Graphviz separately
with `sfdp` available on PATH; run `sfdp -V` to check it. Java is not required.

Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m multidynnos_py --help
```

On Windows PowerShell, activate the virtual environment with `.venv/Scripts/Activate.ps1`.
To install only the core implementation (Python 3.10 or later), use
`python -m pip install -e ./multidynnos_py`.

The full requirements already include Numba. To add acceleration to a core-only installation:

```bash
python -m pip install -e './multidynnos_py[fast]'
```

## Usage

Run the small event example:

```bash
python -m multidynnos_py multidynnos_py/examples/events.json --repulsion-backend numpy -o output/events.json
```

Run the Newcomb dataset with the default layout or an aspect ratio adjustment:

```bash
python -m multidynnos_py raw/newcomb --repulsion-backend numpy -o output/newcomb.json
python -m multidynnos_py raw/newcomb --repulsion-backend numpy --aspect-ratio 3 -o output/newcomb_ar3.json
```

Also adjust the initial coordinates to width:height = 1:3, using either initializer:

```bash
python -m multidynnos_py raw/newcomb --aspect-ratio 3 --initial-ratio -o output/newcomb_spring_initial_ar3.json
python -m multidynnos_py raw/newcomb --graphviz --aspect-ratio 3 --initial-ratio -o output/newcomb_graphviz_initial_ar3.json
```

To download Newcomb instead of using local input, replace `raw/newcomb` with
`--dataset newcomb`. Published samples and the commands used to generate them
are described in [samples/README.md](samples/README.md).

| Option | Description |
| --- | --- |
| `--algorithm multi`, `single`, `static` | Multilevel MultiDynNoS, single-level DynNoSlice, or static initial layout; default: `multi`. `sfdp` is a legacy alias for `static`. |
| `--graphviz` | Use Graphviz for initial XY coordinates; otherwise use NetworkX `spring_layout`, including with the `sfdp` alias |
| `--graphviz-engine sfdp`, `fdp` | Engine used with `--graphviz`; default: `sfdp` |
| `--aspect-ratio N` | Target width:height = 1:N; N is height/width |
| `--aspect-ratio-fit exact`, `none` | Whether to fit the final global XY bounding box of all trajectories; default: `exact` |
| `--initial-ratio` | Also adjust initial XY coordinates using the N from `--aspect-ratio N`; accepts no value of its own |
| `--tau T` | Set the time-axis scale explicitly; uses AutoTau when omitted |
| `--manual-tau` | Use the dataset's recommended time-axis scale; 5 for Newcomb |
| `--iterations N` | Initial iteration count; defaults: 75 for multi, 100 for single |
| `--seed N` | Initialization seed; default: 0 |
| `--repulsion-backend auto`, `numpy`, `numba` | Repulsion implementation; default: `auto` |
| `--threads N` | Number of threads for Numba repulsion |

Omitting the aspect ratio option disables aspect ratio adjustment for the chosen
initializer. To retain the previous Graphviz initialization, pass `--graphviz`.
`--aspect-ratio 1` applies a square adjustment, so it differs from omitting the
option. `exact` fits the global bounding box of trajectories across all times;
it does not guarantee the same ratio for each individual snapshot. The ARCOL
correction is added to the XY forces, and the final bounding box fit is applied
after optimization.

`--initial-ratio` applies before dynamic refinement: on the coarsest graph for
`multi`, or on the input graph for `single` and `static`. For Spring coordinates,
let W and H be the node-center bounding-box dimensions. If H/W is below N, only
Y is expanded by N/(H/W); if it is above N, only X is expanded by (H/W)/N.
Scaling is centered on the bounding-box center, and neither axis is shrunk.
A zero-width or zero-height box is left unchanged because scaling cannot give
it a finite positive ratio. This transformation changes only XY coordinates.

This follows Graphviz's [numeric `ratio` axis-expansion rule](https://graphviz.org/docs/attrs/ratio/).
With `--graphviz`, the existing `-Gratio=N` request is passed to Graphviz itself;
node sizes and drawing extents mean its node-center ratio may be approximate.
Spring uses node centers directly, so the two initializers do not produce the
same coordinates or coordinate scale. NetworkX's other `spring_layout` defaults
are retained, including their version-dependent choice of layout method;
`--seed` sets its random seed (reduced modulo 2^32 for NetworkX). Spring uses an
undirected graph with edge-presence durations as weights; parallel and opposite
edges contribute summed weights. Its default coordinate scale is 1, rather than
Graphviz's inches. Dynamic ARCOL forces and final fitting are independent of
this initializer choice. The `single` algorithm now also computes initial XY
with the selected backend, replacing supplied coordinates rather than retaining
them or scattering missing positions as before.

## Input and output

Input examples are available in [events.json](multidynnos_py/examples/events.json)
and [snapshots.json](multidynnos_py/examples/snapshots.json).

- Event JSON: node and edge lifetimes expressed as `start/end`, `start/duration`, `intervals`, or `presence`.
- Timesliced JSON: graphs at each time in `snapshots` or `timeslices`.
- CSV: `source,target,start,duration` for edges and `id,start,duration` in an optional `--nodes` file.
- Three-column `source,target,timestamp` CSV: requires `--event-duration` in the same time units as the input.
- Newcomb: a directory or ZIP containing `newfratNN.csv` rank matrices.

See [raw/README.md](raw/README.md) for the datasets in `raw/`.
Output JSON contains `graphnodes`, `graphedges`, and `metadata`. Each node's
`position` stores time intervals and their endpoint XY coordinates, with linear
interpolation within each interval. Times are saved in the original input units.
Use `--format interval-map` to store intervals as JSON keys, or `--no-metadata`
to omit metadata.

Python API:

```python
from multidynnos_py.io import load_graph, save_graph
from multidynnos_py.multilevel import layout

graph = load_graph("multidynnos_py/examples/events.json")
result = layout(graph, seed=0, aspect_ratio=3, repulsion_backend="numpy")
save_graph(result, "output/events_ar3.json")
```

Pass `graphviz=True` to `layout` for Graphviz initialization and
`initial_ratio=True` to also apply the requested `aspect_ratio` during
initialization. The low-level `static_layout` function remains a Graphviz utility.

## Sample visualization

Open `view_snapshots.ipynb` in your notebook editor and select the Python
environment where the requirements are installed. The requirements include
IPython and the kernel; JupyterLab is not installed in the reference `aspgd`
environment. To use the JupyterLab interface, install it separately:

```bash
python -m pip install jupyterlab
jupyter lab view_snapshots.ipynb
```

Set the three values in the first notebook cell, then run all cells:

```python
JSON_PATH = "samples/newcomb/layout.json"
N_COLUMNS = 24
VISIBLE_COLUMNS = 4
```

`N_COLUMNS` divides the full time range into that many snapshots.
`VISIBLE_COLUMNS` sets how many snapshots appear side by side. The notebook
embeds `viewer.html` with the selected dataset. Use the top slider to move
through the horizontal snapshot strip, the mouse wheel to zoom, and drag to pan.
Each panel includes all nodes and edges present during its time window, with
node coordinates averaged over their presence duration. All panels share the
same coordinate range and scale; the stored layout is not recomputed.
Uncheck **Isolated nodes** to hide nodes with no incident edge in each snapshot's
time window. Counts and **Fit visible** follow this filter. **Edges** independently
controls whether edge lines are drawn.
An offline HTML copy is saved to `output/snapshot_previews/<input_stem>_viewer.html`.

## Upstream and license

This implementation is based on [EngAAlex/MultiDynNos](https://github.com/EngAAlex/MultiDynNos)
at commit `068aa79680b7d670d2338493bc9c88f4ffbd3db6`. It ports the core algorithm
to Python and includes corrected AutoTau calculations, ARCOL aspect ratio forces,
and optional Numba acceleration. It does not include all upstream parsers for
specific datasets or the upstream GUI.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the code license and attribution.
