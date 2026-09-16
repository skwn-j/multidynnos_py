# MultiDynNoS Python core

The core temporal graph layout implementation of MultiDynNoS.
It supports event and time-slice inputs, multilevel coarsening and refinement,
DynNoSlice optimization, ARCOL aspect ratio adjustment, and JSON output.

Install and run from the repository root:

```bash
python -m pip install -e ./multidynnos_py
python -m multidynnos_py multidynnos_py/examples/events.json -o output/events.json
```

Requires Python 3.10 or later, NumPy, NetworkX, and SciPy.
Initial XY coordinates use NetworkX `spring_layout` by default. Pass `--graphviz`
to use a separately installed Graphviz `sfdp` (or `--graphviz-engine fdp`).
Pass `--aspect-ratio 3 --initial-ratio` to adjust initial width:height to 1:3.
For Spring, this expands the insufficient axis about the node-center bounding-box
center; zero-width or zero-height boxes are left unchanged. Graphviz receives
`-Gratio=3`, so its node-center ratio may differ because of drawing extents.
Dynamic ARCOL adjustment and final fitting follow initialization as before.
Install optional Numba acceleration with `python -m pip install -e './multidynnos_py[fast]'`.
See the [project README](../README.md) for full usage instructions, published samples, and visualization guidance.

| File | Purpose |
| --- | --- |
| `model.py` | Data models for time intervals, nodes, edges, and trajectories |
| `io.py`, `datasets.py` | Input parsing, Newcomb loading, and JSON output |
| `multilevel.py` | Multilevel coarsening, initial placement, and refinement |
| `dynamics.py` | DynNoSlice forces and trajectory optimization |
| `aspect_ratio.py` | ARCOL aspect ratio correction |
| `acceleration.py` | Optional Numba acceleration for repulsive forces |
| `cli.py`, `__main__.py` | Command-line execution |
| `viewer.py` | Snapshot aggregation and notebook integration with `viewer.html` |

Licensed under Apache-2.0. See [NOTICE](../NOTICE) in the repository root for attribution to the original authors.
