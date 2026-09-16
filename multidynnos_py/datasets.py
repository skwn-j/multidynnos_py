"""The original MultiDynNos Newcomb matrix parser, with a pinned download."""
from __future__ import annotations

import io
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

from .model import Edge, Interval, Node, TemporalGraph

UPSTREAM_COMMIT = "068aa79680b7d670d2338493bc9c88f4ffbd3db6"
NEWCOMB_URL = ("https://raw.githubusercontent.com/EngAAlex/MultiDynNos/"
               f"{UPSTREAM_COMMIT}/data/Newcomb/newfrat.zip")


def load_newcomb(source: str | Path | None = None, *,
                 presence_mode: str = "keepAppearedNode") -> TemporalGraph:
    """Load local ``newfratNN.csv`` matrices/a zip, or download upstream's zip.

    Port of NewcombFraternity.parseGraph: node IDs 0..16, an undirected edge for
    a top-three choice in either direction, and slice t mapped to (t-.5,t+.5].
    The upstream files are numbered 1..15; those exact indexes are preserved.
    """
    matrices = {}
    if source is not None and Path(str(source)).is_dir():
        files = [(path.name, path.read_text()) for path in Path(source).glob("newfrat*.csv")]
    else:
        if source is None or str(source).startswith(("https://", "http://")):
            with urlopen(NEWCOMB_URL if source is None else str(source), timeout=60) as response:
                archive = response.read()
        else:
            archive = Path(source).read_bytes()
        with ZipFile(io.BytesIO(archive)) as zipped:
            files = [(Path(name).name, zipped.read(name).decode("utf-8-sig"))
                     for name in zipped.namelist() if not name.endswith("/")]
    for name, content in files:
        match = re.fullmatch(r"newfrat(\d+)\.csv", name, re.IGNORECASE)
        if not match:
            continue
        t = int(match.group(1))
        if t in matrices:
            raise ValueError(f"Duplicate Newcomb slice: {t}")
        matrix = [[int(v) for v in line.split()] for line in content.splitlines() if line.strip()]
        if not matrix or any(len(row) != len(matrix) for row in matrix):
            raise ValueError(f"Newcomb slice {t} must be a square matrix")
        matrices[t] = matrix
    if not matrices or sorted(matrices) != list(range(1, max(matrices) + 1)):
        raise ValueError("Newcomb input requires consecutive newfrat01.csv ... matrices")
    count, last = len(matrices[1]), max(matrices)
    if any(len(matrix) != count for matrix in matrices.values()):
        raise ValueError("Newcomb matrix dimensions differ between slices")
    nodes = {str(i): Node(str(i), [Interval(0.0, float(last + 2))], [], f"{i:02d}")
             for i in range(count)}
    edges = {}
    for t, matrix in sorted(matrices.items()):
        for i in range(count):
            for j in range(i + 1, count):
                if 0 < matrix[i][j] <= 3 or 0 < matrix[j][i] <= 3:
                    key = i, j
                    if key not in edges:
                        edges[key] = Edge(f"{len(edges)}e", str(i), str(j), [])
                    edges[key].presence.append(Interval(t - 0.5, t + 0.5, False, True))
    graph = TemporalGraph(nodes, list(edges.values()), "timesliced", False, {
        "dataset": "newcomb", "source": str(source) if source is not None else NEWCOMB_URL,
        "upstream_commit": UPSTREAM_COMMIT, "snapshot_times": sorted(matrices),
        "suggested_time_factor": 5.0, "suggested_interval": [1.0, float(last)],
        "initial_scatter_distance": 40.0, "initial_scatter_seed": 73,
        "detection_reason": "numbered Newcomb adjacency/rank matrices",
        "newcomb_relationship": "rank 1..3 in either direction",
        "newcomb_time_axis": "upstream file indexes 1..15; no historical week renumbering",
    })
    from .io import apply_presence_mode
    apply_presence_mode(graph, presence_mode, float(last + 1.5))
    graph.validate()
    return graph
