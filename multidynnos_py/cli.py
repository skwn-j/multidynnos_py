"""Read temporal data, run MultiDynNoS, and write a space-time cube JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

from .datasets import load_newcomb
from .io import load_graph, save_graph
from .multilevel import _validate_initial_ratio, layout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Python MultiDynNoS: temporal graph → space-time cube JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("input", nargs="?", help="JSON/CSV path or URL; Newcomb matrix directory/zip")
    result.add_argument("--dataset", choices=["newcomb"], help="Load the original dataset; download if no input")
    result.add_argument("--nodes", help="Java custom node CSV (id,start,duration); input is edge CSV")
    result.add_argument("-o", "--output", required=True, help="Output JSON path")
    result.add_argument("--data-type", choices=["auto", "timesliced", "event"], default="auto")
    result.add_argument("--event-duration", type=float, help="Lifetime of timestamp-only events, in input time units")
    result.add_argument("--presence-mode", choices=["plain", "keepAppearedNode", "keepAppearedEdges"],
                        help="Default: keepAppearedNode for Newcomb, plain for generic inputs")
    result.add_argument("--algorithm", choices=["multi", "single", "static", "sfdp"], default="multi",
                        help="Layout algorithm; static (legacy alias: sfdp) only initializes XY")
    result.add_argument("--delta", type=float, default=5.0, help="Desired spatial distance")
    tau = result.add_mutually_exclusive_group()
    tau.add_argument("--tau", type=float, help="Time-to-space scale; default: paper AutoTau, equation (2)")
    tau.add_argument("--manual-tau", action="store_true", help="Use the dataset's suggested time factor")
    result.add_argument("--iterations", type=int, help="Initial iterations per level; multi=75, single=100")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--bend-transfer", action="store_true", help="Enable original optional bend transfer")
    result.add_argument("--graphviz", action="store_true",
                        help="Initialize XY with Graphviz; otherwise use NetworkX spring_layout")
    result.add_argument("--graphviz-engine", choices=["sfdp", "fdp"], default="sfdp",
                        help="Initializer engine when --graphviz is enabled")
    result.add_argument("--repulsion-backend", choices=["auto", "numpy", "numba"], default="auto",
                        help="Repulsion implementation; auto uses installed Numba for >=256 mirror segments")
    result.add_argument("--threads", type=int, help="Numba worker threads; default: Numba runtime setting")
    result.add_argument("--aspect-ratio", type=float, metavar="N",
                        help="ARCOL XY aspect ratio: width:height = 1:N; omit to disable ratio adjustment")
    result.add_argument("--initial-ratio", action="store_true",
                        help="Also use --aspect-ratio N during initialization: Graphviz ratio=N or expand the insufficient axis of Spring's node-center box")
    result.add_argument("--aspect-ratio-fit", choices=["exact", "none"], default="exact",
                        help="Final XY bounding-box fit after the ARCOL additive force; none uses only force refinement")
    result.add_argument("--format", choices=["timelighting", "interval-map"], default="timelighting")
    result.add_argument("--no-metadata", action="store_true", help="Write only graphnodes and graphedges")
    result.add_argument("--quiet", action="store_true", help="Suppress layout progress messages")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.input is None and args.dataset is None:
        parser().error("provide an input path/URL or --dataset newcomb")
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")
    started = time.monotonic()
    try:
        _validate_initial_ratio(args.initial_ratio, args.aspect_ratio, args.algorithm)
        newcomb = args.dataset == "newcomb" or (args.input is not None and
                  (Path(args.input).is_dir() or args.input.lower().endswith(".zip")))
        mode = args.presence_mode or ("keepAppearedNode" if newcomb else "plain")
        if args.dataset:
            if args.data_type == "event":
                raise ValueError("Newcomb matrices are timesliced data")
            graph = load_newcomb(args.input, presence_mode=mode)
        else:
            graph = load_graph(args.input, args.data_type, nodes_path=args.nodes,
                               event_duration=args.event_duration, presence_mode=mode)
        graph.validate()
        tau = args.tau
        if args.manual_tau:
            tau = graph.metadata.get("suggested_time_factor")
            if tau is None:
                raise ValueError("This dataset has no suggested time factor; provide --tau")
        logging.info("Input: %s, %d nodes, %d edges", graph.data_type, len(graph.nodes), len(graph.edges))
        iterations = args.iterations if args.iterations is not None else (100 if args.algorithm == "single" else 75)
        result = layout(graph, delta=args.delta, tau=tau, seed=args.seed,
                        bend_transfer=args.bend_transfer, iterations=iterations,
                        algorithm=args.algorithm, graphviz_engine=args.graphviz_engine,
                        repulsion_backend=args.repulsion_backend, num_threads=args.threads,
                        aspect_ratio=args.aspect_ratio, aspect_ratio_fit=args.aspect_ratio_fit,
                        initial_ratio=args.initial_ratio, graphviz=args.graphviz)
        path = save_graph(result, args.output, interval_maps=args.format == "interval-map",
                          include_metadata=not args.no_metadata)
        print(json.dumps({"output": str(path), "data_type": result.data_type,
                          "nodes": len(result.nodes), "edges": len(result.edges),
                          "seconds": round(time.monotonic() - started, 3)}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"multidynnos: {exc}", file=sys.stderr)
        return 2
