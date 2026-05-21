"""Command-line helpers for the MultiDynNoS Python port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multidynnos_py.data.event_io import (
    EdgeEvent,
    build_graph_from_edge_events,
    export_binned_edge_event_snapshots,
    load_edge_events,
)
from multidynnos_py.data.graph import DynamicGraph
from multidynnos_py.data.io import export_graph_csv, export_graph_json, load_custom_graph
from multidynnos_py.layout.dynnoslice import DynamicLayout, DynNoSliceConfig, run_dynnoslice
from multidynnos_py.layout.multidynnos import MultiDynNoSConfig, run_multidynnos
from multidynnos_py.visualization.snapshot_rendering import render_layout_snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multidynnos-py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a custom MultiDynNoS graph.")
    parse_parser.add_argument("node_file", type=Path, help="CSV file with node_id,start_time,duration rows.")
    parse_parser.add_argument("edge_file", type=Path, help="CSV file with source_id,target_id,start_time,duration rows.")
    parse_parser.add_argument("--json", type=Path, help="Write the parsed graph as JSON.")
    parse_parser.add_argument("--nodes-csv", type=Path, help="Write parsed node appearances as CSV.")
    parse_parser.add_argument("--edges-csv", type=Path, help="Write parsed edge appearances as CSV.")
    parse_parser.add_argument("--include-edge-ids", action="store_true", help="Append deterministic edge IDs to CSV edge rows.")
    parse_parser.add_argument("--at", type=float, help="Print active node and edge counts at the supplied time.")
    parse_parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid non-blank input rows.")
    parse_parser.set_defaults(func=_run_parse)

    event_parser = subparsers.add_parser("parse-events", help="Parse an edge-event file with source,target,timestamp rows.")
    event_parser.add_argument("event_file", type=Path, help="Text or CSV file with source,target,timestamp rows.")
    event_parser.add_argument("--event-duration", type=float, default=0.0, help="Duration assigned to each timestamped event.")
    event_parser.add_argument("--json", type=Path, help="Write the parsed graph as JSON.")
    event_parser.add_argument("--at", type=float, help="Print active node and edge counts at the supplied time.")
    event_parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid non-blank input rows.")
    event_parser.add_argument("--has-header", action="store_true", help="Treat the first non-blank row as a header.")
    event_parser.add_argument("--time-bins", type=int, help="Aggregate events into this many temporal bins before graph construction.")
    event_parser.set_defaults(func=_run_parse_events)

    layout_parser = subparsers.add_parser("layout", help="Run a dynamic layout on a custom MultiDynNoS graph.")
    layout_parser.add_argument("node_file", type=Path, help="CSV file with node_id,start_time,duration rows.")
    layout_parser.add_argument("edge_file", type=Path, help="CSV file with source_id,target_id,start_time,duration rows.")
    layout_parser.add_argument("--method", choices=["single", "multi"], default="single", help="Layout method to run.")
    layout_parser.add_argument("--json", type=Path, help="Write the layout as JSON.")
    layout_parser.add_argument("--tau", type=float, default=1.0, help="Temporal smoothing weight.")
    layout_parser.add_argument("--delta", type=float, default=5.0, help="Preferred edge length.")
    layout_parser.add_argument("--iterations", type=int, default=100, help="DynNoSlice iterations per level.")
    layout_parser.add_argument("--learning-rate", type=float, default=0.05, help="Force integration learning rate.")
    layout_parser.add_argument("--seed", type=int, default=73, help="Deterministic layout seed.")
    layout_parser.add_argument("--max-levels", type=int, default=4, help="Maximum multilevel hierarchy depth.")
    layout_parser.add_argument("--min-coarse-nodes", type=int, default=2, help="Stop coarsening at or below this node count.")
    layout_parser.add_argument("--use-sfdp", action="store_true", help="Try GraphViz sfdp for static initialization.")
    layout_parser.add_argument("--postprocess-passes", type=int, default=1, help="Flexible trajectory smoothing passes.")
    layout_parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid non-blank input rows.")
    layout_parser.add_argument("--visualize", type=int, help="After layout, render this many temporal snapshot figures.")
    layout_parser.add_argument("--show-labels", action="store_true", help="Show node labels in rendered snapshots.")
    layout_parser.set_defaults(func=_run_layout)

    event_layout_parser = subparsers.add_parser("layout-events", help="Run a dynamic layout on an edge-event file.")
    event_layout_parser.add_argument("event_file", type=Path, help="Text or CSV file with source,target,timestamp rows.")
    event_layout_parser.add_argument("--event-duration", type=float, default=0.0, help="Duration assigned to each timestamped event.")
    event_layout_parser.add_argument("--method", choices=["single", "multi"], default="single", help="Layout method to run.")
    event_layout_parser.add_argument("--json", type=Path, help="Write the layout as JSON.")
    event_layout_parser.add_argument("--tau", type=float, default=1.0, help="Temporal smoothing weight.")
    event_layout_parser.add_argument("--delta", type=float, default=5.0, help="Preferred edge length.")
    event_layout_parser.add_argument("--iterations", type=int, default=100, help="DynNoSlice iterations per level.")
    event_layout_parser.add_argument("--learning-rate", type=float, default=0.05, help="Force integration learning rate.")
    event_layout_parser.add_argument("--seed", type=int, default=73, help="Deterministic layout seed.")
    event_layout_parser.add_argument("--max-levels", type=int, default=4, help="Maximum multilevel hierarchy depth.")
    event_layout_parser.add_argument("--min-coarse-nodes", type=int, default=2, help="Stop coarsening at or below this node count.")
    event_layout_parser.add_argument("--use-sfdp", action="store_true", help="Try GraphViz sfdp for static initialization.")
    event_layout_parser.add_argument("--postprocess-passes", type=int, default=1, help="Flexible trajectory smoothing passes.")
    event_layout_parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid non-blank input rows.")
    event_layout_parser.add_argument("--has-header", action="store_true", help="Treat the first non-blank row as a header.")
    event_layout_parser.add_argument("--time-bins", type=int, help="Aggregate events into this many temporal bins before layout.")
    event_layout_parser.add_argument("--visualize", type=int, help="After layout, render this many temporal snapshot figures.")
    event_layout_parser.add_argument("--show-labels", action="store_true", help="Show node labels in rendered snapshots.")
    event_layout_parser.set_defaults(func=_run_layout_events)

    render_parser = subparsers.add_parser("render-snapshots", help="Render static snapshot figures from a layout JSON.")
    render_parser.add_argument("--layout", type=Path, required=True, help="Layout JSON produced by the layout pipeline.")
    render_parser.add_argument("--num-snapshots", type=int, required=True, help="Number of temporal snapshot windows to render.")
    render_parser.add_argument("--edges", type=Path, help="Optional edge CSV with source,target and temporal columns.")
    render_parser.add_argument("--dataset-name", help="Optional dataset-name override for output paths.")
    render_parser.add_argument("--output-root", type=Path, default=Path("output"), help="Root output directory.")
    render_parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png", help="Output image format.")
    render_parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    render_parser.add_argument("--show-labels", action="store_true", help="Draw small node labels.")
    render_parser.add_argument("--allow-nodes-only", action="store_true", help="Render nodes without edges when no edge data is available.")
    render_parser.add_argument(
        "--active-node-policy",
        choices=["in_window", "all", "midpoint_available"],
        default="in_window",
        help="Policy for deciding which nodes are visible.",
    )
    render_parser.add_argument(
        "--position-policy",
        choices=["midpoint_interpolate", "mean_in_window", "nearest_midpoint"],
        default="mean_in_window",
        help="Policy for selecting each node position inside a window.",
    )
    render_parser.add_argument("--fig-width", type=float, default=6.0, help="Figure width in inches.")
    render_parser.add_argument("--fig-height", type=float, default=6.0, help="Figure height in inches.")
    render_parser.add_argument(
        "--title-format",
        choices=["index_datetime", "datetime_range", "none"],
        default="index_datetime",
        help="Snapshot title format.",
    )
    render_parser.add_argument("--timezone", default="UTC", help="Timezone for ISO timestamps. Currently only UTC is supported.")
    render_parser.set_defaults(func=_run_render_snapshots)
    return parser


def _run_parse(args: argparse.Namespace) -> int:
    graph = load_custom_graph(args.node_file, args.edge_file, skip_invalid=args.skip_invalid)

    if args.json is not None:
        export_graph_json(graph, args.json)

    if args.nodes_csv is not None or args.edges_csv is not None:
        if args.nodes_csv is None or args.edges_csv is None:
            raise SystemExit("--nodes-csv and --edges-csv must be supplied together.")
        export_graph_csv(
            graph,
            args.nodes_csv,
            args.edges_csv,
            include_edge_ids=args.include_edge_ids,
        )

    print(f"Parsed {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    if args.at is not None:
        active_nodes = graph.active_nodes(args.at)
        active_edges = graph.active_edges(args.at)
        print(f"At t={args.at:g}: {len(active_nodes)} active nodes, {len(active_edges)} active edges.")
    return 0


def _run_parse_events(args: argparse.Namespace) -> int:
    events = _load_events(args)
    graph = build_graph_from_edge_events(events, event_duration=args.event_duration, time_bins=args.time_bins)
    _write_binned_snapshots_if_requested(args, events)

    if args.json is not None:
        export_graph_json(graph, args.json)

    print(f"Parsed {len(graph.nodes)} nodes and {len(graph.edges)} edges from edge events.")
    if args.at is not None:
        active_nodes = graph.active_nodes(args.at)
        active_edges = graph.active_edges(args.at)
        print(f"At t={args.at:g}: {len(active_nodes)} active nodes, {len(active_edges)} active edges.")
    return 0


def _run_layout(args: argparse.Namespace) -> int:
    graph = load_custom_graph(args.node_file, args.edge_file, skip_invalid=args.skip_invalid)
    layout = _layout_graph(args, graph)
    output_path = _write_layout(args, layout, graph)

    print(
        f"Ran {args.method} layout for {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} edges, and {len(layout.sample_times)} sampled times."
    )
    print(f"Wrote layout to {output_path}.")
    _render_visualization_if_requested(args, output_path, edges_path=args.edge_file)
    return 0


def _run_layout_events(args: argparse.Namespace) -> int:
    events = _load_events(args)
    graph = build_graph_from_edge_events(events, event_duration=args.event_duration, time_bins=args.time_bins)
    _write_binned_snapshots_if_requested(args, events)
    layout = _layout_graph(args, graph)
    output_path = _write_layout(args, layout, graph)

    print(
        f"Ran {args.method} layout for {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} edges, and {len(layout.sample_times)} sampled times from edge events."
    )
    print(f"Wrote layout to {output_path}.")
    _render_visualization_if_requested(args, output_path, edges_path=args.event_file)
    return 0


def _layout_graph(args: argparse.Namespace, graph: DynamicGraph) -> DynamicLayout:
    dyn_config = DynNoSliceConfig(
        tau=args.tau,
        delta=args.delta,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    if args.method == "single":
        layout = run_dynnoslice(graph, dyn_config)
    else:
        layout = run_multidynnos(
            graph,
            MultiDynNoSConfig(
                dynnoslice_config=dyn_config,
                max_levels=args.max_levels,
                min_coarse_nodes=args.min_coarse_nodes,
                use_sfdp=args.use_sfdp,
                postprocess_passes=args.postprocess_passes,
            ),
        )
    return layout


def _write_layout(args: argparse.Namespace, layout: DynamicLayout, graph: DynamicGraph) -> Path:
    output_path = args.json if args.json is not None else _default_layout_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = layout.to_dict()
    payload["edges"] = graph.to_dict()["edges"]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _default_layout_output_path(args: argparse.Namespace) -> Path:
    dataset_path = getattr(args, "event_file", None) or getattr(args, "node_file", None)
    dataset_name = dataset_path.stem if dataset_path is not None else "layout"
    return Path("output") / dataset_name / "layout.json"


def _load_events(args: argparse.Namespace) -> list[EdgeEvent]:
    return load_edge_events(
        args.event_file,
        skip_invalid=args.skip_invalid,
        has_header=True if args.has_header else None,
    )


def _write_binned_snapshots_if_requested(args: argparse.Namespace, events: list[EdgeEvent]) -> list[Path]:
    if args.time_bins is None:
        return []
    dataset_name = _dataset_name_from_input(args)
    output_dir = Path("output") / dataset_name / "snapshots"
    paths = export_binned_edge_event_snapshots(events, time_bins=args.time_bins, output_dir=output_dir)
    print(f"Wrote {len(paths)} binned graph CSV snapshots to {output_dir}.")
    return paths


def _render_visualization_if_requested(args: argparse.Namespace, layout_path: Path, *, edges_path: Path) -> None:
    if args.visualize is None:
        return
    if args.visualize <= 0:
        raise SystemExit("--visualize must be a positive integer.")

    dataset_name = _dataset_name_from_input(args)
    output_root = _output_root_for_layout_path(layout_path)
    results = render_layout_snapshots(
        layout_path=layout_path,
        n_snapshots=args.visualize,
        dataset_name=dataset_name,
        output_root=output_root,
        position_policy="mean_in_window",
        show_labels=args.show_labels,
        title_format="index_datetime",
    )
    figures_dir = Path(results[0].figure_path).parent if results else output_root / dataset_name / "figures"
    print(f"Rendered {len(results)} snapshots to {figures_dir}.")


def _dataset_name_from_input(args: argparse.Namespace) -> str:
    dataset_path = getattr(args, "event_file", None) or getattr(args, "node_file", None)
    return dataset_path.stem if dataset_path is not None else "layout"


def _output_root_for_layout_path(layout_path: Path) -> Path:
    if layout_path.name == "layout.json" and layout_path.parent.parent != layout_path.parent:
        return layout_path.parent.parent
    return Path("output")


def _run_render_snapshots(args: argparse.Namespace) -> int:
    results = render_layout_snapshots(
        layout_path=args.layout,
        n_snapshots=args.num_snapshots,
        edges_path=args.edges,
        dataset_name=args.dataset_name,
        output_root=args.output_root,
        image_format=args.format,
        allow_nodes_only=args.allow_nodes_only,
        show_labels=args.show_labels,
        active_node_policy=args.active_node_policy,
        position_policy=args.position_policy,
        dpi=args.dpi,
        figure_size=(args.fig_width, args.fig_height),
        title_format=args.title_format,
        timezone=args.timezone,
    )
    if results:
        figures_dir = Path(results[0].figure_path).parent
    else:
        figures_dir = args.output_root
    print(f"Rendered {len(results)} snapshots to {figures_dir}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
