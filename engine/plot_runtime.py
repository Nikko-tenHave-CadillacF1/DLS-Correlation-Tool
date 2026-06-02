"""Plot-entry helpers: plot constructors, plotter builder, job runner, and PowerPoint export."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from typing import Optional, Union

# Ensure stdout/stderr can handle Unicode on Windows (cp1252 terminals).
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

from .dataplotter import DataPlotter
from .logger import log, configure as configure_logging


# Default figure sizes keyed by plot type for clarity.
DEFAULT_FIG_SIZE = {
    "waveform": (15.5, 6.4),
    "scatter": (10, 8),
    "psd": (10, 8),
    "histogram": (10, 8),
    "bar": (10, 6),
}
_ASPECT_RATIO_CACHE = {}


# ================================================================
# JOB CONFIGURATION
# ================================================================

@dataclass
class PlotJobConfig:
    """Bundle all parameters needed for a plotting job.

    Required fields: title, root_folder, output_dir, runs, plot_definitions.
    All other fields have sensible defaults and can be overridden per-job.

    Fields
    ------
    title : str
        Console banner title.
    root_folder : Path
        Root directory for input data files.
    output_dir : Path
        Directory where plots and reports are written.
    runs : list
        List of run definition dicts (name, file, color, type, nrun/nlap).
    plot_definitions : tuple
        7-slot tuple from ``build_plot_groups()`` — one list per plot type.
    channel_mappings : dict, optional
        Raw→canonical name mappings per source type.
    channel_transforms : dict, optional
        Sign flips and unit conversions per source type.
    calculated_channels : dict, optional
        Lambda-based derived channels evaluated after loading.
    filters : dict, optional
        Per-channel Butterworth filter configurations.
    resample_rate : float, optional
        Uniform resampling rate in Hz (default from channel_config).
    units_map : dict, optional
        Channel→unit label for axis annotations.
    fig_size : list or dict, optional
        Figure dimensions [width, height] or per-type dict.
    scatter_max_points : int
        Max points per scatter plot before random decimation (default 45000).
    bar_secondary_axis_ratio : float
        Scale factor triggering dual y-axis on bar charts (default 20.0).
    box_plot_settings : dict, optional
        Visual overrides for box plot styling.
    verbose : bool
        Enable debug-level logging (default False).
    powerpoint_template : Path, optional
        Path to .pptx template file.
    powerpoint_output : Path, optional
        Output path for generated PowerPoint.
    export_map : list, optional
        Slide-to-plot mapping — a list of ``Slide()`` dicts.
    powerpoint_start_slide : int
        1-based slide index where export_map entries begin (default 1).
    open_output : bool
        Auto-open output folder on completion (default True).
    output_dpi : int
        PNG resolution in dots per inch (default 300).
    """

    title: str
    root_folder: Path
    output_dir: Path
    runs: list
    plot_definitions: tuple

    channel_mappings: Optional[dict] = None
    channel_transforms: Optional[dict] = None
    calculated_channels: Optional[dict] = None
    filters: Optional[dict] = None
    resample_rate: Optional[float] = None
    units_map: Optional[dict] = None
    fig_size: Optional[Union[list, dict]] = None
    scatter_max_points: int = 45000
    bar_secondary_axis_ratio: float = 20.0
    box_plot_settings: Optional[dict] = None
    verbose: bool = False

    # PowerPoint
    powerpoint_template: Optional[Path] = None
    powerpoint_output: Optional[Path] = None
    export_map: Optional[list] = None
    # Slide number (1-based) where the first list-style export_map entry is placed.
    # Useful when the template has cover/intro slides that should be left untouched.
    powerpoint_start_slide: int = 1

    # Output behaviour
    open_output: bool = True
    output_dpi: int = 300


# ================================================================
# POWERPOINT SLIDE HELPER
# ================================================================

def Slide(layout: str, *plot_refs: str) -> dict:
    """Declarative slide definition for PowerPoint export.

    Usage:
        Slide("main_plot", "waveform/Driver Input")
        Slide("double_plot", "scatter/Gear Ratios", "scatter/Engine Power")

    plot_refs: 'type/Plot Name' — auto-converts to filename format.
    """
    images = [_plot_ref_to_filename(ref) for ref in plot_refs]
    return {"layout": layout, "images": images}


def _plot_ref_to_filename(ref: str) -> str:
    """Convert 'type/Plot Name' to 'type/type_plot_name.png' (per-type subfolder, #35)."""
    if "/" in ref:
        prefix, name = ref.split("/", 1)
    else:
        # Assume it's already a filename
        return ref if ref.endswith(".png") else f"{ref}.png"
    safe = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "").replace(")", "")
        .replace("/", "_").replace("\\", "_")
    )
    pref = prefix.lower()
    return f"{pref}/{pref}_{safe}.png"

def _resolve_export_map(export_map, plot_definitions, start_slide=1):
    """Number a list of Slide() dicts starting at ``start_slide``.

    ``export_map`` is a list of dicts produced by ``Slide()``; returns
    ``{slide_number: {"layout": ..., "images": [...]}}``.
    """
    if export_map is None:
        return None
    offset = max(1, int(start_slide))
    return {i + offset: slide for i, slide in enumerate(export_map)}


# ================================================================
# PRE-FLIGHT VALIDATION
# ================================================================

_VALID_RUN_TYPES = {"OC", "CAR", "DLS", "DIL"}


def validate_config(config: PlotJobConfig) -> list[str]:
    """Pre-flight validation of a PlotJobConfig. Returns list of error strings (empty = OK)."""
    issues: list[str] = []

    # --- Validate RUNS ---
    if not config.runs:
        issues.append("RUNS list is empty — nothing to plot.")
    for i, run in enumerate(config.runs):
        label = run.get("name", f"<unnamed run[{i}]>")
        if not run.get("name"):
            issues.append(f"Run[{i}]: missing 'name' key.")
        if not run.get("file"):
            issues.append(f"Run '{label}': missing 'file' key.")
        else:
            file_path = config.root_folder / run["file"]
            if not file_path.exists():
                issues.append(f"Run '{label}': file not found -> {file_path}")
        if not run.get("color"):
            # Color is now optional — DataPlotter auto-assigns from a palette (#15).
            pass
        run_type = run.get("type")
        if run_type and run_type not in _VALID_RUN_TYPES:
            issues.append(
                f"Run '{label}': unknown type '{run_type}'. "
                f"Expected one of: {', '.join(sorted(_VALID_RUN_TYPES))}"
            )

    # --- Validate PowerPoint template ---
    if config.powerpoint_template and not config.powerpoint_template.exists():
        issues.append(f"PowerPoint template not found: {config.powerpoint_template}")

    return issues


def validate_export_map(plot_definitions: tuple, export_map: Optional[dict]) -> list[str]:
    """Check for generated plots not referenced in the export map. Returns warnings."""
    if not export_map or not plot_definitions:
        return []

    # Collect all filenames that will be generated
    from .plot_definitions import PLOT_TYPE_ORDER as type_prefixes
    generated_names: set[str] = set()

    for group_idx, group in enumerate(plot_definitions):
        if not group:
            continue
        prefix = type_prefixes[group_idx] if group_idx < len(type_prefixes) else "plot"
        for plot_def in group:
            plot_name = getattr(plot_def, "name", None)
            if not plot_name:
                continue
            safe = (
                plot_name.lower()
                .replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_").replace("\\", "_")
            )
            generated_names.add(f"{prefix}_{safe}.png")

    # Collect all filenames referenced in the export map
    mapped_names: set[str] = set()
    for slide_config in export_map.values():
        for img in slide_config.get("images", []):
            mapped_names.add(img)

    orphans = generated_names - mapped_names
    warnings = []
    if orphans:
        warnings.append(
            f"[WARNING] {len(orphans)} plot(s) generated but NOT in POWERPOINT_EXPORT_MAP:"
        )
        for img in sorted(orphans):
            warnings.append(f"  • {img}")
    return warnings


def run_workflow(
    workflow: str,
    *,
    title: str,
    runs: list,
    waveforms=None,
    scatters=None,
    psds=None,
    histograms=None,
    bars=None,
    boxes=None,
    heatmaps=None,
    powerpoint_template=None,
    powerpoint_output=None,
    export_map=None,
    fig_size=None,
    cli_description: Optional[str] = None,
    **overrides,
):
    """One-call entry point: build plot groups, parse CLI, run the job.

    Parameters
    ----------
    workflow : str
        Workflow identifier — 'correlation', 'boxplots', 'dampers', 'ride_dil',
        or any custom name. Determines default channel configs and directories.
    title : str
        Display title for the console banner.
    runs : list
        Run definitions (dicts with 'name', 'file', 'color', 'type', etc.).
    waveforms, scatters, psds, histograms, bars, boxes, heatmaps :
        Plot definition lists for each category. Omit or pass None for empty.
    powerpoint_template : Path, optional
        Path to .pptx template for slide export.
    powerpoint_output : Path, optional
        Output path for the generated PowerPoint file.
    export_map : list of Slide dicts, optional
        Mapping of slides to plot images.
    fig_size : list or dict, optional
        Override figure dimensions (width, height) or per-type dict.
    cli_description : str, optional
        Description shown in --help output.
    **overrides :
        Any PlotJobConfig field: root_folder, output_dir, verbose, output_dpi,
        scatter_max_points, open_output, powerpoint_start_slide,
        calculated_channels, filters, resample_rate, etc.
    """
    plot_definitions = build_plot_groups(
        waveforms=waveforms, scatters=scatters, psds=psds,
        histograms=histograms, bars=bars, boxes=boxes, heatmaps=heatmaps,
    )
    config = workflow_config(
        workflow,
        title=title,
        runs=runs,
        plot_definitions=plot_definitions,
        powerpoint_template=powerpoint_template,
        powerpoint_output=powerpoint_output,
        export_map=export_map,
        fig_size=fig_size,
        **overrides,
    )
    return run_from_config(config, parse_plot_cli(cli_description or title))


def run_from_config(config: PlotJobConfig, cli_args=None):
    """Build a plotter from a PlotJobConfig and run the job.

    cli_args: optional argparse.Namespace with .only / .no_open / .runs
              / .dry_run / .list_plots / .list_channels / .check_only
              overrides from parse_plot_cli().
    """
    # --- Configure logging verbosity ---
    configure_logging(verbose=config.verbose)

    # --- Resolve export map (list → dict) ---
    resolved_export_map = _resolve_export_map(
        config.export_map, config.plot_definitions, start_slide=config.powerpoint_start_slide,
    )

    # --- Handle --list-plots (early exit) ---
    if cli_args is not None and getattr(cli_args, "list_plots", False):
        _print_plot_list(config.plot_definitions)
        return

    # --- Handle --runs filter ---
    runs = config.runs
    if cli_args is not None and getattr(cli_args, "runs", None):
        requested = {r.lower() for r in cli_args.runs}
        runs = [r for r in runs if r["name"].lower() in requested]
        if not runs:
            log.error("No runs matched: %s", cli_args.runs)
            print(f"  Available: {[r['name'] for r in config.runs]}")
            raise SystemExit(1)

    # --- Pre-flight validation ---
    issues = validate_config(config)
    if issues:
        print("\n[ERROR] Configuration validation failed:")
        for issue in issues:
            print(f"  X  {issue}")
        raise SystemExit(1)

    if resolved_export_map:
        orphan_warnings = validate_export_map(config.plot_definitions, resolved_export_map)
        if orphan_warnings:
            for line in orphan_warnings:
                print(line)
            print()

    # --- Handle --dry-run (early exit after validation) ---
    if cli_args is not None and getattr(cli_args, "dry_run", False):
        _print_dry_run(config, runs, resolved_export_map)
        return

    plot_names = None
    open_output = config.open_output

    if cli_args is not None:
        if getattr(cli_args, "only", None):
            plot_names = cli_args.only
        if getattr(cli_args, "no_open", False):
            open_output = False

    # --- Resolve fig_size ---
    fig_size = config.fig_size

    # --- Resolve PPT aspect ratios ---
    plot_aspect_ratios = {}
    if config.powerpoint_template and resolved_export_map:
        cache_key = (
            str(config.powerpoint_template),
            json.dumps(resolved_export_map, sort_keys=True),
        )
        plot_aspect_ratios = _ASPECT_RATIO_CACHE.get(cache_key)
        if plot_aspect_ratios is None:
            plot_aspect_ratios = get_template_plot_aspect_ratios(config.powerpoint_template, resolved_export_map)
            _ASPECT_RATIO_CACHE[cache_key] = plot_aspect_ratios

    plotter = DataPlotter(
        root_folder=config.root_folder,
        runs=runs,
        plot_definitions=config.plot_definitions,
        channel_mappings=config.channel_mappings,
        channel_transforms=config.channel_transforms,
        calculated_channels=config.calculated_channels,
        filters=config.filters,
        fig_size=fig_size or DEFAULT_FIG_SIZE,
        units_map=config.units_map,
        plot_aspect_ratios=plot_aspect_ratios,
        scatter_max_points=config.scatter_max_points,
        bar_secondary_axis_ratio=config.bar_secondary_axis_ratio,
        box_plot_settings=config.box_plot_settings,
        output_dir=config.output_dir,
        verbose=config.verbose,
        output_dpi=config.output_dpi,
        resample_rate=config.resample_rate,
    )

    # --- Handle --list-channels (after load, before plotting) ---
    if cli_args is not None and getattr(cli_args, "list_channels", False):
        _print_run_channels(plotter.run_data)
        return

    # --- Fail-fast on channel typos: referenced but absent from EVERY run ---
    _enforce_channel_typo_check(config.plot_definitions, plotter.run_data)

    # --- Handle --check-only (data quality report only) ---
    if cli_args is not None and getattr(cli_args, "check_only", False):
        from .data_quality_report import (
            build_quality_sections, write_data_quality_report, print_quality_summary,
        )
        sections = build_quality_sections(
            runs, plotter.run_data, config.plot_definitions,
            run_sample_rates=plotter.run_sample_rates,
            outlier_log=plotter._outlier_log,
        )
        report_path = write_data_quality_report(plotter.plots_dir, sections)
        print_quality_summary(sections)
        print(f"\nFull report: {report_path}")
        return

    run_plot_job(
        title=config.title,
        plotter=plotter,
        plot_types=None,
        plot_names=plot_names,
        powerpoint_template=config.powerpoint_template,
        powerpoint_output=config.powerpoint_output,
        export_map=resolved_export_map,
        open_output=open_output,
    )


def parse_plot_cli(description: str = "Run plotting job"):
    """CLI parser for Run_*.py entry points.

    Minimal, plug-and-play surface. Five flags total:

      --only NAME [NAME ...]   Run only plots whose name matches (case-insensitive).
      --runs RUN [RUN ...]     Process only these runs by name (case-insensitive).
      --no-open                Don't auto-open the output folder.
      --list-plots             Print configured plots and exit.
      --list-channels          Print channels available in each loaded run and exit.
      --check-only             Run data-quality checks and exit (no plots).
      --dry-run                Preview plots without loading data.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Generate only plots whose name matches (case-insensitive).",
    )
    parser.add_argument(
        "--runs", nargs="+", metavar="RUN",
        help="Process only these runs by name (case-insensitive).",
    )
    parser.add_argument(
        "--no-open", action="store_true", default=False,
        help="Do not auto-open the output folder after completion.",
    )
    parser.add_argument(
        "--list-plots", action="store_true", default=False,
        help="Print all configured plot names and exit.",
    )
    parser.add_argument(
        "--list-channels", action="store_true", default=False,
        help="Load each run and print its available channel names, then exit.",
    )
    parser.add_argument(
        "--check-only", action="store_true", default=False,
        help="Load data, run quality checks, and exit without generating plots.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview what would be generated without running the pipeline.",
    )
    return parser.parse_args()


def _print_run_channels(run_data):
    """Print available channels in each loaded run.

    Channels common to every run are listed first under "Common"; per-run
    extras follow. Helps newcomers discover what they can plot.
    """
    if not run_data:
        print("\n(No runs loaded.)\n")
        return
    per_run = {name: set(df.columns) for name, df in run_data.items()}
    common = set.intersection(*per_run.values()) if per_run else set()
    print("\nAvailable channels")
    print("-" * 50)
    print(f"\n  COMMON to all {len(per_run)} run(s) — {len(common)} channel(s):")
    for ch in sorted(common):
        print(f"    {ch}")
    for name, cols in per_run.items():
        extras = sorted(cols - common)
        if extras:
            print(f"\n  ONLY in '{name}' — {len(extras)} channel(s):")
            for ch in extras:
                print(f"    {ch}")
    print()


def _enforce_channel_typo_check(plot_definitions, run_data):
    """Fail fast if any referenced channel is absent from every loaded run.

    Channels missing from some-but-not-all runs are tolerated (handled by the
    data quality report and per-plot skipping). Channels missing from *all*
    runs are almost always typos, so we raise SystemExit with a suggestion
    drawn from the actual run columns.
    """
    if not run_data:
        return
    from .dataplotter import collect_referenced_channels
    import difflib

    referenced = set(collect_referenced_channels(plot_definitions))
    union = set()
    for df in run_data.values():
        union.update(df.columns)
    bogus = sorted(ch for ch in referenced if ch not in union)
    if not bogus:
        return
    lower_to_actual = {c.lower(): c for c in union}
    print("\n[ERROR] Plot definitions reference channels that exist in no loaded run:")
    for ch in bogus:
        cands = difflib.get_close_matches(ch, union, n=3, cutoff=0.6)
        if not cands:
            cands_lower = difflib.get_close_matches(ch.lower(), lower_to_actual.keys(), n=3, cutoff=0.55)
            cands = [lower_to_actual[c] for c in cands_lower]
        hint = f"  did you mean: {', '.join(cands)}?" if cands else ""
        print(f"  X  '{ch}'{hint}")
    print("\nRun with --list-channels to see all available channel names.\n")
    raise SystemExit(1)


def _print_plot_list(plot_definitions):
    """Print all configured plot names grouped by type."""
    from .plot_definitions import PLOT_TYPE_ORDER as type_names
    print("\nConfigured Plots:")
    print("-" * 50)
    total = 0
    for i, group in enumerate(plot_definitions or []):
        if not group:
            continue
        label = type_names[i] if i < len(type_names) else f"group_{i}"
        print(f"\n  {label.upper()} ({len(group)}):")
        for plot_def in group:
            name = getattr(plot_def, "name", plot_def)
            try:
                refs = _plot_referenced_channels(plot_def)
            except Exception:
                refs = []
            if refs:
                # Cap channel list at 8 for readability.
                shown = ", ".join(refs[:8])
                more = f" (+{len(refs) - 8} more)" if len(refs) > 8 else ""
                print(f"    *  {name}  [{shown}{more}]")
            else:
                print(f"    *  {name}")
            total += 1
    print(f"\n  Total: {total} plot(s)")


def _print_dry_run(config, runs, export_map):
    """Print a detailed summary of what would be generated (#12).

    For each configured plot, indicates which referenced channels are present
    in each run (peeking at the file schema where possible) and estimates the
    on-disk size of the resulting PNG.
    """
    from .plot_definitions import PLOT_TYPE_ORDER as type_names
    print("\n" + "=" * 60)
    print(f"{'DRY RUN':^60}")
    print("=" * 60)

    print(f"\n  Title:   {config.title}")
    print(f"  Output:  {config.output_dir}")
    print(f"\n  Runs ({len(runs)}):")
    for run in runs:
        print(f"    *  {run['name']} ({run.get('type', '?')}) -- {run.get('file', '?')}")

    # Peek the file schemas to flag missing channels per run.
    available_by_run = {}
    for run in runs:
        file_path = config.root_folder / run.get("file", "")
        cols = set()
        try:
            if file_path.suffix.lower() in (".parquet", ".pq"):
                # Try a column-name-only read.
                try:
                    import pyarrow.parquet as pq
                    cols = set(pq.read_schema(str(file_path)).names)
                except Exception:
                    try:
                        from fastparquet import ParquetFile  # type: ignore[import-not-found]
                        cols = set(ParquetFile(str(file_path)).columns)
                    except Exception:
                        cols = set()
            elif file_path.suffix.lower() in (".txt", ".csv"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    # 3-row DLS header: metadata, headers (row 2), units.
                    fh.readline()
                    header_line = fh.readline()
                    if header_line:
                        cols = {c.strip() for c in header_line.replace("\t", ",").split(",") if c.strip()}
        except Exception:
            cols = set()
        available_by_run[run["name"]] = cols

    total = 0
    print("\n  Plots:")
    est_bytes = 0
    figsize_default = (10, 8)
    dpi = config.output_dpi if hasattr(config, "output_dpi") else 300
    for i, group in enumerate(config.plot_definitions or []):
        if not group:
            continue
        label = type_names[i] if i < len(type_names) else f"group_{i}"
        print(f"    {label}: {len(group)}")
        for plot_def in group:
            name = getattr(plot_def, "name", str(plot_def))
            # Per-plot referenced channels
            refs = _plot_referenced_channels(plot_def)
            missing_per_run = []
            for run in runs:
                avail = available_by_run.get(run["name"]) or set()
                if not avail:
                    continue
                missing = [c for c in refs if c not in avail]
                if missing:
                    missing_per_run.append(f"{run['name']}: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
            extra = f"  [missing in: {' | '.join(missing_per_run)}]" if missing_per_run else ""
            print(f"      *  {name}{extra}")
            total += 1
            # Crude PNG size estimate: w*h*dpi^2 px, ~3 bytes/px before compression
            # then divide by ~6 to account for PNG/zlib compression on telemetry plots.
            est_bytes += int(figsize_default[0] * figsize_default[1] * dpi * dpi * 3 / 6)
    print(f"    ---------")
    print(f"    total: {total}")
    if total:
        print(f"    estimated on-disk size: ~{est_bytes/1024/1024:.0f} MB")

    if config.powerpoint_template:
        print(f"\n  PowerPoint: {config.powerpoint_output}")
        if export_map:
            print(f"    Slides mapped: {len(export_map)}")

    print("\n" + "=" * 60 + "\n")


def _plot_referenced_channels(plot_def) -> list:
    """Best-effort list of channel names referenced by a single plot dataclass."""
    out: list = []
    kind = getattr(plot_def, "kind", None)
    if kind == "waveform":
        def _walk(item):
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                for v in item:
                    _walk(v)
        _walk(plot_def.channels)
        out.append(plot_def.x_channel)
    elif kind == "scatter":
        out.extend([plot_def.x_channel, plot_def.y_channel])
    elif kind in ("psd", "histogram"):
        ch = plot_def.channel
        if isinstance(ch, (list, tuple)):
            out.extend(ch)
        else:
            out.append(ch)
    elif kind == "bar":
        for m in plot_def.metrics or ():
            if isinstance(m, str):
                out.append(m)
            elif isinstance(m, (list, tuple)) and m:
                out.append(m[0])
    elif kind == "box":
        for ch in plot_def.channels:
            out.append(ch)
    elif kind == "heatmap":
        out.extend([plot_def.x_channel, plot_def.y_channel])
        if plot_def.z_channel:
            out.append(plot_def.z_channel)
    return sorted(set(c for c in out if c))


def build_plot_groups(
    *,
    waveforms=None,
    scatters=None,
    psds=None,
    histograms=None,
    bars=None,
    boxes=None,
    heatmaps=None,
):
    """Build the 7-slot plot-definition tuple expected by DataPlotter.

    Order matches ``plot_definitions.PLOT_TYPE_ORDER``.
    All arguments are keyword-only; omitted slots default to [].

    BoxPlotGrid instances in `boxes` with render_mode='expand' are expanded
    into individual BoxPlot objects. Grid-mode instances are passed through
    for the renderer to handle.
    """
    boxes = _expand_box_grids(boxes) if boxes else []
    return tuple(
        group or []
        for group in (waveforms, scatters, psds, histograms, bars, boxes, heatmaps)
    )


def _expand_box_grids(boxes):
    """Expand BoxPlotGrid(render_mode='expand') into individual BoxPlot instances."""
    from .plot_definitions import BoxPlotGrid

    expanded = []
    for item in boxes:
        if isinstance(item, BoxPlotGrid) and item.render_mode == "expand":
            expanded.extend(item.expand())
        else:
            expanded.append(item)
    return expanded


def workflow_config(
    workflow: str,
    *,
    title: str,
    runs: list,
    plot_definitions: tuple,
    powerpoint_template=None,
    powerpoint_output=None,
    export_map=None,
    fig_size=None,
    **overrides,
) -> PlotJobConfig:
    """Create a PlotJobConfig with workflow-specific defaults auto-resolved.

    workflow: 'correlation' | 'boxplots' | 'dampers' — auto-selects input/output dirs,
              calculated channels, and filters from channel_config.

    Any PlotJobConfig field can be overridden via **overrides.
    """
    from channel_config import (
        CHANNEL_MAPPINGS, UNITS_MAP, CHANNEL_TRANSFORMS,
        SCATTER_MAX_POINTS, BAR_SECONDARY_AXIS_RATIO, BOX_PLOT_SETTINGS,
    )
    _WORKFLOW_MAP = {
        "correlation": ("CORRELATION_INPUT_DIR", "CORRELATION_OUTPUT_DIR",
                        "CORRELATION_CALCULATED", "CORRELATION_FILTERS"),
        "boxplots":    ("BOXPLOT_INPUT_DIR",     "BOXPLOT_OUTPUT_DIR",
                        "BOXPLOT_CALCULATED",    "BOXPLOT_FILTERS"),
        "dampers":     ("DAMPER_INPUT_DIR",      "DAMPER_OUTPUT_DIR",
                        "DAMPER_CALCULATED",     "DAMPER_FILTERS"),
        "ride_dil":    ("RIDE_DIL_INPUT_DIR",    "RIDE_DIL_OUTPUT_DIR",
                        "RIDE_DIL_CALCULATED",   "RIDE_DIL_FILTERS"),
    }

    import channel_config as _cc

    # Allow explicit root_folder / output_dir overrides (e.g. event-scoped dirs)
    explicit_root = overrides.pop("root_folder", None)
    explicit_out = overrides.pop("output_dir", None)

    # Global resample rate (applies to all workflows; see channel_config).
    resample_rate = overrides.pop("resample_rate", getattr(_cc, "RESAMPLE_RATE", None))

    if workflow in _WORKFLOW_MAP:
        input_dir, output_dir, calc_attr, filt_attr = _WORKFLOW_MAP[workflow]
        root_folder = explicit_root or getattr(_cc, input_dir)
        out_folder = explicit_out or getattr(_cc, output_dir)
        calculated = overrides.pop("calculated_channels", getattr(_cc, calc_attr))
        filters = overrides.pop("filters", getattr(_cc, filt_attr))
    else:
        # Auto-create directories for custom workflows
        from channel_config import get_workflow_dirs
        _root, _out = get_workflow_dirs(workflow)
        root_folder = explicit_root or _root
        out_folder = explicit_out or _out
        calculated = overrides.pop("calculated_channels", getattr(_cc, "CALCULATED_CHANNELS", {}))
        filters = overrides.pop("filters", getattr(_cc, "DEFAULT_FILTERS", {}))

    return PlotJobConfig(
        title=title,
        root_folder=root_folder,
        output_dir=out_folder,
        runs=runs,
        plot_definitions=plot_definitions,
        channel_mappings=overrides.pop("channel_mappings", CHANNEL_MAPPINGS),
        channel_transforms=overrides.pop("channel_transforms", CHANNEL_TRANSFORMS),
        calculated_channels=calculated,
        filters=filters,
        resample_rate=resample_rate,
        units_map=overrides.pop("units_map", UNITS_MAP),
        scatter_max_points=overrides.pop("scatter_max_points", SCATTER_MAX_POINTS),
        bar_secondary_axis_ratio=overrides.pop("bar_secondary_axis_ratio", BAR_SECONDARY_AXIS_RATIO),
        box_plot_settings=overrides.pop("box_plot_settings", BOX_PLOT_SETTINGS),
        fig_size=fig_size,
        powerpoint_template=powerpoint_template,
        powerpoint_output=powerpoint_output,
        export_map=export_map,
        **overrides,
    )


def run_plot_job(
    *,
    title,
    plotter,
    plot_types=None,
    plot_names=None,
    powerpoint_template=None,
    powerpoint_output=None,
    export_map=None,
    open_output=True,
):
    """Run a plotting job with consistent console output and optional PPT export."""
    import time as _time

    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")

    # --- Data quality report (before plotting) ---
    from .data_quality_report import (
        build_quality_sections, write_data_quality_report, print_quality_summary,
    )
    sections = build_quality_sections(
        plotter.runs, plotter.run_data, plotter.PLOT_DEFINITIONS,
        run_sample_rates=getattr(plotter, "run_sample_rates", None),
        outlier_log=getattr(plotter, "_outlier_log", None),
    )
    report_path = write_data_quality_report(plotter.plots_dir, sections)
    has_issues = any(values for _, values in sections)
    if has_issues:
        print_quality_summary(sections)
        print(f"  Full report: {report_path}\n")

    # Snapshot existing PNG modification times to count new/updated plots
    pre_mtimes = {
        p: p.stat().st_mtime for p in plotter.plots_dir.glob("*.png")
    }
    t0 = _time.perf_counter()

    print("\nGenerating plots...")
    plotter.plot_data(plot_types=plot_types, plot_names=plot_names)

    elapsed = _time.perf_counter() - t0
    plot_count = sum(
        1 for p in plotter.plots_dir.glob("*.png")
        if p not in pre_mtimes or p.stat().st_mtime > pre_mtimes[p]
    )
    print(f"\nGenerated {plot_count} plot(s) in {elapsed:.1f}s -> {plotter.plots_dir}")

    if powerpoint_template and powerpoint_output and export_map:
        print("\nExporting to PowerPoint...")
        try:
            powerpoint_output.parent.mkdir(parents=True, exist_ok=True)
            export_report_to_powerpoint(
                template_path=powerpoint_template,
                output_path=powerpoint_output,
                plots_dir=plotter.plots_dir,
                export_map=export_map,
                visible=False,
            )
            try:
                os.startfile(powerpoint_output)
            except Exception as open_err:
                log.warning("Could not auto-open PowerPoint file: %s", open_err)
                print(f"File saved to: {powerpoint_output}")
        except Exception as export_err:
            log.error("PowerPoint export failed: %s", export_err)
            traceback.print_exc()

    # Auto-open the output folder on Windows
    if open_output:
        try:
            os.startfile(plotter.plots_dir)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print(f"{'PROCESSING COMPLETE':^80}")
    print("=" * 80 + "\n")


# ================================================================
# PLOT CONSTRUCTORS
# ================================================================
#
# The typed plot dataclasses now live in ``plot_definitions``. We re-export
# them here so existing config files (``from plot_runtime import WaveformPlot``)
# keep working unchanged. The dataclass ``__post_init__`` performs validation
# that previously lived in this file (#9, #23, #24).

from .plot_definitions import (  # noqa: E402
    Marker,
    WaveformPlot,
    ScatterPlot,
    PsdPlot,
    HistogramPlot,
    BarPlot,
    BoxPlot,
    BoxPlotGrid,
    HeatmapPlot,
)


# ================================================================
# POWERPOINT EXPORT
# ================================================================

MAIN_PLOT_BOX = {
    "left_ratio": 0.079,
    "top_ratio": 0.260,
    "width_ratio": 0.90,
    "height_ratio": 0.65,
}

DOUBLE_PLOT_LAYOUT = {
    "left_ratio": 0.0,
    "top_ratio": 0.245,
    "width_ratio": 1.2,
    "height_ratio": 0.9,
    "gap_ratio": 0.0,
}

# PowerPoint picture MsoShapeType values
MSO_PICTURE_TYPES = {11, 13}

PPTX_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


# ================================================================
# GEOMETRY HELPERS
# ================================================================

def _resolve_box(layout_name, slide_width, slide_height, slot_index=0, slot_count=1):
    """Compute a plot box for main/double layouts."""
    if layout_name == "main_plot":
        box = MAIN_PLOT_BOX
        return (
            slide_width * box["left_ratio"],
            slide_height * box["top_ratio"],
            slide_width * box["width_ratio"],
            slide_height * box["height_ratio"],
        )

    if layout_name == "double_plot":
        box = DOUBLE_PLOT_LAYOUT
        left = slide_width * box["left_ratio"]
        top = slide_height * box["top_ratio"]
        width = slide_width * box["width_ratio"]
        height = slide_height * box["height_ratio"]
        gap = slide_width * box["gap_ratio"]

        slot_width = (width - gap) / 2
        return (
            left + slot_index * (slot_width + gap),
            top,
            slot_width,
            height,
        )

    raise ValueError(f"Unsupported PowerPoint layout: {layout_name}")


# ================================================================
# SLIDE TEMPLATE PARSING
# ================================================================

def _replace_slide_pictures(slide):
    """Delete all picture shapes from a slide."""
    # reversed loop because PowerPoint collection mutates on delete
    for idx in range(slide.Shapes.Count, 0, -1):
        shape = slide.Shapes(idx)
        if shape.Type in MSO_PICTURE_TYPES:
            shape.Delete()


def _get_picture_boxes(slide):
    """Extract picture bounding boxes from a slide."""
    boxes = []
    for idx in range(1, slide.Shapes.Count + 1):
        sh = slide.Shapes(idx)
        if sh.Type in MSO_PICTURE_TYPES:
            boxes.append((sh.Left, sh.Top, sh.Width, sh.Height))

    return sorted(boxes, key=lambda b: (b[0], b[1]))


def _get_double_plot_boxes(picture_boxes, slide_width, slide_height):
    """
    For double-plot layouts:
    - If template already contains picture placeholders, use those.
    - Otherwise derive new ones using the DOUBLE_PLOT_LAYOUT ratios.
    """
    if len(picture_boxes) >= 2:
        # Use template-detected bounding boxes
        sorted_boxes = sorted(picture_boxes, key=lambda b: b[0])
        return sorted_boxes[:2]

    # Fall back to generic
    layout = DOUBLE_PLOT_LAYOUT
    left = slide_width * layout["left_ratio"]
    top = slide_height * layout["top_ratio"]
    total_width = slide_width * layout["width_ratio"]
    total_height = slide_height * layout["height_ratio"]
    gap = slide_width * layout["gap_ratio"]

    slot_width = max((total_width - gap) / 2, 0)

    return [
        (left, top, slot_width, total_height),
        (left + slot_width + gap, top, slot_width, total_height),
    ]


def _get_main_plot_box(picture_boxes, slide_width, slide_height):
    """
    For main-plot layouts:
    - If template has a placeholder, expand horizontally
    - Otherwise use layout constants
    """
    if picture_boxes:
        _, top, _, height = picture_boxes[0]
        return (0, top, slide_width, height)

    box = MAIN_PLOT_BOX
    return (
        slide_width * box["left_ratio"],
        slide_height * box["top_ratio"],
        slide_width * box["width_ratio"],
        slide_height * box["height_ratio"],
    )


# ================================================================
# IMAGE INSERTION
# ================================================================

def _add_picture_fit(slide, image_path, left, top, width, height, fill_factor=1.0):
    """
    Insert image into slide, preserving aspect ratio and centering it.
    fill_factor > 1 expands slightly to avoid white bands.
    """
    image_path = str(image_path)
    shape = slide.Shapes.AddPicture(image_path, False, True, 0, 0, -1, -1)

    shape.LockAspectRatio = True

    scale = min(width / shape.Width, height / shape.Height)
    scale *= fill_factor

    shape.Width *= scale
    shape.Height *= scale

    shape.Left = left + (width - shape.Width) / 2
    shape.Top = top + (height - shape.Height) / 2

    # Cosmetic border to separate plots visually
    shape.Line.Visible = True
    shape.Line.ForeColor.RGB = 0
    shape.Line.Weight = 1

    return shape


# ================================================================
# TEMPLATE ASPECT RATIO EXTRACTION
# ================================================================

def get_template_plot_aspect_ratios(template_path, export_map):
    """
    Reads the PPTX template and extracts the native aspect ratios of picture
    placeholders so exported plots match layout proportions precisely.
    Returns empty dict if template cannot be read (graceful degradation).
    """
    template_path = Path(template_path).resolve()
    if not template_path.exists():
        log.warning("PowerPoint template not found: %s. Using default aspect ratios.", template_path)
        return {}

    aspect_ratios = {}

    try:
        with ZipFile(template_path) as pptx:
            pres_root = ET.fromstring(pptx.read("ppt/presentation.xml"))
            slide_size = pres_root.find("p:sldSz", PPTX_NS)
            slide_width = int(slide_size.attrib.get("cx", 0)) if slide_size is not None else None

            for slide_num, config in export_map.items():
                slide_xml = f"ppt/slides/slide{slide_num}.xml"
                if slide_xml not in pptx.namelist():
                    continue

                root = ET.fromstring(pptx.read(slide_xml))

                # Collect all <p:pic> shapes
                picture_boxes = []
                for pic in root.findall(".//p:pic", PPTX_NS):
                    xfrm = pic.find("p:spPr/a:xfrm", PPTX_NS)
                    if xfrm is None:
                        continue

                    ext = xfrm.find("a:ext", PPTX_NS)
                    off = xfrm.find("a:off", PPTX_NS)
                    if ext is None:
                        continue

                    width = int(ext.attrib.get("cx", 0))
                    height = int(ext.attrib.get("cy", 0))
                    left = int(off.attrib.get("x", 0)) if off is not None else 0
                    top = int(off.attrib.get("y", 0)) if off is not None else 0

                    if width > 0 and height > 0:
                        picture_boxes.append((left, top, width, height))

                picture_boxes.sort(key=lambda b: (b[0], b[1]))

                image_files = config.get("images", [])
                slide_aspects = []

                for i, img_file in enumerate(image_files):
                    if i >= len(picture_boxes):
                        break

                    _, _, w, h = picture_boxes[i]
                    # For main plot with only one picture, stretch horizontally
                    if (
                        config.get("layout") == "main_plot"
                        and slide_width is not None
                        and len(image_files) == 1
                    ):
                        w = slide_width  # stretch to full width

                    slide_aspects.append((img_file, w / h))

                # For two-up scatter plots → average aspect ratio
                if (
                    config.get("layout") == "double_plot"
                    and len(slide_aspects) == 2
                    and not all(
                        name.startswith(("scatter_", "psd_", "bar_"))
                        for name, _ in slide_aspects
                    )
                ):
                    avg = sum(a for _, a in slide_aspects) / len(slide_aspects)
                    for img, _ in slide_aspects:
                        aspect_ratios[img] = (avg,)
                else:
                    for img, ar in slide_aspects:
                        aspect_ratios[img] = ar
    except Exception as e:
        log.warning("Error reading template aspect ratios: %s. Using default aspect ratios.", e)
        return {}

    return aspect_ratios


# ================================================================
# MAIN EXPORT FUNCTION
# ================================================================

def _export_via_pptx(template_path, output_path, plots_dir, export_map):
    """Cross-platform PowerPoint export using python-pptx (#42).

    Mirrors the layout logic of the COM-based path but works on macOS/Linux
    and on Windows machines without PowerPoint installed.
    """
    from pptx import Presentation
    from pptx.util import Emu
    from PIL import Image as _PILImage  # python-pptx already pulls Pillow in

    prs = Presentation(str(template_path))
    slide_width = int(prs.slide_width)   # EMU
    slide_height = int(prs.slide_height)

    def _iter_pic_elements(slide):
        """Yield every <p:pic> element on the slide, including those inside placeholders."""
        sp_tree = slide.shapes._spTree
        # Use lxml's findall on the spTree subtree.
        return list(sp_tree.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}pic")) + \
               list(sp_tree.iter("{http://schemas.openxmlformats.org/presentationml/2006/main}pic"))

    def _picture_boxes(slide):
        """Extract (left, top, width, height) in EMU for every picture on the slide."""
        boxes = []
        for pic in _iter_pic_elements(slide):
            xfrm = pic.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm")
            if xfrm is None:
                continue
            ext = xfrm.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ext")
            off = xfrm.find("{http://schemas.openxmlformats.org/drawingml/2006/main}off")
            if ext is None:
                continue
            try:
                w = int(ext.get("cx", 0))
                h = int(ext.get("cy", 0))
                left = int(off.get("x", 0)) if off is not None else 0
                top = int(off.get("y", 0)) if off is not None else 0
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                boxes.append((left, top, w, h))
        return sorted(boxes, key=lambda b: (b[0], b[1]))

    def _delete_pictures(slide):
        """Remove every <p:pic> element from the slide."""
        for pic in _iter_pic_elements(slide):
            parent = pic.getparent()
            if parent is not None:
                parent.remove(pic)

    def _add_picture_fit(slide, image_path, left, top, width, height, fill_factor=1.0):
        # Read native aspect via Pillow so we can preserve it ourselves.
        try:
            with _PILImage.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = (1, 1)
        if img_w <= 0 or img_h <= 0:
            img_w, img_h = (1, 1)

        # Both `width`/`height` are EMU; compute scale that fits within the box,
        # then center. fill_factor>1 expands slightly to remove whitespace bands.
        scale = min(width / img_w, height / img_h) * fill_factor
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        new_l = int(left + (width - new_w) / 2)
        new_t = int(top + (height - new_h) / 2)
        slide.shapes.add_picture(
            str(image_path),
            Emu(new_l), Emu(new_t),
            width=Emu(new_w), height=Emu(new_h),
        )

    def _is_misc_plot(img_file: str) -> bool:
        # Filenames are now `<type>/<type>_<name>.png` (per-type subfolders, #35).
        # Treat scatter/psd/histogram/bar as "misc" requiring extra padding.
        base = img_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return base.startswith(("scatter_", "psd_", "histogram_", "bar_"))

    for slide_num, config in export_map.items():
        idx = int(slide_num) - 1
        if idx < 0 or idx >= len(prs.slides):
            log.warning("Slide %s out of range (template has %d slides). Skipping.",
                        slide_num, len(prs.slides))
            continue
        slide = prs.slides[idx]
        layout = config.get("layout", "main_plot")
        image_files = config.get("images", [])
        if not image_files:
            continue

        existing = _picture_boxes(slide)

        # Resolve target boxes BEFORE deleting (COM path also reads first, deletes second).
        used_template_boxes = False
        if layout == "main_plot" and len(image_files) == 1:
            target_boxes = [_get_main_plot_box(existing, slide_width, slide_height)]
            used_template_boxes = bool(existing)
        elif layout == "double_plot" and len(image_files) == 2:
            target_boxes = _get_double_plot_boxes(existing, slide_width, slide_height)
            used_template_boxes = len(existing) >= 2
        else:
            # Fallback: reuse whatever placeholders exist, otherwise compute via ratios.
            target_boxes = existing or [
                _resolve_box(layout, slide_width, slide_height,
                             slot_index=i, slot_count=len(image_files))
                for i in range(len(image_files))
            ]
            used_template_boxes = bool(existing)

        _delete_pictures(slide)

        for i, img_file in enumerate(image_files):
            img_path = (Path(plots_dir) / img_file).resolve()
            if not img_path.exists():
                log.warning("Missing plot for slide %s: %s", slide_num, img_file)
                continue

            if i < len(target_boxes):
                left, top, w, h = target_boxes[i]
            else:
                left, top, w, h = _resolve_box(
                    layout, slide_width, slide_height,
                    slot_index=i, slot_count=len(image_files),
                )

            # Trust template placeholder dimensions exactly (designer-chosen).
            # Only expand when falling back to ratio-based DOUBLE_PLOT_LAYOUT
            # (which is intentionally oversized to remove whitespace bands).
            if used_template_boxes:
                fill_factor = 1.0
            elif layout == "double_plot" and _is_misc_plot(img_file):
                fill_factor = 1.2
            else:
                fill_factor = 1.0

            _add_picture_fit(slide, img_path, left, top, w, h, fill_factor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))


def export_report_to_powerpoint(template_path, output_path, plots_dir, export_map, visible=False):
    """
    Insert generated plots into a PowerPoint template according to export_map.

    Tries python-pptx first (cross-platform); falls back to win32com on
    Windows when python-pptx is unavailable.
    """
    template_path = Path(template_path).resolve()
    plots_dir = Path(plots_dir).resolve()
    output_path = Path(output_path).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"PowerPoint template not found: {template_path}")

    # Preferred: python-pptx (cross-platform, no PowerPoint app needed).
    try:
        import pptx  # noqa: F401
        import PIL  # noqa: F401
        _export_via_pptx(template_path, output_path, plots_dir, export_map)
        log.info("PowerPoint exported via python-pptx: %s", output_path)
        return
    except ImportError:
        log.debug("python-pptx not available; falling back to win32com.")
    except Exception as exc:
        log.warning(
            "python-pptx export failed (%s); falling back to win32com.", exc,
        )

    try:
        import win32com.client
    except ImportError as exc:
        raise ImportError(
            "Neither python-pptx nor pywin32 is available for PowerPoint export.\n"
            "Install one of:\n"
            "    pip install python-pptx   # cross-platform (preferred)\n"
            "    pip install pywin32       # Windows only"
        ) from exc

    # PowerPoint COM object
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True  # PowerPoint does not allow True/False control here

    pres = None
    try:
        pres = ppt.Presentations.Open(str(template_path), WithWindow=visible)

        slide_width = pres.PageSetup.SlideWidth
        slide_height = pres.PageSetup.SlideHeight

        for slide_num, cfg in export_map.items():
            slide = pres.Slides(slide_num)
            layout = cfg["layout"]
            image_list = cfg["images"]

            picture_boxes = _get_picture_boxes(slide)

            used_template_boxes = False
            if layout == "main_plot" and len(image_list) == 1:
                target_boxes = [_get_main_plot_box(picture_boxes, slide_width, slide_height)]
                used_template_boxes = bool(picture_boxes)
            elif layout == "double_plot" and len(image_list) == 2:
                target_boxes = _get_double_plot_boxes(picture_boxes, slide_width, slide_height)
                used_template_boxes = len(picture_boxes) >= 2
            else:
                target_boxes = picture_boxes or [
                    _resolve_box(layout, slide_width, slide_height, slot_index=i, slot_count=len(image_list))
                    for i in range(len(image_list))
                ]
                used_template_boxes = bool(picture_boxes)

            _replace_slide_pictures(slide)

            for i, img in enumerate(image_list):
                img_path = plots_dir / img
                if not img_path.exists():
                    log.warning("Missing plot for slide %d: %s", slide_num, img)
                    continue

                if i < len(target_boxes):
                    left, top, width, height = target_boxes[i]
                else:
                    left, top, width, height = _resolve_box(
                        layout,
                        slide_width,
                        slide_height,
                        slot_index=i,
                        slot_count=len(image_list),
                    )

                # Aggressive padding for scatter/PSD in double-layout — only when
                # falling back to ratio-based layout (DOUBLE_PLOT_LAYOUT is oversized
                # to remove whitespace bands). Trust template placeholder boxes 1:1.
                _basename = img.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                if (
                    not used_template_boxes
                    and layout == "double_plot"
                    and _basename.startswith(("scatter_", "psd_", "histogram_", "bar_"))
                ):
                    fill_factor = 1.2
                else:
                    fill_factor = 1.0

                _add_picture_fit(slide, img_path, left, top, width, height, fill_factor)

        # save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pres.SaveAs(str(output_path))
            final = output_path
        except Exception as exc:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
            print(
                f"[WARNING][powerpointexporter] Could not save to {output_path} ({exc}). Using fallback: {fallback}"
            )
            pres.SaveAs(str(fallback))
            final = fallback

        print(f"PowerPoint report saved to: {final}")

    except Exception as exc:
        log.error("PowerPoint export failed: %s", exc)

    finally:
        try:
            ppt.Quit()
        except Exception as quit_err:
            log.warning("Error quitting PowerPoint: %s", quit_err)

        # Release COM objects
        pres = None
        ppt = None
