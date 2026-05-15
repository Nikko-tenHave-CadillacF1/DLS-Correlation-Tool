"""Plot-entry helpers: plot constructors, plotter builder, job runner, and PowerPoint export."""

from __future__ import annotations

import argparse
import os
import traceback
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from typing import Optional, Union

from dataplotter import DataPlotter


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
    """Bundle all parameters needed for a plotting job in one place."""

    title: str
    root_folder: Path
    output_dir: Path
    runs: list
    plot_definitions: tuple

    channel_mappings: Optional[dict] = None
    channel_transforms: Optional[dict] = None
    calculated_channels: Optional[dict] = None
    filters: Optional[dict] = None
    units_map: Optional[dict] = None
    fig_size: Optional[Union[list, dict]] = None
    scatter_max_points: int = 45000
    bar_secondary_axis_ratio: float = 20.0
    box_plot_settings: Optional[dict] = None
    verbose: bool = False

    # PowerPoint
    powerpoint_template: Optional[Path] = None
    powerpoint_output: Optional[Path] = None
    export_map: Optional[Union[dict, list]] = None
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
    """Convert 'type/Plot Name' to 'type_plot_name.png'."""
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
    return f"{prefix.lower()}_{safe}.png"


def _resolve_export_map(export_map, plot_definitions, start_slide=1):
    """Resolve export_map: if it's a list of Slide dicts, convert to numbered dict.

    Accepts:
      - dict (legacy format): {slide_num: {"layout": ..., "images": [...]}}
      - list (new format): [Slide(...), Slide(...), ...] — auto-numbered starting at ``start_slide``
    Returns a dict in legacy format.
    """
    if export_map is None:
        return None
    if isinstance(export_map, list):
        offset = max(1, int(start_slide))
        return {i + offset: slide for i, slide in enumerate(export_map)}
    return export_map


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
                issues.append(f"Run '{label}': file not found → {file_path}")
        if not run.get("color"):
            issues.append(f"Run '{label}': missing 'color' key.")
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
    type_prefixes = ["waveform", "scatter", "psd", "histogram", "bar", "box"]
    generated_names: set[str] = set()

    for group_idx, group in enumerate(plot_definitions):
        if not group:
            continue
        prefix = type_prefixes[group_idx] if group_idx < len(type_prefixes) else "plot"
        for plot_def in group:
            if not plot_def or len(plot_def) < 1:
                continue
            plot_name = plot_def[0]
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
    powerpoint_template=None,
    powerpoint_output=None,
    export_map=None,
    fig_size=None,
    cli_description: Optional[str] = None,
    **overrides,
):
    """One-call entry point: build plot groups, parse CLI, run the job.

    workflow: 'correlation' | 'boxplots' | 'dampers'.
    Pass plot lists by category (waveforms=, scatters=, ...); empty categories may be omitted.
    Any PlotJobConfig field can be overridden via **overrides.
    """
    plot_definitions = build_plot_groups(
        waveforms=waveforms, scatters=scatters, psds=psds,
        histograms=histograms, bars=bars, boxes=boxes,
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

    cli_args: optional argparse.Namespace with .only / .types / .no_open / .runs
              / .dry_run / .list_plots / .check_only overrides from parse_plot_cli().
    """
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
            print(f"[ERROR] No runs matched: {cli_args.runs}")
            print(f"  Available: {[r['name'] for r in config.runs]}")
            raise SystemExit(1)

    # --- Pre-flight validation ---
    issues = validate_config(config)
    if issues:
        print("\n[ERROR] Configuration validation failed:")
        for issue in issues:
            print(f"  ✗ {issue}")
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

    plot_types = None
    plot_names = None
    open_output = config.open_output

    if cli_args is not None:
        if getattr(cli_args, "only", None):
            plot_names = cli_args.only
        if getattr(cli_args, "types", None):
            plot_types = cli_args.types
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
    )

    # --- Handle --check-only (data quality report only) ---
    if cli_args is not None and getattr(cli_args, "check_only", False):
        from dataplotter import build_quality_sections, write_data_quality_report
        sections = build_quality_sections(runs, plotter.run_data, config.plot_definitions)
        report_path = write_data_quality_report(plotter.plots_dir, sections)
        _print_quality_summary(sections)
        print(f"\nFull report: {report_path}")
        return

    run_plot_job(
        title=config.title,
        plotter=plotter,
        plot_types=plot_types,
        plot_names=plot_names,
        powerpoint_template=config.powerpoint_template,
        powerpoint_output=config.powerpoint_output,
        export_map=resolved_export_map,
        open_output=open_output,
    )


def parse_plot_cli(description: str = "Run plotting job"):
    """CLI parser for Run_*.py entry points with filtering and diagnostic modes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Generate only plots whose name matches (case-insensitive).",
    )
    parser.add_argument(
        "--types", nargs="+", metavar="TYPE",
        help="Generate only these plot types (waveform, scatter, psd, histogram, bar, box).",
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
        "--dry-run", action="store_true", default=False,
        help="Preview what would be generated without running the pipeline.",
    )
    parser.add_argument(
        "--list-plots", action="store_true", default=False,
        help="Print all configured plot names and exit.",
    )
    parser.add_argument(
        "--check-only", action="store_true", default=False,
        help="Load data, run quality checks, and exit without generating plots.",
    )
    return parser.parse_args()


def _print_plot_list(plot_definitions):
    """Print all configured plot names grouped by type."""
    type_names = ["waveform", "scatter", "psd", "histogram", "bar", "box"]
    print("\nConfigured Plots:")
    print("-" * 50)
    total = 0
    for i, group in enumerate(plot_definitions or []):
        if not group:
            continue
        label = type_names[i] if i < len(type_names) else f"group_{i}"
        print(f"\n  {label.upper()} ({len(group)}):")
        for plot_def in group:
            print(f"    • {plot_def[0]}")
            total += 1
    print(f"\n  Total: {total} plot(s)")


def _print_dry_run(config, runs, export_map):
    """Print a summary of what would be generated."""
    type_names = ["waveform", "scatter", "psd", "histogram", "bar", "box"]
    print("\n" + "=" * 60)
    print(f"{'DRY RUN':^60}")
    print("=" * 60)

    print(f"\n  Title:   {config.title}")
    print(f"  Output:  {config.output_dir}")
    print(f"\n  Runs ({len(runs)}):")
    for run in runs:
        print(f"    • {run['name']} ({run.get('type', '?')}) — {run.get('file', '?')}")

    total = 0
    print(f"\n  Plots:")
    for i, group in enumerate(config.plot_definitions or []):
        if not group:
            continue
        label = type_names[i] if i < len(type_names) else f"group_{i}"
        print(f"    {label}: {len(group)}")
        total += len(group)
    print(f"    ─────────")
    print(f"    total: {total}")

    if config.powerpoint_template:
        print(f"\n  PowerPoint: {config.powerpoint_output}")
        if export_map:
            print(f"    Slides mapped: {len(export_map)}")

    print("\n" + "=" * 60 + "\n")


def _print_quality_summary(sections):
    """Print a brief summary of data quality findings."""
    print("\n  Data Quality Summary:")
    print("  " + "-" * 40)
    for title, values in sections:
        status = "✓" if not values else f"⚠ {len(values)}"
        print(f"    {status}  {title}")
    print()


def build_plot_groups(
    *,
    waveforms=None,
    scatters=None,
    psds=None,
    histograms=None,
    bars=None,
    boxes=None,
):
    """Build the 6-slot plot-definition tuple expected by DataPlotter.

    All arguments are keyword-only; omitted slots default to [].
    """
    return tuple(group or [] for group in (waveforms, scatters, psds, histograms, bars, boxes))


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
        "dampers":     ("DAMPER_INPUT_DIR",      "DAMPER_PLOTS_DIR",
                        "DAMPER_CALCULATED",     "DAMPER_FILTERS"),
    }

    if workflow not in _WORKFLOW_MAP:
        raise ValueError(f"Unknown workflow '{workflow}'. Expected: {list(_WORKFLOW_MAP)}")

    import channel_config as _cc
    input_dir, output_dir, calc_attr, filt_attr = _WORKFLOW_MAP[workflow]

    return PlotJobConfig(
        title=title,
        root_folder=getattr(_cc, input_dir),
        output_dir=getattr(_cc, output_dir),
        runs=runs,
        plot_definitions=plot_definitions,
        channel_mappings=overrides.pop("channel_mappings", CHANNEL_MAPPINGS),
        channel_transforms=overrides.pop("channel_transforms", CHANNEL_TRANSFORMS),
        calculated_channels=overrides.pop("calculated_channels", getattr(_cc, calc_attr)),
        filters=overrides.pop("filters", getattr(_cc, filt_attr)),
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
    from dataplotter import build_quality_sections, write_data_quality_report
    sections = build_quality_sections(plotter.runs, plotter.run_data, plotter.PLOT_DEFINITIONS)
    report_path = write_data_quality_report(plotter.plots_dir, sections)
    has_issues = any(values for _, values in sections)
    if has_issues:
        _print_quality_summary(sections)
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
    print(f"\nGenerated {plot_count} plot(s) in {elapsed:.1f}s  \u2192  {plotter.plots_dir}")

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
                print(f"[WARNING] Could not auto-open PowerPoint file: {open_err}")
                print(f"File saved to: {powerpoint_output}")
        except Exception as export_err:
            print(f"[ERROR] PowerPoint export failed: {export_err}")
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

# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

def WaveformPlot(
    name: str,
    channels: tuple,
    axis_limits: Optional[tuple] = None,
    reference_lines: Optional[tuple] = None,
    subplot_heights: Optional[tuple] = None,
    x_limits: Optional[tuple] = None,
    x_channel: str = "sLap",
    highlight_zones=None,
    normalise: bool = False,
    legend_position: str = "top",
    show_delta: bool = False,
) -> list:
    """Define a waveform subplot figure.

    channels: one entry per row — 'ch' or ('left', 'right').
    axis_limits: per-row (ymin, ymax) or ((y1_min, y1_max), (y2_min, y2_max)).
    reference_lines: per-row scalar / tuple-of-scalars / None.
    subplot_heights: relative row heights; default equal.
    x_channel: 'sLap' (default) or 'tLap' etc.
    highlight_zones: ('ch', 'op', val[, '#color']) — shade matching x-regions.
    normalise: rescale all channels to [0, 1].
    legend_position: 'top' (default) or 'right' — places run legend above or to the side.
    show_delta: if True and exactly 2 runs are loaded, append a thin difference
        row (run_B − run_A) below each primary row. Default False.
    """
    _require_str(name, "name")
    _require_nonempty(channels, "channels")
    if legend_position not in ("top", "right"):
        raise ValueError("legend_position must be 'top' or 'right'.")
    return [name, channels, axis_limits, reference_lines, subplot_heights, x_limits, x_channel, highlight_zones, normalise, legend_position, bool(show_delta)]


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

def ScatterPlot(
    name: str,
    x_channel: str,
    y_channel: str,
    axis_limits: Optional[list] = None,
    best_fit: Union[int, list, None] = 0,
    gate: Union[tuple, list, None] = None,
    show_equations: bool = True,
    show_error: bool = True,
    color_gate=None,
    annotate_fit_at=None,
) -> list:
    """Define a scatter (XY correlation) plot.

    best_fit: 0/None=no fit, 1=single, list=segmented [('ch', lo, hi), ...].
    gate: ('ch', 'op', val) or list thereof — pre-filter data.
    color_gate: ('ch', 'op', val, '#hex') — colour matching points differently.
    annotate_fit_at: x-value to mark on fit line with vertical dashed line.
    """
    _require_str(name, "name")
    _require_str(x_channel, "x_channel")
    _require_str(y_channel, "y_channel")
    return [name, (x_channel, y_channel), axis_limits, best_fit, gate, show_equations, show_error, color_gate, annotate_fit_at]


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def PsdPlot(
    name: str,
    channel: Union[str, list],
    axis_limits: Optional[list] = None,
    log_scale: bool = True,
    nperseg: Optional[int] = None,
    annotate_at=None,
) -> list:
    """Define a PSD plot.

    channel: str or list[str] for multi-channel overlay.
    axis_limits: [(f_min, f_max), (power_min, power_max)].
    nperseg: Welch window size; None for pipeline default.
    annotate_at: frequency or tuple of frequencies to annotate PSD values at.
    """
    _require_str(name, "name")
    if isinstance(channel, (list, tuple)):
        if not channel:
            raise ValueError("'channel' list must not be empty.")
    else:
        _require_str(channel, "channel")
    return [name, channel, axis_limits, log_scale, int(nperseg) if nperseg is not None else None, annotate_at]


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def HistogramPlot(
    name: str,
    channel: str,
    axis_limits: Optional[list] = None,
    log_scale: bool = False,
) -> list:
    """Define a histogram plot. axis_limits: [(x_min, x_max), (y_min, y_max)]."""
    _require_str(name, "name")
    _require_str(channel, "channel")
    return [name, channel, axis_limits, log_scale]


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------

def BarPlot(
    name: str,
    metrics: tuple,
    default_aggregation: str = "last",
    axis_limits: Optional[tuple] = None,
    target_line=None,
) -> list:
    """Define a bar chart.

    metrics: ('ch',) or (('ch', 'agg'),). Aggregations: integral/sum/last/mean/max/min.
    target_line: optional horizontal reference value.
    """
    _require_str(name, "name")
    _require_nonempty(metrics, "metrics")
    if default_aggregation not in {"integral", "sum", "last", "mean", "max", "min"}:
        raise ValueError(
            f"BarPlot '{name}': default_aggregation must be one of "
            "'integral', 'sum', 'last', 'mean', 'max', 'min'. Got: {default_aggregation!r}"
        )
    return [name, metrics, default_aggregation, axis_limits, target_line]


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def BoxPlot(
    name: str,
    channels: Union[str, list],
    aggregation_mode: str = "per_run",
    axis_limits: Optional[tuple] = None,
    gate: Union[tuple, list, None] = None,
    options: Optional[dict] = None,
) -> list:
    """Define a box plot.

    aggregation_mode: 'per_run' (one box per run) or 'aggregated' (all merged).
    gate: same format as ScatterPlot.gate.
    options: override visual settings (show_points, box_width, jitter, etc.).
    """
    _require_str(name, "name")
    if aggregation_mode not in {"per_run", "aggregated"}:
        raise ValueError(
            f"BoxPlot '{name}': aggregation_mode must be 'per_run' or 'aggregated'. "
            f"Got: {aggregation_mode!r}"
        )
    return [name, channels, aggregation_mode, axis_limits, gate, options]


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _require_str(value, param_name: str):
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"'{param_name}' must be a non-empty string. Got: {value!r}")


def _require_nonempty(value, param_name: str):
    if not value:
        raise ValueError(f"'{param_name}' must not be empty. Got: {value!r}")


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
        print(f"[WARNING][powerpointexporter] PowerPoint template not found: {template_path}. Using default aspect ratios.")
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
        print(f"[WARNING][powerpointexporter] Error reading template aspect ratios: {e}. Using default aspect ratios.")
        return {}

    return aspect_ratios


# ================================================================
# MAIN EXPORT FUNCTION
# ================================================================

def export_report_to_powerpoint(template_path, output_path, plots_dir, export_map, visible=False):
    """
    Insert generated plots into a PowerPoint template according to export_map.
    """
    try:
        import win32com.client
    except ImportError as exc:
        raise ImportError(
            "pywin32 is required for PowerPoint export. Install with:\n"
            "    pip install pywin32"
        ) from exc

    template_path = Path(template_path).resolve()
    plots_dir = Path(plots_dir).resolve()
    output_path = Path(output_path).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"PowerPoint template not found: {template_path}")

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

            if layout == "main_plot" and len(image_list) == 1:
                target_boxes = [_get_main_plot_box(picture_boxes, slide_width, slide_height)]
            elif layout == "double_plot" and len(image_list) == 2:
                target_boxes = _get_double_plot_boxes(picture_boxes, slide_width, slide_height)
            else:
                target_boxes = picture_boxes or [
                    _resolve_box(layout, slide_width, slide_height, slot_index=i, slot_count=len(image_list))
                    for i in range(len(image_list))
                ]

            _replace_slide_pictures(slide)

            for i, img in enumerate(image_list):
                img_path = plots_dir / img
                if not img_path.exists():
                    print(f"[WARNING][powerpointexporter] Missing plot for slide {slide_num}: {img}")
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

                # Aggressive padding for scatter/PSD in double-layout
                if (
                    layout == "double_plot"
                    and img.startswith(("scatter_", "psd_", "histogram_", "bar_"))
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
        print(f"[ERROR][powerpointexporter] PowerPoint export failed: {exc}")

    finally:
        try:
            ppt.Quit()
        except Exception as quit_err:
            print(f"[WARNING][powerpointexporter] Error quitting PowerPoint: {quit_err}")

        # Release COM objects
        pres = None
        ppt = None
