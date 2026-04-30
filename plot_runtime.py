"""Plot-entry helpers: plot constructors, plotter builder, job runner, and PowerPoint export."""

from __future__ import annotations

import argparse
import os
import traceback
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from typing import Optional, Union

from dataplotter import DataPlotter


DEFAULT_FIG_SIZE = [(15.5, 6.4), (10, 8), (10, 8), (10, 8), (10, 6)]
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
    low_pass_filters: Optional[dict] = None
    units_map: Optional[dict] = None
    fig_size: Optional[list] = None
    scatter_max_points: int = 45000
    bar_secondary_axis_ratio: float = 20.0
    box_plot_settings: Optional[dict] = None
    verbose: bool = False

    # Plot method and filtering
    plot_method: str = "plot_data"
    generate_message: str = "Generating plots..."

    # PowerPoint
    powerpoint_template: Optional[Path] = None
    powerpoint_output: Optional[Path] = None
    export_map: Optional[dict] = None

    # Output behaviour
    open_output: bool = True


def run_from_config(config: PlotJobConfig, cli_args=None):
    """Build a plotter from a PlotJobConfig and run the job.

    cli_args: optional argparse.Namespace with .only / .types / .no_open
              overrides from parse_plot_cli().
    """
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

    plotter = build_plotter(
        root_folder=config.root_folder,
        runs=config.runs,
        plot_definitions=config.plot_definitions,
        channel_mappings=config.channel_mappings,
        channel_transforms=config.channel_transforms,
        calculated_channels=config.calculated_channels,
        low_pass_filters=config.low_pass_filters,
        units_map=config.units_map,
        fig_size=config.fig_size,
        scatter_max_points=config.scatter_max_points,
        bar_secondary_axis_ratio=config.bar_secondary_axis_ratio,
        box_plot_settings=config.box_plot_settings,
        output_dir=config.output_dir,
        verbose=config.verbose,
        template_path=config.powerpoint_template,
        export_map=config.export_map,
    )

    run_plot_job(
        title=config.title,
        plotter=plotter,
        plot_method=config.plot_method,
        generate_message=config.generate_message,
        plot_types=plot_types,
        plot_names=plot_names,
        powerpoint_template=config.powerpoint_template,
        powerpoint_output=config.powerpoint_output,
        export_map=config.export_map,
        open_output=open_output,
    )


def parse_plot_cli(description: str = "Run plotting job"):
    """Minimal CLI parser for Run_*.py entry points."""
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
        "--no-open", action="store_true", default=False,
        help="Do not auto-open the output folder after completion.",
    )
    return parser.parse_args()


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


def build_plotter(
    *,
    root_folder,
    runs,
    plot_definitions,
    channel_mappings=None,
    channel_transforms=None,
    calculated_channels=None,
    low_pass_filters=None,
    units_map=None,
    template_path=None,
    export_map=None,
    fig_size=None,
    scatter_max_points=45000,
    bar_secondary_axis_ratio=20.0,
    box_plot_settings=None,
    output_dir=None,
    verbose=False,
):
    """Build a DataPlotter with optional PowerPoint aspect-ratio hints."""
    plot_aspect_ratios = {}
    if template_path and export_map:
        cache_key = (
            str(template_path),
            json.dumps(export_map, sort_keys=True),
        )
        plot_aspect_ratios = _ASPECT_RATIO_CACHE.get(cache_key)
        if plot_aspect_ratios is None:
            plot_aspect_ratios = get_template_plot_aspect_ratios(template_path, export_map)
            _ASPECT_RATIO_CACHE[cache_key] = plot_aspect_ratios

    return DataPlotter(
        root_folder=root_folder,
        runs=runs,
        plot_definitions=plot_definitions,
        channel_mappings=channel_mappings,
        channel_transforms=channel_transforms,
        calculated_channels=calculated_channels,
        low_pass_filters=low_pass_filters,
        fig_size=fig_size or DEFAULT_FIG_SIZE,
        units_map=units_map,
        plot_aspect_ratios=plot_aspect_ratios,
        scatter_max_points=scatter_max_points,
        bar_secondary_axis_ratio=bar_secondary_axis_ratio,
        box_plot_settings=box_plot_settings,
        output_dir=output_dir,
        verbose=verbose,
    )


def run_plot_job(
    *,
    title,
    plotter,
    plot_method="plot_all",
    generate_message="Generating plots...",
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

    # Snapshot existing PNG modification times to count new/updated plots
    pre_mtimes = {
        p: p.stat().st_mtime for p in plotter.plots_dir.glob("*.png")
    }
    t0 = _time.perf_counter()

    print(f"\n{generate_message}")
    if (plot_types is not None or plot_names is not None) and plot_method in ("plot_all", "plot_data"):
        plotter.plot_data(plot_types=plot_types, plot_names=plot_names)
    else:
        getattr(plotter, plot_method)()

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
) -> list:
    """Define a waveform subplot figure.

    channels: one entry per row — 'ch' or ('left', 'right').
    axis_limits: per-row (ymin, ymax) or ((y1_min, y1_max), (y2_min, y2_max)).
    reference_lines: per-row scalar / tuple-of-scalars / None.
    subplot_heights: relative row heights; default equal.
    x_channel: 'sLap' (default) or 'tLap' etc.
    highlight_zones: ('ch', 'op', val[, '#color']) — shade matching x-regions.
    normalise: rescale all channels to [0, 1].
    """
    _require_str(name, "name")
    _require_nonempty(channels, "channels")
    return [name, channels, axis_limits, reference_lines, subplot_heights, x_limits, x_channel, highlight_zones, normalise]


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
    definition = [name, channel, axis_limits, log_scale]
    if nperseg is not None or annotate_at is not None:
        definition.append(int(nperseg) if nperseg is not None else 512)
    if annotate_at is not None:
        definition.append(annotate_at)
    return definition


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
    definition = [name, channels, aggregation_mode, axis_limits]
    if gate is not None:
        definition.append(gate)
    if options is not None:
        definition.append(options)
    return definition


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
