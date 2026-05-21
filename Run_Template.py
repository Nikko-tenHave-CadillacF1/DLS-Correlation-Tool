"""
TEMPLATE — Reference configuration for the DLS Correlation Tool pipeline.

Copy this file as a starting point for new workflows. Each section below
demonstrates the available plot types, options, and run configurations.
Refer to this file for syntax, available parameters, and working examples.

Run types:
  DLS  — DLS/LTS lap simulation parquet files (use nlap or nrun to select)
  OC   — Optimum Capture parquet exports
  CAR  — Car telemetry .txt files (tab-separated)
  DIL  — Driver-in-the-loop simulator exports

Usage:
  python Run_Template.py               # generate all plots
  python Run_Template.py --dry-run     # preview without generating
  python Run_Template.py --list-plots  # list configured plot names
  python Run_Template.py --types scatter waveform  # generate only these types
  python Run_Template.py --only "Driver Input" "GG Plot"  # by name
"""

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot, HeatmapPlot,
    Marker,
)

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW NAME & EVENT
# ═══════════════════════════════════════════════════════════════════════════════
# Change these to create a new workflow. Directories are auto-created:
#   Data/inputs/<WORKFLOW_NAME>/              — without EVENT
#   Data/inputs/<WORKFLOW_NAME>/<EVENT>/      — with EVENT (recommended)
#   Data/outputs/<WORKFLOW_NAME>/<EVENT>/     — plots are saved here
#
# Set EVENT = None to use a flat workflow folder without event separation.
# For cross-event comparisons, set EVENT = None and prefix filenames with the
# event subfolder (see RUNS examples below).
WORKFLOW_NAME = "correlation"
EVENT = "26R04MIA"

_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ═══════════════════════════════════════════════════════════════════════════════
# RUNS
# ═══════════════════════════════════════════════════════════════════════════════
# Each run needs:
#   name:  display label used in plots and legends
#   file:  path relative to the workflow input folder (Data/inputs/<workflow>/)
#   color: hex colour for this run's traces
#   type:  OC | CAR | DLS | DIL — selects channel mappings and transforms
#
# Optional keys:
#   nrun:  (parquet only) rank-based run selection; nrun=1 → lowest nRun value
#   nlap:  exact lap number filter; ignored when nrun is also set

RUNS = [
    # ── DLS / LTS example ──────────────────────────────────────────────────────
    {
        "name": "LTS Baseline",
        "file": r"26R04MIA  PER Q1R3_LTS_Iteration_3.parquet",
        "color": "#0083BF",
        "nlap": 1,
        "type": "DLS",
    },

    # ── OC example ─────────────────────────────────────────────────────────────
    # {
    #     "name": "OC Reference",
    #     "file": r"my_oc_file.parquet",
    #     "color": "#51FF00",
    #     "nrun": 1,
    #     "type": "OC",
    # },

    # ── CAR example ────────────────────────────────────────────────────────────
    {
        "name": "Test Run 4",
        "file": "26R04MIA_260502_MAC26-01_PER_Q_R03_3.txt",
        "color": "#D70000",
        "type": "CAR",
    },

    # ── DIL example ────────────────────────────────────────────────────────────
    # {
    #     "name": "DIL Baseline",
    #     "file": r"my_dil_file.parquet",
    #     "color": "#FF8800",
    #     "nlap": 1,
    #     "type": "DIL",
    # },

    # ── Cross-event comparison ─────────────────────────────────────────────────
    # To compare runs from different events, set EVENT = None above and prefix
    # each filename with its event subfolder:
    #
    # EVENT = None
    # _INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)
    #
    # {
    #     "name": "MIA LTS",
    #     "file": "26R04MIA/26R04MIA  PER Q1R3_LTS_Iteration_3.parquet",
    #     "color": "#0083BF",
    #     "nlap": 1,
    #     "type": "DLS",
    # },
    # {
    #     "name": "SUZ LTS",
    #     "file": "26R03SUZ/26R03SUZ  77  Quali  Run 3 Q1R3  Stint 1 stint 3_-BSL_DLS.parquet",
    #     "color": "#D70000",
    #     "nlap": 1,
    #     "type": "DLS",
    # },
]

# ═══════════════════════════════════════════════════════════════════════════════
# POWERPOINT EXPORT (optional)
# ═══════════════════════════════════════════════════════════════════════════════
EXPORT_TO_POWERPOINT  = False
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = _OUTPUT_DIR / "Report.pptx"
# Slide number (1-based) where the first POWERPOINT_EXPORT_MAP entry is placed.
# Leaves cover / intro slides untouched.
POWERPOINT_START_SLIDE = 4

# ═══════════════════════════════════════════════════════════════════════════════
# WAVEFORM PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# channels:         one entry per subplot row — 'channel' or ('left_ch', 'right_ch')
# axis_limits:      per-row y-limits — (ymin, ymax) or ((y1_min, y1_max), (y2_min, y2_max))
# reference_lines:  per-row horizontal lines — scalar, tuple of scalars, or None
# subplot_heights:  relative row heights (e.g. 0.4 = half of 0.8)
# x_channel:        x-axis channel, default "sLap". Use "tLap" for time-based.
# x_limits:         (x_min, x_max) to zoom to a section of the lap
# normalise:        True to normalise all channels 0–1 (useful for overlays)
# highlight_zones:  ('channel', 'op', threshold, color) — shade regions
# markers:          list of Marker() objects — vertical reference lines

WAVEFORM_PLOT_DEFINITIONS = [
    # ── Basic waveform ─────────────────────────────────────────────────────────
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((60, 400), (-1, 9)), (-180, 180), None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    ),

    # ── Zoomed waveform with x_limits ──────────────────────────────────────────
    WaveformPlot(
        name="Zoomed Section",
        channels=('vCar', 'pBrakeF', 'rThrottle'),
        axis_limits=(None, None, (0, 105)),
        reference_lines=(None, None, None),
        subplot_heights=(0.6, 0.4, 0.4),
        x_limits=(1200, 1800),
    ),

    # ── Highlight zones — shade regions where a condition is true ──────────────
    WaveformPlot(
        name="Highlight Zones Demo",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, None, (20,)),
        subplot_heights=(0.6, 0.4, 0.4),
        highlight_zones=('rThrottle', '<', 20, '#FF4444'),
    ),

    # ── Normalised — all channels mapped 0..1 for shape comparison ─────────────
    WaveformPlot(
        name="Normalised Overlay",
        channels=('PMGUK', 'vCar', 'pBrakeF', 'rThrottle'),
        subplot_heights=(0.4, 0.4, 0.4, 0.4),
        normalise=True,
    ),

    # ── Markers on waveform ────────────────────────────────────────────────────
    # Static markers:    fixed x-position, drawn on every run.
    # Condition markers: resolved per run at rising/falling edges of a boolean.
    WaveformPlot(
        name="Waveform Markers Demo",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        subplot_heights=(0.6, 0.4, 0.4),
        markers=[
            Marker(x=1500, label="SM zone", color="#00B050", linestyle="--"),
            Marker(
                condition=[('pBrakeF', '>', 50), ('vCar', '>', 100)],
                edge="rising",
                label="hard brake",
                linestyle="-.",
                show_label=False,
            ),
        ],
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# SCATTER PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# best_fit: None/0 = no fit | 1 = single fit | list = segmented fits by condition
#   Segment format: ('channel', low, high) or ('x'/'y', low, high) for axis splits
# gate: filter data before plotting — ('channel', 'operator', value)
#   Operators: '>' '<' '>=' '<=' '==' 'between'  |  list for AND conditions
# show_equations: show fit equation text on the plot
# show_error:     show R² and RMSE for fit lines
# axis_limits:    [(x_min, x_max), (y_min, y_max)]
# color_gate:     ('channel', 'op', value, '#hexcolor') — colour subset differently
# annotate_fit_at: x-value or tuple of x-values where fit is annotated
# robust:         True to use Theil-Sen + MAD outlier rejection
# robust_threshold: number of MADs for outlier classification (default 3.0)
# markers:        list of Marker() for vertical reference lines

SCATTER_PLOT_DEFINITIONS = [
    # ── Basic scatter (no fit) ─────────────────────────────────────────────────
    ScatterPlot(
        name="GG Plot",
        x_channel="gLat",
        y_channel="gLong",
        best_fit=0,
    ),

    # ── Single best fit line ───────────────────────────────────────────────────
    ScatterPlot(
        name="Engine Efficiency",
        x_channel="dmInjector",
        y_channel="PEngine",
        best_fit=1,
        show_equations=True,
        show_error=True,
    ),

    # ── Segmented fits — split by channel value ────────────────────────────────
    ScatterPlot(
        name="Gear Ratios",
        x_channel="nWheelAvg_R",
        y_channel="nEngine",
        best_fit=[('NGear', 1.5, 2.5), ('NGear', 2.5, 3.5), ('NGear', 3.5, 4.5),
                  ('NGear', 4.5, 5.5), ('NGear', 5.5, 6.5), ('NGear', 6.5, 7.5), ('NGear', 7.5, 8.5)],
        show_equations=False,
    ),

    # ── Segmented fits — split by axis value ───────────────────────────────────
    ScatterPlot(
        name="Front Heave",
        x_channel="xDamperAvgF",
        y_channel="FPRodAvgF",
        best_fit=[('y', -6000, None), ('y', None, -6000)],
    ),

    # ── Gate — filter data before plotting ─────────────────────────────────────
    ScatterPlot(
        name="Braking Efficiency",
        x_channel="pBrakeF",
        y_channel="gLong",
        best_fit=[('y', None, -0.2)],
        gate=('gLong', '<', 0),
    ),

    # ── Multiple gate conditions (AND) ─────────────────────────────────────────
    ScatterPlot(
        name="Front Pushrod vCar",
        x_channel="vCar",
        y_channel="FPRodAvgF",
        best_fit=[('gLat_Abs', 0, 1)],
        gate=[('SM', '<', 1)],
    ),

    # ── Custom axis limits ─────────────────────────────────────────────────────
    ScatterPlot(
        name="Yaw Rate Response",
        x_channel="aSteerWheel",
        y_channel="nYaw",
        axis_limits=[(-160, 160), (None, None)],
        best_fit=[('x', -20, 20)],
    ),

    # ── Color gate — highlight a subset in a different colour ──────────────────
    ScatterPlot(
        name="Color Gate Demo",
        x_channel="vCar",
        y_channel="hRideF",
        best_fit=[('SM', 0, 0.5)],
        color_gate=('SM', '<', 0.3, '#FF00CC'),
    ),

    # ── Annotate fit at specific x-values ──────────────────────────────────────
    ScatterPlot(
        name="Annotate Fit At Demo",
        x_channel="vCar",
        y_channel="hRideR",
        best_fit=1,
        gate=[('SM', '<', 0.5)],
        annotate_fit_at=(100, 200, 300),
    ),

    # ── Robust fit — Theil-Sen + MAD outlier rejection ─────────────────────────
    ScatterPlot(
        name="Robust Fit Demo",
        x_channel="vCar",
        y_channel="PEngine",
        best_fit=1,
        robust=True,
        robust_threshold=3.0,
    ),

    # ── Vertical markers ──────────────────────────────────────────────────────
    ScatterPlot(
        name="Scatter Markers Demo",
        x_channel="vCar",
        y_channel="gLong",
        markers=[
            Marker(x=100, label="100 km/h"),
            Marker(x=300, label="300 km/h", color="#FF6600"),
        ],
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PSD PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# channel:      single channel string or list of channels (overlaid)
# axis_limits:  [(freq_min, freq_max), (psd_min, psd_max)]
# annotate_at:  tuple of frequencies where PSD values are annotated per run

PSD_PLOT_DEFINITIONS = [
    # ── Single channel PSD ─────────────────────────────────────────────────────
    PsdPlot(
        name="Front Vertical Acceleration PSD",
        channel="gVertF",
        axis_limits=[(0, 20), (1e-4, None)],
        annotate_at=(5, 15),
    ),

    # ── Multi-channel PSD — overlaid on same axes ──────────────────────────────
    # Line style cycles (solid → dashed) per channel; legend shows "RUN — channel".
    PsdPlot(
        name="Multi-Channel PSD Demo",
        channel=["FPRodAvgF", "FPRodAvgR"],
        axis_limits=[(0, 20), (1e-4, None)],
        annotate_at=(5, 15),
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# channel:      channel to histogram
# axis_limits:  [(bin_min, bin_max), (count_min, count_max)]

HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot(
        name="Plank Power Distribution",
        channel="PPlank_F",
        axis_limits=[(1, 51), (None, None)],
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# BAR PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# metrics: tuple of ("channel",) or (("channel", "aggregation"),)
# aggregations: "integral" "sum" "last" "mean" "max" "min"
# target_line:  draw a horizontal dashed reference line at this value

BAR_PLOT_DEFINITIONS = [
    # ── Multiple metrics with integral aggregation ─────────────────────────────
    BarPlot(
        name="Cumulative Metrics",
        metrics=(("dmInjector (kg/s)", "integral"), ("PMGUK_Deploy (MJ)", "integral"), ("PMGUK_Charge (MJ)", "integral")),
    ),

    # ── Single metric with max aggregation ─────────────────────────────────────
    BarPlot(
        name="Plank Energy",
        metrics=(("EPlank_F", "max"),),
    ),

    # ── Target line — horizontal reference for pass/fail comparison ────────────
    BarPlot(
        name="Plank Energy Target Demo",
        metrics=(("EPlank_F", "max"),),
        target_line=500.0,
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# BOX PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# channels:          channel or list of channels to plot
# aggregation_mode:  "per_run" (one box per run) | "aggregated" (all merged)
# gate:              filter data — ('channel', 'operator', value) or list of conditions

BOX_PLOT_DEFINITIONS = [
    BoxPlot(
        name="Low Speed Corner Distribution",
        channels="vCar",
        aggregation_mode="per_run",
        gate=[("gLong", "between", (-0.1, 0.1)), ("vCar", "<", 120)],
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# x_channel, y_channel: axes of the 2D grid
# z_channel:  None → count-based (2D histogram) | channel name → aggregation of z
# aggregation: "mean" "median" "std" "sum" "max" "min" (used with z_channel)
# bins:        number of bins per axis

HEATMAP_PLOT_DEFINITIONS = [
    # ── Count-based heatmap (2D histogram) ─────────────────────────────────────
    HeatmapPlot(
        name="gLat vs gLong Density",
        x_channel="gLat",
        y_channel="gLong",
        bins=100,
    ),

    # ── Aggregation heatmap — mean of z_channel per bin ────────────────────────
    HeatmapPlot(
        name="Ride Height vs Speed (mean SM)",
        x_channel="vCar",
        y_channel="hRideF",
        z_channel="SM",
        aggregation="mean",
        bins=100,
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# POWERPOINT EXPORT MAP (optional)
# ═══════════════════════════════════════════════════════════════════════════════
# Maps slides to generated plot images using Slide() helper.
# Layouts: "main_plot" (full-width) | "double_plot" (two side-by-side images)
# Reference format: "type/Plot Name" — auto-converts to filename.

POWERPOINT_EXPORT_MAP = [
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("double_plot", "scatter/GG Plot",         "scatter/Engine Efficiency"),
    Slide("double_plot", "scatter/Gear Ratios",     "scatter/Front Heave"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Multi-Channel PSD Demo"),
]

# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_workflow(
        WORKFLOW_NAME,
        title=f"{WORKFLOW_NAME.upper()} PLOT GENERATION",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        psds=PSD_PLOT_DEFINITIONS,
        histograms=HISTOGRAM_PLOT_DEFINITIONS,
        bars=BAR_PLOT_DEFINITIONS,
        boxes=BOX_PLOT_DEFINITIONS,
        heatmaps=HEATMAP_PLOT_DEFINITIONS,
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )
