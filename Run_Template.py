"""
TEMPLATE — Reference configuration for the DLS Correlation Tool pipeline.

Copy this file as a starting point for new workflows. Each section below
demonstrates the available plot types, options, and run configurations.
Refer to this file for syntax, available parameters, and working examples.

Run ``python Run_Template.py --help`` for CLI options.
"""

from bootstrap import ensure_dependencies

ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    BarPlot,
    BoxPlot,
    BoxPlotGrid,
    HeatmapPlot,
    HistogramPlot,
    Marker,
    PsdPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)

# ─── WORKFLOW NAME & EVENT ────────────────────────────────────────────────────
# Change these to create a new workflow. Directories are auto-created:
#   Data/inputs/<WORKFLOW_NAME>/              — without EVENT
#   Data/inputs/<WORKFLOW_NAME>/<EVENT>/      — with EVENT (recommended)
#   Data/outputs/<WORKFLOW_NAME>/<EVENT>/     — plots are saved here
#
# Set EVENT = None to use a flat workflow folder without event separation.
# For cross-event comparisons, set EVENT = None and prefix filenames with the
# event subfolder (see RUNS examples below).
WORKFLOW_NAME = "template"
EVENT = "26R04MIA"  # e.g. "26R04MIA", "26R03SUZ", or None for no event separation

_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
# Each run needs:
#   name:  display label used in plots and legends
#   file:  path relative to the workflow input folder (Data/inputs/<workflow>/)
#   color: hex colour for this run's traces
#   type:  OC | CAR | DLS | DIL — selects channel mappings and transforms
#
# Optional keys:
#   nrun:       (parquet only) rank-based run selection; nrun=1 → lowest nRun value
#   nlap:       exact lap number filter; ignored when nrun is also set
#   reference:  set True on exactly one run to mark it as the workflow-wide
#               reference. This run is used as the baseline for waveform delta
#               subplots (``show_delta=True``) and for the ``tDiff`` channel
#               (lap-time delta vs reference at each sLap). If no run has
#               ``reference=True``, the first loaded run is used.
#
# ── Folder-based runs ────────────────────────────────────────────────────────
# Instead of an explicit ``file``, supply ``folder`` + ``filetype`` to load
# every matching file in a directory as its own run (auto-named & coloured):
#
#   {"folder": "2xStopChoc", "filetype": ".parquet", "type": "DLS", "nlap": 1}
#
# Optional folder-mode keys:
#   contains:    substring filter (case-insensitive) — only files whose name
#                contains this string are loaded. Useful for slicing a folder
#                by session, driver, lap qualifier, etc.
#   name_prefix: string prepended to each auto-generated run name
#   colors:      list of hex colours, cycled per file (overrides auto colours)

RUNS = [
    # ── DLS / LTS example ──────────────────────────────────────────────────────
    {
        "name": "DLS Baseline",
        "file": r"26R04MIA  PER Q1R3_DLS.parquet",
        "color": "#0083BF",
        "nlap": 1,
        "type": "DLS",
        "reference": True,  # baseline for show_delta and tDiff
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
        "file": "26R04MIA_260502_MAC26-01_PER_Q_R03_4.txt",
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

    # ── Folder-based runs ──────────────────────────────────────────────────────
    # Load every matching file in a directory as its own run. The ``contains``
    # key narrows the selection to filenames containing a given substring
    # (case-insensitive), so a single folder can drive multiple per-condition
    # configurations.
    #
    # {"folder": "2xStopChoc", "filetype": ".parquet", "contains": "FP1", "type": "DLS", "nlap": 1},
    # {"folder": "2xStopChoc", "filetype": ".txt",     "contains": "FP1", "type": "CAR"},

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

# ─── POWERPOINT EXPORT (optional) ─────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = False
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = _OUTPUT_DIR / "Report.pptx"
# Slide number (1-based) where the first POWERPOINT_EXPORT_MAP entry is placed.
# Leaves cover / intro slides untouched.
POWERPOINT_START_SLIDE = 4

# ─── CALCULATED CHANNELS (optional override) ─────────────────────────────────
# All workflows (including custom ones) automatically receive the full
# CALCULATED_CHANNELS dict from channel_config.py. Only define overrides here
# if you need workflow-specific derived channels not in the shared config.
#
# To ADD channels for this workflow only:
#   from channel_config import CALCULATED_CHANNELS as _SHARED_CALC
#   CALCULATED_CHANNELS = {
#       **_SHARED_CALC,
#       "MyCustomChannel": lambda df: df["A"] + df["B"],
#   }
#
# To declare explicit dependencies (when lambda body is too dynamic):
#   "EngineEff": calc_channel("nEngine", "tThrottle")(
#       lambda df: df["nEngine"] * df["tThrottle"] / 1000.0
#   ),

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────
# channels:         one entry per subplot row — 'channel' or ('left_ch', 'right_ch')
# axis_limits:      per-row y-limits — (ymin, ymax) or ((y1_min, y1_max), (y2_min, y2_max))
# reference_lines:  per-row horizontal lines — scalar, tuple of scalars, or None
# subplot_heights:  relative row heights (e.g. 0.4 = half of 0.8)
# x_channel:        x-axis channel, default "sLap". Use "tLap" for time-based.
# x_limits:         (x_min, x_max) to zoom to a section of the lap
# normalise:        True to normalise all channels 0–1 (useful for overlays)
# highlight_zones:  ('channel', 'op', threshold, color) — shade regions
# legend_position:  "top" (default) or "right" for vertical side legend
# show_delta:       True/False or per-row tuple — append difference rows (requires exactly 2 runs)
# markers:          list of Marker() objects — vertical reference lines
# annotate_at:      tuple of x-values where data values are read off and annotated per run

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

    # ── Show delta — difference row between runs ──────────────────────────────
    # Requires 2+ loaded runs. A thin delta row is appended below each primary
    # row where show_delta is True, showing (run_i − reference) for every
    # non-reference run in that run's color.
    # show_delta accepts:
    #   True/False        — apply to all rows uniformly
    #   (True, False, ..) — per-row control (must match length of channels)
    # The reference run is selected workflow-wide via ``"reference": True`` on
    # a RUN entry above (NOT per-plot). If no run is flagged, the first loaded
    # run is used.
    WaveformPlot(
        name="Delta Comparison",
        channels=('vCar', 'pBrakeF', 'rThrottle'),
        axis_limits=(None, None, (0, 105)),
        reference_lines=(None, None, None),
        subplot_heights=(0.6, 0.4, 0.4),
        show_delta=True,
    ),

    # ── Per-row delta — only show delta for selected channels ──────────────────
    WaveformPlot(
        name="Selective Delta Demo",
        channels=('vCar', 'pBrakeF', 'rThrottle'),
        axis_limits=(None, None, (0, 105)),
        subplot_heights=(0.6, 0.4, 0.4),
        show_delta=(True, False, True),
    ),

    # ── Lap-time difference (tDiff) — auto-computed cross-run channel ──────────
    # tDiff is computed automatically when 2+ runs are loaded: for each run,
    # tDiff = tLap_this − interp(sLap → tLap of the first loaded run).
    # The reference run's tDiff is identically zero.
    WaveformPlot(
        name="Lap Time Delta",
        channels=('vCar', 'tDiff'),
        axis_limits=(None, (-5,5)),
        reference_lines=(None, 0),
        subplot_heights=(0.6, 0.4),
    ),

    # ── Legend position — move legend to the right ─────────────────────────────
    WaveformPlot(
        name="Right Legend Demo",
        channels=('vCar', 'pBrakeF'),
        axis_limits=(None, None),
        subplot_heights=(0.6, 0.4),
        legend_position="right",
    ),

    # ── Annotate at — read off data values at specific x-positions ─────────────
    # Draws a vertical guide line at each x-value and annotates the interpolated
    # y-value for every run (dot + label). Secondary channels use square markers.
    WaveformPlot(
        name="Annotate At Demo",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        subplot_heights=(0.6, 0.4, 0.4),
        annotate_at=(500, 1000, 2000),
    ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────
# best_fit: None/0 = no fit | 1 = single fit | 2 = quadratic | list = segmented fits by condition
#   Segment format: ('channel', low, high) or ('x'/'y', low, high) for axis splits
# gate: filter data before plotting — ('channel', 'operator', value)
#   Operators: '>' '<' '>=' '<=' '==' '!=' 'between' 'outside'  |  list for AND conditions
# show_equations: show fit equation text on the plot
# show_error:     show gradient delta between runs (% or factor)
# error_as_factor: True to show delta as "x 1.10" instead of "+10.0%"
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

    # ── Error as factor — show gradient delta as multiplicative factor ─────────
    # Displays "x 1.10" instead of "+10.0%" for fit comparison.
    ScatterPlot(
        name="Error as Factor Demo",
        x_channel="xDamperAvgF",
        y_channel="FPRodAvgF",
        best_fit=[('y', -6000, None), ('y', None, -6000)],
        error_as_factor=True,
    ),

    # ── Quadratic fit — second-order polynomial ────────────────────────────────
    ScatterPlot(
        name="Quadratic Fit Demo",
        x_channel="vCar",
        y_channel="hRideF",
        best_fit=2,
        show_equations=True,
        gate=[('SM', '<', 0.5)],
    ),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
# channel:        single channel string or list of channels (overlaid)
# axis_limits:    [(freq_min, freq_max), (psd_min, psd_max)]
# annotate_at:    tuple of frequencies where PSD values are annotated per run
# log_scale:      True (default) for semilogy axis
# nperseg:        Welch window length override (≥8); None uses default 512
# gate:           segment-aware Welch — filter data before PSD computation
# show_envelope:  True to show ±1σ shading when multiple runs are present
# markers:        list of Marker() — static vertical reference lines
# lorentz_fit:    (f_lo, f_hi) tuple or list of such tuples — fits a single-DOF
#                 Lorentzian + baseline inside each [f_lo, f_hi] window and
#                 annotates the fitted f₀ and damping ratio ζ on the plot.
#                 The optimiser seeds f₀ at the in-window argmax. Example:
#                 lorentz_fit=[(4, 7), (13, 17)].

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

    # ── Gated PSD — compute PSD only on filtered segments ──────────────────────
    PsdPlot(
        name="Gated PSD Demo",
        channel="gVertF",
        axis_limits=[(0, 30), (1e-4, None)],
        gate=[('vCar', '>', 100), ('SM', '<', 0.5)],
        nperseg=256,
    ),

    # ── Show envelope — ±1σ shading across runs ───────────────────────────────
    PsdPlot(
        name="PSD Envelope Demo",
        channel="FPRodAvgF",
        axis_limits=[(0, 20), (1e-4, None)],
        show_envelope=True,
        nperseg=512,
    ),

    # ── Markers on PSD ─────────────────────────────────────────────────────────
    PsdPlot(
        name="PSD Markers Demo",
        channel="gVertF",
        axis_limits=[(0, 30), (1e-4, None)],
        markers=[
            Marker(x=5, label="5 Hz"),
            Marker(x=15, label="15 Hz", color="#FF6600"),
        ],
    ),

    # ── Lorentzian damping estimator ───────────────────────────────────────────
    # Provide an (f_lo, f_hi) window or a list of windows. Each curve gets a
    # single-DOF Lorentzian + baseline fit inside the window; f₀ is seeded
    # at the in-window argmax and free to roam across the full window. The
    # fitted f₀ and damping ratio ζ are annotated in the curve's colour.
    PsdPlot(
        name="Lorentz Fit Demo",
        channel="gVertF",
        axis_limits=[(0, 30), (1e-4, None)],
        lorentz_fit=[(4, 7), (13, 17)],
    ),
]

# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
# channel:      channel to histogram
# axis_limits:  [(bin_min, bin_max), (count_min, count_max)]
# log_scale:    True for log-scale y-axis (useful for long-tail distributions)
# markers:      list of Marker() — static vertical reference lines

HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot(
        name="Plank Power Distribution",
        channel="PPlank_F",
        axis_limits=[(1, 51), (None, None)],
    ),

    # ── Log-scale histogram ────────────────────────────────────────────────────
    HistogramPlot(
        name="Log-Scale Histogram Demo",
        channel="PPlank_F",
        axis_limits=[(0, 100), (None, None)],
        log_scale=True,
    ),

    # ── Histogram with markers ─────────────────────────────────────────────────
    HistogramPlot(
        name="Histogram Markers Demo",
        channel="vCar",
        axis_limits=[(50, 350), (None, None)],
        markers=[
            Marker(x=100, label="Low speed"),
            Marker(x=250, label="High speed", color="#D70000"),
        ],
    ),
]

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────
# metrics:              tuple of ("channel",) or (("channel", "aggregation"),)
# aggregations:         "integral" "abs_integral" "sum" "abs_sum" "mean"
#                       "median" "max" "min" "first" "last"
# default_aggregation:  fallback when metric tuple omits aggregation (default "last")
# reference_lines:      list of y-values to draw as horizontal dashed reference lines
# gate:                 pre-filter applied to every run before aggregation
# axis_limits:          (y_min, y_max) to override y-axis range

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

    # ── Reference lines — horizontal benchmark levels (multiple supported) ─────
    BarPlot(
        name="Plank Energy Target Demo",
        metrics=(("EPlank_F", "max"),),
        reference_lines=[300.0, 500.0],  # warn / fail thresholds
    ),

    # ── Custom axis limits and default aggregation ─────────────────────────────
    BarPlot(
        name="Bar Axis Limits Demo",
        metrics=(("vCar",), ("nEngine",)),
        default_aggregation="mean",
        axis_limits=(0, 400),
    ),
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
# channels:          channel or list of channels to plot
# aggregation_mode:  "per_run" (one box per run) | "aggregated" (all merged)
#                    "per_run_aggregated" (per-run boxes + aggregated box at end)
# gate:              filter data — ('channel', 'operator', value) or list of conditions
# axis_limits:       (y_min, y_max) to override y-axis range

BOX_PLOT_DEFINITIONS = [
    BoxPlot(
        name="Low Speed Corner Distribution",
        channels="vCar",
        aggregation_mode="per_run",
        gate=[("gLong", "between", (-0.1, 0.1)), ("vCar", "<", 120)],
    ),

    # ── Per-run + aggregated — individual boxes with a combined "ALL" box ──────
    BoxPlot(
        name="Combined Ride Height Distribution",
        channels=("hRideF", "hRideR"),
        aggregation_mode="per_run_aggregated",
    ),

    # ── Custom axis limits ─────────────────────────────────────────────────────
    BoxPlot(
        name="Box Axis Limits Demo",
        channels="hRideF",
        aggregation_mode="per_run",
        axis_limits=(20, 80),
    ),

    # ── BoxPlotGrid — 2D grid of box plots (rows × cols gating) ────────────────
    # Each cell combines row + column gate conditions (AND). Two render modes:
    #   "expand" — one individual BoxPlot figure per cell (default)
    #   "grid"   — single figure with a rows×cols subplot matrix
    BoxPlotGrid(
        name="Ride Height Grid",
        channels="hRideF",
        rows={
            "LS": [("vCar", "<", 120)],
            "MS": [("vCar", ">=", 120), ("vCar", "<", 200)],
            "HS": [("vCar", ">=", 200)],
        },
        cols={
            "Entry": [("gLong", "<", -0.5)],
            "Apex":  [("gLong", "between", (-0.5, 0.5))],
            "Exit":  [("gLong", ">", 0.5)],
        },
        aggregation_mode="aggregated",
        render_mode="grid",
    ),

    # ── BoxPlotGrid with expand mode — individual files per cell ───────────────
    BoxPlotGrid(
        name="Speed Band Yaw",
        channels="nYaw",
        rows={
            "Low Speed":  [("vCar", "<", 150)],
            "High Speed": [("vCar", ">=", 150)],
        },
        cols={
            "Left":  [("gLat", "<", -0.5)],
            "Right": [("gLat", ">", 0.5)],
        },
        aggregation_mode="per_run",
        render_mode="expand",
    ),
]

# ─── HEATMAP PLOTS ────────────────────────────────────────────────────────────
# x_channel, y_channel: axes of the 2D grid
# z_channel:    None → count-based (2D histogram) | channel name → aggregation of z
# aggregation:  "mean" "median" "std" "count" "sum" "max" "min" (used with z_channel)
# bins:         number of bins per axis (int or (nx, ny) tuple for non-square)
# cmap:         matplotlib colormap name (default "viridis")
# z_limits:     (z_min, z_max) to clamp colour bar range
# min_count:    cells with fewer points are masked (default 3)
# gate:         filter data before binning
# markers:      list of Marker() — static vertical reference lines
# axis_limits:  [(x_min, x_max), (y_min, y_max)]

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

    # ── Custom colormap and z_limits ───────────────────────────────────────────
    HeatmapPlot(
        name="Custom Cmap Demo",
        x_channel="vCar",
        y_channel="gLat",
        z_channel="hRideF",
        aggregation="median",
        bins=(80, 60),
        cmap="plasma",
        z_limits=(25, 70),
    ),

    # ── Gated heatmap with min_count and markers ───────────────────────────────
    HeatmapPlot(
        name="Gated Heatmap Demo",
        x_channel="vCar",
        y_channel="hRideR",
        z_channel="gLong",
        aggregation="mean",
        bins=60,
        gate=[('SM', '<', 0.5)],
        min_count=5,
        axis_limits=[(50, 350), (20, 80)],
        markers=[Marker(x=200, label="200 km/h")],
    ),
]

# ─── POWERPOINT EXPORT MAP (optional) ─────────────────────────────────────────
# Maps slides to generated plot images using Slide() helper.
# Layouts: "main_plot" (full-width) | "double_plot" (two side-by-side images)
# Reference format: "type/Plot Name" — auto-converts to filename.

POWERPOINT_EXPORT_MAP = [
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("double_plot", "scatter/GG Plot",         "scatter/Engine Efficiency"),
    Slide("double_plot", "scatter/Gear Ratios",     "scatter/Front Heave"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Multi-Channel PSD Demo"),
]

# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
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
        # ── Optional overrides (uncomment as needed) ───────────────────────────
        # verbose=True,             # enable debug-level logging
        # output_dpi=150,           # lower DPI for faster iteration (default 300)
        # scatter_max_points=30000, # max scatter points before decimation
        # open_output=False,        # don't auto-open output folder
        # fig_size=[12, 8],         # override figure size (width, height in inches)
        # calculated_channels=MY_EXTRA_CHANNELS,  # override shared CALCULATED_CHANNELS
        # filters=MY_FILTERS,       # override shared DEFAULT_FILTERS
    )


if __name__ == "__main__":
    main()