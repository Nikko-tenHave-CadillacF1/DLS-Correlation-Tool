"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import CORRELATION_OUTPUT_DIR, resolve_template_path
from plot_runtime import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot,
)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
# name:  display label used in plots and reports
# file:  path relative to ROOT_FOLDER (Data/inputs/correlation/)
# color: hex color for this run's traces
# type:  OC | CAR | DLS | DIL  — selects channel mappings and transforms
# nrun:  (parquet only) rank-based run selection; nrun=1 → lowest nRun value
# nlap:  exact lap number filter; ignored when nrun is also set

RUNS = [
    # {"name": "v37", "file": "...", "color": "#0051FF", "nrun": 1, "type": "OC"},
    {
        "name": "DRY",
        "file": r"VPG Baselines  MTL  26R05MTL v2b - No FARB_-DRY_LTS_Iteration_3.parquet",
        "color": "#D80000",
        "nlap": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "DLS",
    },
    {
        "name": "WET",
        "file": r"VPG Baselines  MTL  26R05MTL v2b - No FARB_-WETv2_LTS_Iteration_3.parquet",
        "color": "#0051FF",
        "nlap": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "DLS",
    },  

]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = True
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = CORRELATION_OUTPUT_DIR / "Correlation_Report.pptx"
# Slide number (1-based) where the first POWERPOINT_EXPORT_MAP entry is placed.
# Leaves cover / intro slides untouched.
POWERPOINT_START_SLIDE = 4

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────
# channels: one entry per subplot row — 'channel' or ('left_channel', 'right_channel')
# axis_limits: per-row y-limits — (ymin, ymax) or ((y1_min, y1_max), (y2_min, y2_max)) for dual rows
# reference_lines: per-row horizontal lines — scalar, tuple of scalars, or None
# subplot_heights: relative row heights (e.g. 0.4 = half height of 0.8)
# x_channel: x-axis channel, default "sLap". Use "tLap" for time-based.
# x_limits: (x_min, x_max) to zoom to a section of the lap

WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((60, 400), (-1, 9)), (-180, 180), None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
        # highlight_zones=('SM', '>', 0.5)
    ),
    WaveformPlot(
        name="Power Unit",
        channels=('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((50, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), None, (10000,), None, None),
        subplot_heights=(0.4, 0.4, 0.6, 0.4, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Plank Wear",
        channels=('PMGUK', 'vCar', 'FzPlankF', 'EPlank_F', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0, 7500), None, (0, 100), None),
        subplot_heights=(0.4, 0.6, 0.4, 0.6, 0.4, 0.4),
    ),
    WaveformPlot(
        name="DIL TELEM",
        channels=('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
        axis_limits=((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-180, 180), ((None, None), (None, None))),
        reference_lines=(None, None, (-350, 0, 350), None, (0,), None),
        subplot_heights=(0.15, 0.2, 0.3, 0.5, 0.3, 0.3),
    ),
    WaveformPlot(
        name="OC SM Check",
        channels=('PMGUK', ('vCar', 'NGear'), "aUndersteerFromSlip", 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((60, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    ),

    # ── Demo: highlight_zones ──────────────────────────────────────────────────
    # WaveformPlot(
    #     name="[Demo] Highlight Zones",
    #     channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
    #     axis_limits=(None, None, ((0, 105), (0, 1.3))),
    #     reference_lines=(None, None, (20,)),
    #     subplot_heights=(0.6, 0.4, 0.4),
    #     highlight_zones=('rThrottle', '<', 20, '#FF4444'),
    # ),
    # ── Demo: normalise ─────────────────────────────────────────────────────────
    # WaveformPlot(
    #     name="[Demo] Normalised",
    #     channels=('PMGUK', 'vCar', 'pBrakeF', 'rThrottle'),
    #     subplot_heights=(0.4, 0.4, 0.4, 0.4),
    #     normalise=True,
    # ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────
# best_fit: None/0 = no fit | 1 = single fit | list = segmented fits by condition
#   Segment format: ('channel', low, high) or ('x'/'y', low, high) for axis-based splits
# gate: filter data before plotting — ('channel', 'operator', value)
#   Operators: '>' '<' '>=' '<=' '==' 'between'  |  list for multiple conditions (all must match)

SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Gear Ratios", "nWheelAvg_R", "nEngine",
                best_fit=[('NGear', 1.5, 2.5), ('NGear', 2.5, 3.5), ('NGear', 3.5, 4.5),
                          ('NGear', 4.5, 5.5), ('NGear', 5.5, 6.5), ('NGear', 6.5, 7.5), ('NGear', 7.5, 8.5)],
                show_equations=False),
    ScatterPlot("Engine Power",            "nEngine",       "PEngine",              best_fit=0),
    ScatterPlot("Engine Efficiency",       "dmInjector",       "PEngine",
                best_fit=1, show_equations=True, show_error=True),
    ScatterPlot("Long Acceleration",       "vCar",          "gLong"),
    ScatterPlot("Lat Acceleration",        "vCar",          "gLat_Abs",             best_fit=0),
    ScatterPlot("GG Plot",                 "gLat",          "gLong",                best_fit=0),
    ScatterPlot("Braking Efficiency",      "pBrakeF",       "gLong",
                best_fit=[('y', None, -0.2)],  gate=('gLong', '<', 0)),
    ScatterPlot("Understeer Plot",         "vCar",          "aUndersteerFromSlip",   best_fit=0),
    ScatterPlot("Yaw Rate Response",       "aSteerWheel",   "nYaw",
                axis_limits=[(-160, 160), (None, None)],    best_fit=[('x', -20, 20)]),
    ScatterPlot("Lateral Acceleration Response", "aSteerWheel", "gLat",
                axis_limits=[(-160, 160), (None, None)],    best_fit=[('x', -20, 20)]),
    ScatterPlot("Steering Moment",         "aSteerWheel",   "MSteerWheel",
                axis_limits=[(-160, 160), (None, None)],    best_fit=0),
    ScatterPlot("Damper gLat front",       "gLat",          "xDamperDeltaF",        best_fit=[('x', None, None)]),
    ScatterPlot("Damper gLat rear",        "gLat",          "xDamperDeltaR",        best_fit=[('x', None, None)]),
    ScatterPlot("Pushrod gLat front",      "gLat",          "FPRodDeltaF",          best_fit=[('x', None, None)]),
    ScatterPlot("Pushrod gLat rear",       "gLat",          "FPRodDeltaR",          best_fit=[('x', None, None)]),
    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
                best_fit=[('y', -6000, None), ('y', None, -6000)]),
    ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
                best_fit=[('y', None, None)]),
    ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)]),
    ScatterPlot("Roll angle gLat",         "gLat",          "aRoll",                best_fit=[('x', None, None)]),
    ScatterPlot("Front Pushrod vCar",      "vCar",          "FPRodAvgF",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1)]),
    ScatterPlot("Rear Pushrod vCar",       "vCar",          "FPRodAvgR",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1)]),
    ScatterPlot("Front Ride vCar",         "vCar",          "hRideF",               best_fit=[('SM', 0, 0.5)],
                axis_limits=[(None, None), (None, 40)],
                annotate_fit_at=(100,200,300)),
    ScatterPlot("Rear Ride vCar",          "vCar",          "hRideR",               best_fit=[('SM', 0, 0.5)],
                axis_limits=[(None, None), (None, 75)],
                annotate_fit_at=(100,200,300)),
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR",               best_fit=0),
    ScatterPlot("Ride Height Compare Gated",   "hRideF",    "hRideR",
                best_fit=0, gate=('SM', '<', 1)),
    ScatterPlot("Plank power acceleration",    "gLong (raw)", "PPlank_F",           best_fit=0),
    ScatterPlot("Driver Line",                 "xCar",      "yCar",
                best_fit=0, show_equations=False, show_error=False),

    # ── Demo: color_gate ────────────────────────────────────────────────────────
    # ScatterPlot(
    #     name="[Demo] Color Gate",
    #     x_channel="vCar",
    #     y_channel="hRideF",
    #     best_fit=[('SM', 0, 0.5)],
    #     color_gate=('SM', '<', 0.3, '#FF00CC'),
    # ),

    # ── Demo: annotate_fit_at ───────────────────────────────────────────────────
    # Annotates the fit-line y-value for each run at vCar = 250 km/h with a
    # vertical dashed line and marker. Makes it easy to read off the delta
    # between runs at a specific operating point.
    ScatterPlot(
        name="[Demo] Annotate Fit At",
        x_channel="vCar",
        y_channel="hRideR",
        best_fit=1,
        gate=[('SM', '<', 0.5)],
        annotate_fit_at=250.0,
    ),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = [
    PsdPlot("Front Vertical Acceleration PSD", "gVertF",       axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR",       axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    PsdPlot("Front Ride PSD",                  "hRideF (raw)", axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    PsdPlot("Rear Ride PSD",                   "hRideR (raw)", axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    PsdPlot("Front Heave PSD",                 ["FPRodAvgF", "FPRodAvgR"],    axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    PsdPlot("Front Roll PSD",                  ["FPRodDeltaF", "FPRodDeltaR"],  axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    # ── Demo: multi-channel PSD ─────────────────────────────────────────────────
    # Both front and rear ride channels overlaid on the same axes. Line style
    # cycles (solid → dashed) per channel; legend shows "RUN — channel".
    # PsdPlot(
    #     name="gVertF PSD Filter Effect",
    #     channel=["gVertF (raw)", "gVertF"],
    #     axis_limits=[(0, 30), (1e-4, None)],
    # ),
]

# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot("Plank Power Distribution", "PPlank_F", axis_limits=[(1, 51), (None, None)]),
]

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────
# metrics: ("channel",) or (("channel", "aggregation"),)
# aggregations: "integral" "sum" "last" "mean" "max" "min"

BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics", (("dmInjector (kg/s)", "integral"), ("PMGUK_Deploy (MJ)", "integral"), ("PMGUK_Charge (MJ)", "integral"))),
    BarPlot("Plank Energy",       (("EPlank_F",          "max"),)),
    BarPlot("Lap Time",           (("tLap_Calc",         "max"),)),

    # ── Demo: target_line ───────────────────────────────────────────────────────
    # Draws a dashed reference line at the target peak plank energy. Runs above
    # the line are immediately flagged without needing to read axis values.
    BarPlot(
        name="[Demo] Plank Energy Target",
        metrics=(("EPlank_F", "max"),),
        target_line=500.0,
    ),
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
BOX_PLOT_DEFINITIONS = []

# ─── POWERPOINT EXPORT MAP ────────────────────────────────────────────────────
# Maps slides to generated plot images using Slide() helper.
# Layouts: "main_plot" (full-width) | "double_plot" (two side-by-side images)
# Reference format: "type/Plot Name" — auto-converts to filename.

POWERPOINT_EXPORT_MAP = [
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("main_plot",   "waveform/Power Unit"),
    Slide("double_plot", "scatter/Gear Ratios",              "scatter/Engine Power"),
    Slide("double_plot", "bar/Cumulative Metrics",           "scatter/Engine Efficiency"),
    Slide("double_plot", "scatter/Long Acceleration",        "scatter/Lat Acceleration"),
    Slide("double_plot", "scatter/GG Plot",                  "scatter/Understeer Plot"),
    Slide("double_plot", "scatter/Yaw Rate Response",        "scatter/Lateral Acceleration Response"),
    Slide("double_plot", "scatter/Braking Efficiency",       "scatter/Steering Moment"),
    Slide("double_plot", "scatter/Damper gLat front",        "scatter/Damper gLat rear"),
    Slide("double_plot", "scatter/Pushrod gLat front",       "scatter/Pushrod gLat rear"),
    Slide("double_plot", "scatter/Front Heave",              "scatter/Rear Heave"),
    Slide("double_plot", "scatter/Front Roll",               "scatter/Rear Roll"),
    Slide("double_plot", "scatter/Front Pushrod vCar",       "scatter/Rear Pushrod vCar"),
    Slide("double_plot", "scatter/Front Ride vCar",          "scatter/Rear Ride vCar"),
    Slide("double_plot", "scatter/Ride Height Compare",      "scatter/Roll angle gLat"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Rear Vertical Acceleration PSD"),
    Slide("double_plot", "psd/Front Ride PSD",               "psd/Rear Ride PSD"),
    Slide("main_plot",   "waveform/Plank Wear"),
    Slide("double_plot", "scatter/Plank power acceleration", "histogram/Plank Power Distribution"),
]

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_workflow(
        "correlation",
        title="CORRELATION PLOT GENERATION",
        runs=RUNS,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        psds=PSD_PLOT_DEFINITIONS,
        histograms=HISTOGRAM_PLOT_DEFINITIONS,
        bars=BAR_PLOT_DEFINITIONS,
        boxes=BOX_PLOT_DEFINITIONS,
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )
