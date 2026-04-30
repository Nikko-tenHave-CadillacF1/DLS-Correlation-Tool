"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import CORRELATION_INPUT_DIR as _INPUT_DIR, CORRELATION_OUTPUT_DIR, resolve_template_path
from plot_runtime import build_plot_groups, build_plotter as _build_plotter, run_plot_job
from plot_runtime import WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot
from channel_config import (
    CHANNEL_MAPPINGS, UNITS_MAP, CHANNEL_TRANSFORMS,
    CORRELATION_CALCULATED, CORRELATION_FILTERS,
    SCATTER_MAX_POINTS, BAR_SECONDARY_AXIS_RATIO,
)

ROOT_FOLDER = _INPUT_DIR

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
        "name": "CAR",
        "file": r"26R03SUZ_260328_MAC26-02_BOT_Q_R03_1.txt",
        "color": "#FF8C00",
        "type": "CAR",
    },
    {
        "name": "DLS",
        "file": r"26R03SUZ  77  Quali  Run 3 Q1R3  Stint 1 stint 3_-Plank Stiffness Halved_DLS.parquet",
        "color": "#0059FF",
        "nlap": 1, #OC uses nrun, DLS uses nLap
        "type": "DLS",
    },
]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = True
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = CORRELATION_OUTPUT_DIR / "Correlation_Report.pptx"

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
        axis_limits=(None, ((60, 400), (-1, 9)), (-160, 160), None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Power Unit",
        channels=('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((60, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), None, (10000,), None, None),
        subplot_heights=(0.4, 0.4, 0.6, 0.4, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Plank Wear",
        channels=('PMGUK', 'vCar', 'FzPlankF', 'EPlank_F', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0, 7500), (0, 100), None, None),
        subplot_heights=(0.4, 0.6, 0.4, 0.6, 0.4, 0.4),
    ),
    WaveformPlot(
        name="DIL TELEM",
        channels=('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
        axis_limits=((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-160, 160), ((None, None), (None, None))),
        reference_lines=(None, None, (-350, 0, 350), None, (0,), None),
        subplot_heights=(0.15, 0.2, 0.3, 0.5, 0.3, 0.3),
    ),
    WaveformPlot(
        name="OC SM Check",
        channels=('PMGUK', ('vCar', 'NGear'), 'Grip Sens.', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((60, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, None, None, None),
        subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    ),

    # ── Demo: highlight_zones ──────────────────────────────────────────────────
    # Shades x-regions where rThrottle < 20% (lift-off / braking zones) in red.
    # rThrottle is always 0-100% so this condition reliably produces zones on
    # any racing lap. Supply a 4th hex string to override the shading colour;
    # omit it to use each run's own colour.
    WaveformPlot(
        name="[Demo] Highlight Zones",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, None, (20,)),
        subplot_heights=(0.6, 0.4, 0.4),
        highlight_zones=('rThrottle', '<', 20, '#FF4444'),
    ),
    # ── Demo: normalise ─────────────────────────────────────────────────────────
    # Channels with very different scales (kW, km/h, %) shown on a shared [0,1]
    # axis. Useful for comparing signal shapes directly.
    WaveformPlot(
        name="[Demo] Normalised",
        channels=('PMGUK', 'vCar', 'pBrakeF', 'rThrottle'),
        subplot_heights=(0.4, 0.4, 0.4, 0.4),
        normalise=True,
    ),
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
    ScatterPlot("Understeer Plot",         "vCar",          "aUndersteerFromSlip",
                gate=("rThrottle", '<', 95)),
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
                best_fit=[('y', None, 10000), ('y', 10000, None)]),
    ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
                best_fit=[('y', None, -20000), ('y', -20000, None)]),
    ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)]),
    ScatterPlot("Roll angle gLat",         "gLat",          "aRoll",                best_fit=[('x', None, None)]),
    ScatterPlot("Front Pushrod vCar",      "vCar",          "FPRodAvgF",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1)]),
    ScatterPlot("Rear Pushrod vCar",       "vCar",          "FPRodAvgR",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1)]),
    ScatterPlot("Front Ride vCar",         "vCar",          "hRideF",               best_fit=[('SM', 0, 0.5)]),
    ScatterPlot("Rear Ride vCar",          "vCar",          "hRideR",               best_fit=[('SM', 0, 0.5)]),
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR",               best_fit=0),
    ScatterPlot("Ride Height Compare Gated",   "hRideF",    "hRideR",
                best_fit=0, gate=('SM', '<', 1)),
    ScatterPlot("Plank power acceleration",    "gLong (raw)", "PPlank_F",           best_fit=0),
    ScatterPlot("Driver Line",                 "xCar",      "yCar",
                best_fit=0, show_equations=False, show_error=False),

    # ── Demo: color_gate ────────────────────────────────────────────────────────
    # Points where SM < 0.3 (simulator margin near minimum) are drawn in magenta
    # so outlier/low-confidence regions are immediately visible against the run
    # color. The fit line still uses all (pre-gate) data.
    ScatterPlot(
        name="[Demo] Color Gate",
        x_channel="vCar",
        y_channel="hRideF",
        best_fit=[('SM', 0, 0.5)],
        color_gate=('SM', '<', 0.3, '#FF00CC'),
    ),

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
    PsdPlot("Front Vertical Acceleration PSD", "gVertF",       axis_limits=[(0, 50), (1e-4, None)]),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR",       axis_limits=[(0, 50), (1e-4, None)]),
    PsdPlot("Front Ride PSD",                  "hRideF (raw)", axis_limits=[(0, 50), (1e-4, None)]),
    PsdPlot("Rear Ride PSD",                   "hRideR (raw)", axis_limits=[(0, 50), (1e-4, None)]),

    # ── Demo: multi-channel PSD ─────────────────────────────────────────────────
    # Both front and rear ride channels overlaid on the same axes. Line style
    # cycles (solid → dashed) per channel; legend shows "RUN — channel".
    PsdPlot(
        name="[Demo] Ride PSD Front+Rear",
        channel=["hRideF (raw)", "hRideR (raw)"],
        axis_limits=[(0, 50), (1e-4, None)],
    ),
]

# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot("Plank Power Distribution", "PPlank_F", axis_limits=[(1, 51), (None, None)]),
]

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────
# metrics: ("channel",) or (("channel", "aggregation"),)
# aggregations: "integral" "sum" "last" "mean" "max" "min"

BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics", (("dmInjector (kg/s)", "integral"),)),
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
# Maps slide numbers to generated plot images.
# Layouts: "main_plot" (full-width) | "double_plot" (two side-by-side images)

POWERPOINT_EXPORT_MAP = {
    4:  {"layout": "main_plot",   "images": ["waveform_Driver_Input.png"]},
    5:  {"layout": "main_plot",   "images": ["waveform_Power_Unit.png"]},
    6:  {"layout": "double_plot", "images": ["scatter_Gear_Ratios.png",              "scatter_Engine_Power.png"]},
    7:  {"layout": "double_plot", "images": ["bar_Cumulative_Metrics.png",           "scatter_engine_efficiency.png"]},
    8:  {"layout": "double_plot", "images": ["scatter_Long_Acceleration.png",        "scatter_Lat_Acceleration.png"]},
    9:  {"layout": "double_plot", "images": ["scatter_GG_Plot.png",                  "scatter_Understeer_Plot.png"]},
    10: {"layout": "double_plot", "images": ["scatter_Yaw_Rate_Response.png",        "scatter_Lateral_Acceleration_Response.png"]},
    11: {"layout": "double_plot", "images": ["scatter_Braking_Efficiency.png",       "scatter_Steering_Moment.png"]},
    12: {"layout": "double_plot", "images": ["scatter_Damper_gLat_front.png",        "scatter_Damper_gLat_rear.png"]},
    13: {"layout": "double_plot", "images": ["scatter_Pushrod_gLat_front.png",       "scatter_Pushrod_gLat_rear.png"]},
    14: {"layout": "double_plot", "images": ["scatter_Front_Heave.png",              "scatter_Rear_Heave.png"]},
    15: {"layout": "double_plot", "images": ["scatter_Front_Roll.png",               "scatter_Rear_Roll.png"]},
    16: {"layout": "double_plot", "images": ["scatter_Front_Pushrod_vCar.png",       "scatter_Rear_Pushrod_vCar.png"]},
    17: {"layout": "double_plot", "images": ["scatter_Front_Ride_vCar.png",          "scatter_Rear_Ride_vCar.png"]},
    18: {"layout": "double_plot", "images": ["scatter_Ride_Height_Compare.png",      "scatter_Roll_angle_gLat.png"]},
    19: {"layout": "double_plot", "images": ["psd_Front_Vertical_Acceleration_PSD.png", "psd_Rear_Vertical_Acceleration_PSD.png"]},
    20: {"layout": "double_plot", "images": ["psd_Front_Ride_PSD.png",               "psd_Rear_Ride_PSD.png"]},
    21: {"layout": "main_plot",   "images": ["waveform_Plank_Wear.png"]},
    22: {"layout": "double_plot", "images": ["scatter_Plank_power_acceleration.png", "histogram_Plank_Power_Distribution.png"]},
}

# ─────────────────────────────────────────────────────────────────────────────

PLOT_DEFINITIONS = build_plot_groups(
    WAVEFORM_PLOT_DEFINITIONS, SCATTER_PLOT_DEFINITIONS,
    PSD_PLOT_DEFINITIONS, HISTOGRAM_PLOT_DEFINITIONS,
    BAR_PLOT_DEFINITIONS, BOX_PLOT_DEFINITIONS,
)

if __name__ == "__main__":
    run_plot_job(
        title="CORRELATION PLOT GENERATION",
        plotter=_build_plotter(
            root_folder=ROOT_FOLDER,
            output_dir=CORRELATION_OUTPUT_DIR,
            runs=RUNS,
            plot_definitions=PLOT_DEFINITIONS,
            channel_mappings=CHANNEL_MAPPINGS,
            channel_transforms=CHANNEL_TRANSFORMS,
            calculated_channels=CORRELATION_CALCULATED,
            low_pass_filters=CORRELATION_FILTERS,
            units_map=UNITS_MAP,
            template_path=POWERPOINT_TEMPLATE,
            export_map=POWERPOINT_EXPORT_MAP,
            scatter_max_points=SCATTER_MAX_POINTS,
            bar_secondary_axis_ratio=BAR_SECONDARY_AXIS_RATIO,
        ),
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
    )
