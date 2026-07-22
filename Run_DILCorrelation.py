"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from bootstrap import ensure_dependencies

ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    BarPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)

WORKFLOW_NAME = "correlation"
EVENT = "26R10SPA"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
# Supported "type" values: "CAR", "OC", "DIL", "DLS", "FMIOpt"
# (FMIOpt = LapSim/AVL-TR parquet output; handled like DLS with a reduced
# channel set — see CHANNEL_MAPPINGS["FMIOpt"] in channel_config.py.)

RUNS = [
    {
        "name": "CAR",
        "file": r"26R10SPA_260718_MAC26-01_BOT_Q_R02.txt",
        "color": "#B93E00",
        "type": "CAR",
    },
    {
        "name": "DLS",
        "file": r"26R10SPA  77  P1  Run 2 P1R2  Stint 1 3_DLS.parquet",
        "color": "#0017C8",
        "nlap": 1,
        "type": "DLS",
    },
]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = True
POWERPOINT_TEMPLATE   = None
POWERPOINT_OUTPUT     = _OUTPUT_DIR / "DIL_Offline_Checks.pptx"
# Slide number (1-based) where the first POWERPOINT_EXPORT_MAP entry is placed.
# Leaves cover / intro slides untouched.
POWERPOINT_START_SLIDE = 4

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────


WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((None, 400), (-1, 9)) ,None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,),None, None),
        subplot_heights=(0.4, 0.7, 0.3, 0.3, 0.3),
        show_delta=(False, True, False, False, False),
        # highlight_zones=('SM', '>', 0.5)
    ),
    WaveformPlot(
        name="Power Unit",
        channels=('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), None, (10000,), None, None),
        subplot_heights=(0.4, 0.4, 0.6, 0.4, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Plank Wear",
        channels=('PMGUK', 'vCar', 'FzPlankF', 'EPlank_F', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0, 7500), None, (0, 100), None),
        subplot_heights=(0.4, 0.6, 0.4, 0.6, 0.4, 0.4),
        # show_delta=(False, False, False, True, False, False)
    ),
    WaveformPlot(
        name="Ride Heights Waveform",
        channels=(('vCar', 'NGear'), 'hRideF', 'hRideR', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, (0,), (0,), None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.5, 0.5),
    ),

]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Gear Ratios", "nWheelAvg_R", "nEngine",
                best_fit=[('NGear', 1.5, 2.5), ('NGear', 2.5, 3.5), ('NGear', 3.5, 4.5),
                          ('NGear', 4.5, 5.5), ('NGear', 5.5, 6.5), ('NGear', 6.5, 7.5), ('NGear', 7.5, 8.5)],
                show_equations=False, error_as_factor=True),
    ScatterPlot("Engine Power",            "nEngine",       "PEngine"),
    ScatterPlot("Engine Efficiency",       "dmInjector",       "PEngine",
                best_fit=1, error_as_factor=True),
    ScatterPlot("Long Acceleration",       "vCar",          "gLong"),
    ScatterPlot("Lat Acceleration",        "vCar",          "gLat_Abs"),
    ScatterPlot("GG Plot",                 "gLat",          "gLong"),
    ScatterPlot("Braking Efficiency",      "pBrakeF",       "gLong",
                best_fit=[('y', None, -0.2)],  gate=('gLong', '<', 0), error_as_factor=True),
    ScatterPlot("Understeer Plot",         "vCar",          "aUndersteerFromSlip"),
    ScatterPlot("Yaw Rate Response",       "aSteerWheel",   "nYaw",
                axis_limits=[(-160, 160), (None, None)],    best_fit=[('x', -20, 20)], error_as_factor=True),
    ScatterPlot("Lateral Acceleration Response", "aSteerWheel", "gLat",
                axis_limits=[(-160, 160), (None, None)],    best_fit=[('x', -20, 20)], error_as_factor=True),
    ScatterPlot("Steering Moment",         "aSteerWheel",   "MSteerWheel",
                axis_limits=[(-160, 160), (None, None)]),
    ## CAR ABSOLUTE OFFSETS - FOR DIL OFFSETS
    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
                best_fit=[('y', -8000, None), ('y', None, -8000)], error_as_factor=True),
    ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)], error_as_factor=True),
    ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
                best_fit=[('y', None, 15000), ('y', 15000, None)], error_as_factor=True),
    ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)], error_as_factor=True),

    ScatterPlot("Roll angle gLat",         "gLat",          "aRoll",                best_fit=[('x', None, None)], error_as_factor=True),
    ScatterPlot("Front Pushrod vCar",      "vCar",          "FPRodAvgF",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1), ("pBrakeF", '<', 1)], error_as_factor=True),
    ScatterPlot("Rear Pushrod vCar",       "vCar",          "FPRodAvgR",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1), ("pBrakeF", '<', 1)], error_as_factor=True),
    ScatterPlot("Front Ride vCar",         "vCar",          "hRideF",  best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)],
                axis_limits=[(None, None), (None, 40)],
                annotate_fit_at=(100,200,300), error_as_factor=True),
    ScatterPlot("Rear Ride vCar",          "vCar",          "hRideR",  best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)],
                axis_limits=[(None, None), (None, 75)],
                annotate_fit_at=(100,200,300), error_as_factor=True),
    ScatterPlot("Front Ride vCar - SM OFF",         "vCar",          "hRideF",  best_fit=[('SM', 0, 0.5)],
                axis_limits=[(None, None), (None, 40)],
                annotate_fit_at=(100,200,300), error_as_factor=True),
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR"),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
# (none for this workflow)

# ─── HISTOGRAM PLOTS ───────────────────────────────────────────────────────────
# (none for this workflow)

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────


BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics", (("dmInjector (kg/s)", "integral"), ("PMGUK_Deploy (MJ)", "integral"), ("PMGUK_Charge (MJ)", "integral"))),
    BarPlot("Lap Time",           (("tLap_Calc",         "max"),)),
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
BOX_PLOT_DEFINITIONS = []

# ─── HEATMAP PLOTS ────────────────────────────────────────────────────────────
HEATMAP_PLOT_DEFINITIONS = []

# ─── POWERPOINT EXPORT MAP ────────────────────────────────────────────────────
# Maps slides to generated plot images using Slide() helper.
# Layouts: "main_plot" (full-width) | "double_plot" (two side-by-side images)
# Reference format: "type/Plot Name" — auto-converts to filename.

POWERPOINT_EXPORT_MAP = [
    # ── 1. Overview / Driver ─────────────────────────────────────────────────
    Slide("main_plot",   "waveform/Driver Input"),
    # ── 2. Power Unit ────────────────────────────────────────────────────────
    Slide("main_plot",   "waveform/Power Unit"),
    Slide("double_plot", "scatter/Gear Ratios",                     "scatter/Engine Power"),
    Slide("double_plot", "scatter/Engine Efficiency",               "bar/Cumulative Metrics"),
    # ── 3. Vehicle Dynamics — g-forces ───────────────────────────────────────
    Slide("double_plot", "scatter/Long Acceleration",               "scatter/Lat Acceleration"),
    Slide("double_plot", "scatter/GG Plot",                         "bar/Lap Time"),
    # ── 4. Handling / Steering ───────────────────────────────────────────────
    Slide("double_plot", "scatter/Understeer Plot",                 "scatter/Steering Moment"),
    Slide("double_plot", "scatter/Yaw Rate Response",               "scatter/Lateral Acceleration Response"),
    # ── 5. Brakes ────────────────────────────────────────────────────────────
    Slide("main_plot",   "scatter/Braking Efficiency"),
    # ── 6. Suspension — Loads & Modes ────────────────────────────────────────
    Slide("double_plot", "scatter/Front Heave",                     "scatter/Rear Heave"),
    Slide("double_plot", "scatter/Front Roll",                      "scatter/Rear Roll"),
    Slide("double_plot", "scatter/Front Pushrod vCar",              "scatter/Rear Pushrod vCar"),
    # ── 7. Ride Height / Chassis Attitude ────────────────────────────────────
    Slide("main_plot",   "waveform/Ride Heights Waveform"),
    Slide("double_plot", "scatter/Front Ride vCar",                 "scatter/Rear Ride vCar"),
    Slide("double_plot", "scatter/Ride Height Compare",             "scatter/Roll angle gLat"),
    # ── 8. Plank Wear / Ground contact ───────────────────────────────────────
    Slide("main_plot",   "waveform/Plank Wear"),
]

# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    plotter = run_workflow(
        WORKFLOW_NAME,
        title=f"{WORKFLOW_NAME.upper()} PLOT GENERATION",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        bars=BAR_PLOT_DEFINITIONS,
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )



if __name__ == "__main__":
    main()
