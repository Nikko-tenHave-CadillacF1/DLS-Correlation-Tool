"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot,
)

WORKFLOW_NAME = "tests"
EVENT = None
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    # Split a single parquet into one run per nRun value. Children are named
    # WD_1, WD_2, ... and inherit the parent's type / units / sample rate.
    {"name": "WD", "type": "OC",
     "file": "WD Scan/20260624-OC-VPG - Sensitivity Check - Mini WD Scan - v1-SPB.parquet",
     "split_by": "nRun", "color_range": ("#FF0000", "#4800FF")},
]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = False
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = _OUTPUT_DIR / "Correlation_Report.pptx"
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
        subplot_heights=(0.4, 0.7, 0.4, 0.4, 0.4),
        # highlight_zones=('SM', '>', 0.5)
    ),
    WaveformPlot(
        name="Power Unit",
        channels=('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), None, (10000,), None, None),
        subplot_heights=(0.4, 0.4, 0.6, 0.4, 0.4, 0.4),
    ),
    # WaveformPlot(
    #     name="OC SM Check",
    #     channels=('PMGUK', ('vCar', 'NGear'), "aUndersteerFromSlip", 'pBrakeF', ('rThrottle', 'SM')),
    #     axis_limits=(None, ((60, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
    #     reference_lines=((-350, 0, 350), None, (0,), None, None),
    #     subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    # ),
    WaveformPlot(
        name="Sensitivities",
        channels=(('vCar', 'NGear'), ('dtLap_dhCoGStatic', "dtLap_dhCoGStatic_Integral"), ('dtLap_dxCoGStatic', "dtLap_dxCoGStatic_Integral"), ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        subplot_heights=(0.8, 0.5, 0.5, 0.5),
    ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Long Acceleration",       "vCar",          "gLong"),
    ScatterPlot("Lat Acceleration",        "vCar",          "gLat_Abs"),
    ScatterPlot("GG Plot",                 "gLat",          "gLong"),
    ScatterPlot("Braking Efficiency",      "pBrakeF",       "gLong",
                best_fit=[('y', None, -0.2)],  gate=('gLong', '<', 0)),
    ScatterPlot("Understeer Plot",         "vCar",          "aUndersteerFromSlip"),
    ScatterPlot("Yaw Rate Response",       "aSteerWheel",   "nYaw",
                axis_limits=[(-160, 160), (None, None)],    best_fit=[('x', -20, 20)]),
    ScatterPlot("Lateral Acceleration Response", "aSteerWheel", "gLat",
                axis_limits=[(-160, 160), (None, None)],    best_fit=[('x', -20, 20)]),

    ## OC SUSPENSION CORRELATION - FOR OC CHECKS
    ScatterPlot("Front Heave",             "xHubVertF_Avg",   "FzTyreF_Avg",
                 best_fit=[('y', 2500, None), ('y', None, 2500)]),
    ScatterPlot("Front Roll",              "xHubVertF_Delta", "FzTyreF_Delta",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xHubVertR_Avg",   "FzTyreR_Avg",
                 best_fit=[('y', None, 5000), ('y', 5000, None)]),
    ScatterPlot("Rear Roll",               "xHubVertR_Delta", "FzTyreR_Delta",          best_fit=[('x', None, None)]),
    
    ScatterPlot("Roll angle gLat",         "gLat",          "aRoll",                best_fit=[('x', None, None)]),
    ScatterPlot("Front Ride vCar",         "vCar",          "hRideF",  best_fit=[('SM', 0, 0.5)],             
                axis_limits=[(None, None), (None, 40)],
                annotate_fit_at=(100,200,300)),
    ScatterPlot("Rear Ride vCar",          "vCar",          "hRideR",  best_fit=[('SM', 0, 0.5)],            
                axis_limits=[(None, None), (None, 75)],
                annotate_fit_at=(100,200,300)),
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR"),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = []
# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = []

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────


BAR_PLOT_DEFINITIONS = [
    BarPlot("COG Sensitivies", (("dtLap_dhCoGStatic", "integral"), ("dtLap_dxCoGStatic_Integral", "max"))),
    BarPlot("Aero Sensitivites", (("dtLap_dCDragTotal_Integral", "max"), ("dtLap_dCLiftTotal_Integral", "max"))),
    BarPlot("Lap Time",           (("tLap_Calc",         "max"),)),
    BarPlot("Time in SM Zones",   (("time_in_SM_100",       "last"), ("time_in_SM_90",        "last"), ("time_in_SM_80",        "last"))),
    BarPlot("Ratio Time in SM Zones",   (("ratio_time_in_SM_100",       "last"), ("ratio_time_in_SM_90",        "last"), ("ratio_time_in_SM_80",        "last"))),
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
    Slide("double_plot", "psd/FL gHub PSD",               "psd/FR gHub PSD"),
    Slide("double_plot", "psd/RL gHub PSD",               "psd/RR gHub PSD"),
    Slide("main_plot",   "waveform/Plank Wear"),
    Slide("double_plot", "scatter/Plank power acceleration", "histogram/Plank Power Distribution"),
]

# ─────────────────────────────────────────────────────────────────────────────

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
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )
