"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    BarPlot,
    HistogramPlot,
    PsdPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)
from engine.plot_definitions import Scatter3DPlot

WORKFLOW_NAME = "correlation"
EVENT = "26R11BUD"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
# Supported "type" values: "CAR", "OC", "DIL", "DLS", "FMIOpt"
# (FMIOpt = LapSim/AVL-TR parquet output; handled like DLS with a reduced
# channel set — see CHANNEL_MAPPINGS["FMIOpt"] in channel_config.py.)

RUNS = [
    {
        "name": "CAR",
        "file": r"26R11BUD_260724_MAC26-02_PER_P1_R02PARTIAL.txt",
        "color": "#B96300",
        "type": "CAR",
    },
    {
        "name": "DLS",
        "file": r"1 FP1R2 nC5 Q Sim_DLS.parquet",
        "color": "#0017C8",
        "nlap": 1,
        "type": "DLS",
    },
]

# ─── POWERPOINT EXPORT ───────────────────────────────────────────────────────────────────────
# Corporate template with cover slides. Set POWERPOINT_OUTPUT = None to disable
# the export, or set POWERPOINT_TEMPLATE = None to fall back to a blank 16:9 deck.
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = None # _OUTPUT_DIR / "Correlation_Report.pptx"
POWERPOINT_START_SLIDE = 4  # skip cover / intro slides

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
                show_equations=False),
    ScatterPlot("Engine Power",            "nEngine",       "PEngine"),
    ScatterPlot("Engine Efficiency",       "dmInjector",       "PEngine",
                best_fit=1),
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
    ScatterPlot("Steering Moment",         "aSteerWheel",   "MSteerWheel",
                axis_limits=[(-160, 160), (None, None)]),

    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF", axis_limits=[(None, None), (0, None)],
                best_fit=[('y', None, 8500), ('y', 10000, None)]),
    ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
                best_fit=[('y', None, 12000), ('y', 15000, None)]),
    ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)]),

    ScatterPlot("Roll angle gLat",         "gLat",          "aRoll",                best_fit=[('x', None, None)]),
    ScatterPlot("Front Pushrod vCar",      "vCar",          "FPRodAvgF",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1), ("pBrakeF", '<', 1)]),
    ScatterPlot("Rear Pushrod vCar",       "vCar",          "FPRodAvgR",
                best_fit=[('gLat_Abs', 0, 1)], gate=[('SM', '<', 1), ("pBrakeF", '<', 1)]),
    ScatterPlot("Front Ride vCar",         "vCar",          "hRideF",  best_fit=[('SM', 0, 0.5)],
                axis_limits=[(None, None), (None, 40)],
                annotate_fit_at=(100,200,300)),
    ScatterPlot("Rear Ride vCar",          "vCar",          "hRideR",  best_fit=[('SM', 0, 0.5)],
                axis_limits=[(None, None), (None, 75)],
                annotate_fit_at=(100,200,300)),
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR"),
    ScatterPlot("Plank power acceleration",    "gLong (raw)", "PPlank_F"),
    ScatterPlot("rLLTD vs vCar", "vCar", "rLLTD", axis_limits=[(None, None), (40, 70)], gate=[("gLat_Abs", '>', 0.5), ("SM", '<', 0.5)]),
    ScatterPlot("rAerobal vs vCar", "vCar", "rAeroBal", axis_limits=None, gate=[("vCar", '>', 100), ("SM", '<', 0.5)]),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = [
    PsdPlot("Front Vertical Acceleration PSD", "gVertF",       axis_limits=[(0, 20), (1e-4, None)], nperseg=320),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR",       axis_limits=[(0, 20), (1e-4, None)], nperseg=320),
    PsdPlot("Front Ride PSD",                  "hRideF (raw)", axis_limits=[(0, 20), (1e-4, None)], nperseg=320),
    PsdPlot("Rear Ride PSD",                   "hRideR (raw)", axis_limits=[(0, 20), (1e-4, None)], nperseg=320),
    # PsdPlot("Front Heave PSD",                 ["FPRodAvgF", "FPRodAvgR"],    axis_limits=[(0, 20), (1e-4, None)], lorentz_fit=(3, 7)),
    # PsdPlot("Front Roll PSD",                  ["FPRodDeltaF", "FPRodDeltaR"],  axis_limits=[(0, 20), (1e-4, None)], lorentz_fit=(3, 7)),
    # PsdPlot("FL gHub PSD",                  "gHubVertFL",    axis_limits=[(0, 20), (1e-3, None)], lorentz_fit=(3, 7)),
    # PsdPlot("FR gHub PSD",                  "gHubVertFR",    axis_limits=[(0, 20), (1e-3, None)], lorentz_fit=(3, 7)),
    PsdPlot("RL gHub PSD",                  "gHubVertRL", nperseg=320,    axis_limits=[(0, 20), (1e-3, None)], log_scale=False),
    PsdPlot("RR gHub PSD",                  "gHubVertRR", nperseg=320,    axis_limits=[(0, 20), (1e-3, None)], log_scale=False),
    PsdPlot("Heave Mode PSD",  "FPRodHeave", axis_limits=[(0, 20), (None, None)], nperseg=320, log_scale=False, lorentz_fit=(3, 7)),
    PsdPlot("Pitch Mode PSD",  "FPRodPitch", axis_limits=[(0, 20), (None, None)], nperseg=320, log_scale=False, lorentz_fit=(5, 10)),
    PsdPlot("Roll Mode PSD",   "FPRodRoll",  axis_limits=[(0, 20), (None, None)], nperseg=320, log_scale=False, lorentz_fit=(3, 7)),
    PsdPlot("Warp Mode PSD",   "FPRodWarp",  axis_limits=[(0, 20), (None, None)], nperseg=320, log_scale=False, lorentz_fit=(10, 15)),

]

# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot("Plank Power Distribution", "PPlank_F", axis_limits=[(1, 51), (None, None)]),
]

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────


BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics", (("dmInjector (kg/s)", "integral"), ("PMGUK_Deploy (MJ)", "integral"), ("PMGUK_Charge (MJ)", "integral"))),
    # BarPlot("CPLV", (("CPLV_Front", "last"), ("CPLV_Rear", "last"))),
    BarPlot("Lap Time",           (("tLap_Calc",         "max"),)),
    # BarPlot("BrakeBal",  (("rBrakeBiasF",    "last"), ("rBrakeBias",     "last"))),
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
BOX_PLOT_DEFINITIONS = []

# ─── HEATMAP PLOTS ────────────────────────────────────────────────────────────
HEATMAP_PLOT_DEFINITIONS = []

# ─── DEBUG 3D SCATTER PLOTS (not exported, interactive) ───────────────────────────
SCATTER3D_PLOT_DEFINITIONS = [
    Scatter3DPlot(
        name="Engine Map (nEngine vs nBoost vs PEngine)",
        x_channel="nEngine",
        y_channel="nBoost",
        z_channel="PEngine",
    ),
]
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
    Slide("double_plot", "scatter/rAerobal vs vCar",        "scatter/rLLTD vs vCar"),
    Slide("double_plot", "scatter/Braking Efficiency",       "scatter/Steering Moment"),
    Slide("double_plot", "scatter/Front Heave",              "scatter/Rear Heave"),
    Slide("double_plot", "scatter/Front Roll",               "scatter/Rear Roll"),
    Slide("double_plot", "scatter/Front Pushrod vCar",       "scatter/Rear Pushrod vCar"),
    Slide("main_plot",   "waveform/Ride Heights Waveform"),
    Slide("double_plot", "scatter/Front Ride vCar",          "scatter/Rear Ride vCar"),
    Slide("double_plot", "scatter/Ride Height Compare",      "scatter/Roll angle gLat"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Rear Vertical Acceleration PSD"),
    Slide("double_plot", "psd/Front Ride PSD",               "psd/Rear Ride PSD"),
    Slide("double_plot", "psd/Heave Mode PSD",               "psd/Pitch Mode PSD"),
    Slide("double_plot", "psd/Roll Mode PSD",               "psd/Warp Mode PSD"),
    Slide("main_plot",   "waveform/Plank Wear"),
    Slide("double_plot", "scatter/Plank power acceleration", "histogram/Plank Power Distribution"),
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
        psds=PSD_PLOT_DEFINITIONS,
        histograms=HISTOGRAM_PLOT_DEFINITIONS,
        bars=BAR_PLOT_DEFINITIONS,
        scatter3d=SCATTER3D_PLOT_DEFINITIONS,
        powerpoint_template=POWERPOINT_TEMPLATE,
        powerpoint_output=POWERPOINT_OUTPUT,
        export_map=POWERPOINT_EXPORT_MAP,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )
    if plotter is not None and SCATTER3D_PLOT_DEFINITIONS:
        import matplotlib
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass
        plotter.plot_data(plot_types=["scatter3d"])


if __name__ == "__main__":
    main()
