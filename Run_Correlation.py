"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot, HeatmapPlot,
)

WORKFLOW_NAME = "correlation"
EVENT = "26R08SPB"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    {
        "name": "LTS",
        "file": r"VPG Baselines  SPB  26R08SPB v0 LF_LTS_Iteration_3.parquet",
        "color": "#1400CA",
        "nlap": 1,
        "type": "DLS",
    },
    {
        "name": "OC",
        "file": r"20260615-OC-XPG - 26R08SPB - RVS Corr v0 - v1-SPB.parquet",
        "color": "#2F7600",
        "nrun": 1,
        "type": "OC",
    },
]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = True
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
    WaveformPlot(
        name="Plank Wear",
        channels=('PMGUK', 'vCar', 'FzPlankF', 'EPlank_F', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0, 7500), None, (0, 100), None),
        subplot_heights=(0.4, 0.6, 0.4, 0.6, 0.4, 0.4),
        # show_delta=(False, False, False, True, False, False)
    ),
    # WaveformPlot(
    #     name="DIL TELEM",
    #     channels=('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
    #     axis_limits=((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-180, 180), ((None, None), (None, None))),
    #     reference_lines=(None, None, (-350, 0, 350), None, (0,), None),
    #     subplot_heights=(0.15, 0.2, 0.3, 0.5, 0.3, 0.3),
    # ),
    # WaveformPlot(
    #     name="OC SM Check",
    #     channels=('PMGUK', ('vCar', 'NGear'), "aUndersteerFromSlip", 'pBrakeF', ('rThrottle', 'SM')),
    #     axis_limits=(None, ((60, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
    #     reference_lines=((-350, 0, 350), None, (0,), None, None),
    #     subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    # ),
    # WaveformPlot(
    #     name="Damper Velocities",
    #     channels=(('vCar', 'NGear'), "vDamperDeltaF", "vDamperDeltaR", 'pBrakeF', ('rThrottle', 'SM')),
    #     axis_limits=(((60, 400), (-1, 9)), None, None, None, ((0, 105), (0, 1.3))),
    #     reference_lines=(None, (-100,100), (-100, 100), None, None),
    #     subplot_heights=(0.8, 0.5, 0.5, 0.5, 0.5),
    # ),
    WaveformPlot(
        name="Brake Powers",
        channels=(('vCar', 'NGear'), 'PBrakeFL', 'PBrakeFR', 'PBrakeRL', 'PBrakeRR', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, None, None, None, None, None),
        subplot_heights=(0.8, 0.5, 0.5, 0.5, 0.5, 0.5),
    ),
    WaveformPlot(
        name="Brake Energies",
        channels=(('vCar', 'NGear'), 'EBrakeFL', 'EBrakeFR', 'EBrakeRL', 'EBrakeRR', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, None, None, None, None, None),
        subplot_heights=(0.8, 0.5, 0.5, 0.5, 0.5, 0.5),
    ),
    WaveformPlot(
        name="tyre slip ratios",
        channels=(('vCar', 'NGear'), 'rSlipTyreFL', 'rSlipTyreFR', 'rSlipTyreRL', 'rSlipTyreRR', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, None, None, None, None, None),
        subplot_heights=(0.8, 0.5, 0.5, 0.5, 0.5, 0.5),
        highlight_zones=[('rSlipTyreRL', '<', 0), ('rSlipTyreRR', '<', 0)],
    ),

    # WaveformPlot(
    #     name="TPG",
    #     channels=(('vCar', 'NGear'), ('aCamberFLKinematic', 'aCamberFRKinematic'),  ('aCamberRLKinematic', 'aCamberRRKinematic')),
    #     axis_limits=(((60, 400), (-1, 9)), None, None),
    #     reference_lines=(None, (0,0), (0,0)),
    #     subplot_heights=(0.8, 0.8, 0.8),
    #     show_delta = (False, True, True),
    # ),
    # WaveformPlot(
    #     name="APG Waveform",
    #     channels=('vCar', "vAir", "vWindHead", "CLiftTotalF", "CLiftTotalR", "CLiftTotal", "SC_CLT", "rAerobal"),
    #     axis_limits=((50, 400), None, None, None, None, None, None, None),
    #     reference_lines=(None, None, None, None, None, None, None, None),
    #     subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
    # ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


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

    # ## CAR ABSOLUTE OFFSETS - FOR DIL OFFSETS
    # ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
    #             best_fit=[('y', -8000, None), ('y', None, -8000)], error_as_factor=True),
    # ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)], error_as_factor=True),
    # ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
    #             best_fit=[('y', None, 15000), ('y', 15000, None)], error_as_factor=True),
    # ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)], error_as_factor=True),

    # ## CAR SUSPENSION CORRELATION - FOR REPORT
    # ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
    #             best_fit=[('y', None, 7500), ('y', 9000, None)]),
    # ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)]),
    # ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
    #             best_fit=[('y', None, 13000), ('y', 16500, None)]),
    # ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)]),

    ## OC SUSPENSION CORRELATION - FOR OC CHECKS
    ScatterPlot("Front Heave",             "xHubVertF_Avg",   "FzTyreF_Avg",
                 best_fit=[('y', 2500, None), ('y', None, 2500)]),
    ScatterPlot("Front Roll",              "xHubVertF_Delta", "FzTyreF_Delta",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xHubVertR_Avg",   "FzTyreR_Avg",
                 best_fit=[('y', None, 5000), ('y', 5000, None)]),
    ScatterPlot("Rear Roll",               "xHubVertR_Delta", "FzTyreR_Delta",          best_fit=[('x', None, None)]),
    
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
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR",               best_fit=0),
    ScatterPlot("Plank power acceleration",    "gLong (raw)", "PPlank_F",           best_fit=0),
    ScatterPlot("Front vDamperDelta vs FPRodDelta", "vDamperDeltaF", "FPRodDeltaF", best_fit=[('x', None, None)]),
    ScatterPlot("Rear vDamperDelta vs FPRodDelta", "vDamperDeltaR", "FPRodDeltaR", best_fit=[('x', None, None)]),
    ScatterPlot("Brake Power Balance", "PBrakeF_Avg", "PBrakeR_Avg", best_fit=[('x', 200, None)]),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = [
    PsdPlot("Front Vertical Acceleration PSD", "gVertF",       axis_limits=[(0, 20), (1e-4, None)], annotate_at=(9,)),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR",       axis_limits=[(0, 20), (1e-4, None)], annotate_at=(9, 15)),
    PsdPlot("Front Ride PSD",                  "hRideF (raw)", axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    PsdPlot("Rear Ride PSD",                   "hRideR (raw)", axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    # PsdPlot("Front Heave PSD",                 ["FPRodAvgF", "FPRodAvgR"],    axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    # PsdPlot("Front Roll PSD",                  ["FPRodDeltaF", "FPRodDeltaR"],  axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),
    # PsdPlot("FL gHub PSD",                  "gHubVertFL",    axis_limits=[(0, 20), (1e-3, None)], annotate_at=(5, 15)),
    # PsdPlot("FR gHub PSD",                  "gHubVertFR",    axis_limits=[(0, 20), (1e-3, None)], annotate_at=(5, 15)),
    # PsdPlot("RL gHub PSD",                  "gHubVertRL",    axis_limits=[(0, 20), (1e-3, None)], annotate_at=(5, 15)),
    # PsdPlot("RR gHub PSD",                  "gHubVertRR",    axis_limits=[(0, 20), (1e-3, None)], annotate_at=(5, 15)),
]

# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot("Plank Power Distribution", "PPlank_F", axis_limits=[(1, 51), (None, None)]),
]

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────


BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics", (("dmInjector (kg/s)", "integral"), ("PMGUK_Deploy (MJ)", "integral"), ("PMGUK_Charge (MJ)", "integral"))),
    BarPlot("Brake Energies Bar", (("EBrakeFL", "max"), ("EBrakeFR", "max"), ("EBrakeRL", "max"), ("EBrakeRR", "max"))),
    BarPlot("CPLV", (("CPLV_Front", "last"), ("CPLV_Rear", "last"))),
    BarPlot("Plank Energy",       (("EPlank_F",          "max"),)),
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
        boxes=BOX_PLOT_DEFINITIONS,
        heatmaps=HEATMAP_PLOT_DEFINITIONS,
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )
