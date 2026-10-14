"""Correlation workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    BarPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)

WORKFLOW_NAME = "correlation"
EVENT = "26R18COT"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    # {
    #     "name": "CAR",
    #     "file": r"26R17SIN_261010_MAC26-02_PER_Q_R02.txt",
    #     "color": "#AD4500",
    #     "type": "CAR"
    # },
    {
        "name": "DLS",
        "file": r"26R18COT - VPG v1 LF_DLS.parquet",
        "color": "#0034B9",
        "nlap": 1,
        "type": "DLS"
    },
    {
        "name": "OC",
        "file": r"20261012-OC-XPG - 26R18COT - RVS Corr V1 LF - v3-Several.parquet",
        "color": "#007B2F",
        "nrun": 1,
        "type": "OC",
    },
    # # Split a single parquet into one run per nRun value. Children are named
    # # WD_1, WD_2, ... and inherit the parent's type / units / sample rate.
    # {"name": "WD", "type": "OC",
    #  "file": "WD Scan/20260624-OC-VPG - Sensitivity Check - Mini WD Scan - v1-SPB.parquet",
    #  "split_by": "nRun", "color_range": ("#FF0000", "#4800FF")},
]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
# Set POWERPOINT_OUTPUT = None to disable the export.
POWERPOINT_TEMPLATE    = resolve_template_path("oc_template.pptx")
POWERPOINT_OUTPUT      = _OUTPUT_DIR / "OC_Correlation_Report.pptx"
POWERPOINT_START_SLIDE = 4  # skip cover / summary / summary table

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────


WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((None, 400), (-1, 9)) ,None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,),None, None),
        subplot_heights=(0.4, 0.7, 0.4, 0.4, 0.4),
        show_delta=(False, True, False, False, False),
        # highlight_zones=('SM', '>', 0.5)
    ),
    WaveformPlot(
        name="Brake Powers",
        channels=(('vCar', 'NGear'), 'PMGUK', 'PBrakeFL', 'PBrakeRL', 'rMDecelTot', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, (-5, 105), ((0, 105), (0, 1.3))),
        subplot_heights=(0.8, 0.5, 0.5, 0.5, 0.5, 0.5),
    ),
    WaveformPlot(
        name="TPG",
        channels=(('vCar', 'NGear'), ('aCamberFLKinematic', 'aCamberFRKinematic'), ('aCamberRLKinematic', 'aCamberRRKinematic')),
        axis_limits=(((60, 400), (-1, 9)), None, None),
        reference_lines=(None, (0, 0), (0, 0)),
        subplot_heights=(0.8, 0.8, 0.8),
        show_delta=(False, True, True),
    ),
    WaveformPlot(
        name="Power Unit",
        channels=('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), None, (10000,), None, None),
        subplot_heights=(0.4, 0.4, 0.6, 0.4, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Ride Heights Waveform",
        channels=(('vCar', 'NGear'), 'hRideF', 'hRideR', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, (0,), (0,), None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.5, 0.5),
    ),
    WaveformPlot(
        name="Corner Radius",
        channels=('PMGUK', ('vCar', 'NGear'), 'cCorner', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.5, 0.7, 0.5, 0.5, 0.5),
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
    ScatterPlot("Gear Ratios", "nWheelAvg_R", "nEngine",
                best_fit=[('NGear', g - 0.5, g + 0.5) for g in range(2, 9)],
                show_equations=False),
    ScatterPlot("Engine Power",            "nEngine",       "PEngine"),
    ScatterPlot("Engine Efficiency",       "dmInjector",    "PEngine",              best_fit=1),
    ScatterPlot("Long Acceleration",       "vCar",          "gLong"),
    ScatterPlot("Lat Acceleration",        "vCar",          "gLat_Abs"),
    ScatterPlot("GG Plot",                 "gLat",          "gLong"),
    ScatterPlot("Braking Efficiency",      "pBrakeF",       "gLong",
                best_fit=[('y', None, -0.2)],  gate=('gLong', '<', 0)),
    ScatterPlot("Understeer Plot",         "vCar",          "aUndersteerFromSlip"),
    # aSteerF, not aSteerWheel: OC has a linear rack.
    ScatterPlot("Yaw Rate Response",       "aSteerF",       "nYaw",
                axis_limits=[(-10, 10), (None, None)],      best_fit=[('x', -1, 1)]),
    ScatterPlot("Lateral Acceleration Response", "aSteerF", "gLat",
                axis_limits=[(-10, 10), (None, None)],      best_fit=[('x', -1, 1)]),
    ScatterPlot("rLLTD vs vCar",           "vCar",          "rLLTD",
                axis_limits=[(None, None), (40, 70)],       gate=[("gLat_Abs", '>', 0.5), ("SM", '<', 0.5)]),

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
                annotate_fit_at=(100,200,300), gate=('pBrakeF', '<', 1)),
    ScatterPlot("Rear Ride vCar",          "vCar",          "hRideR",  best_fit=[('SM', 0, 0.5)],
                axis_limits=[(None, None), (None, 75)],
                annotate_fit_at=(100,200,300), gate=('pBrakeF', '<', 1)),
    ScatterPlot("Ride Height Compare",         "hRideF",    "hRideR"),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = []
# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = []

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────


BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics", (("dmInjector (kg/s)", "integral"), ("PMGUK_Deploy (MJ)", "integral"), ("PMGUK_Charge (MJ)", "integral"))),
    BarPlot("Brake Energies Bar", (("EBrakeFL", "max"), ("EBrakeFR", "max"), ("EBrakeRL", "max"), ("EBrakeRR", "max")), secondary_axis=False),
    BarPlot("COG Sensitivies", (("dtLap_dhCoGStatic", "integral"), ("dtLap_dxCoGStatic_Integral", "max"))),
    BarPlot("Aero Sensitivites", (("dtLap_dCDragTotal_Integral", "max"), ("dtLap_dCLiftTotal_Integral", "max"))),
    BarPlot("Lap Time",           (("tLap_Report",       "max"),)),
    BarPlot("Time in SM Zones",   (("time_in_SM_100",       "last"), ("time_in_SM_90",        "last"), ("time_in_SM_80",        "last")),         secondary_axis=False,),
    BarPlot("Ratio Time in SM Zones",   (("ratio_time_in_SM_100",       "last"), ("ratio_time_in_SM_90",        "last"), ("ratio_time_in_SM_80",        "last")),         secondary_axis=False,),
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
BOX_PLOT_DEFINITIONS = []

# ─── HEATMAP PLOTS ────────────────────────────────────────────────────────────
HEATMAP_PLOT_DEFINITIONS = []

# ─── POWERPOINT EXPORT MAP ────────────────────────────────────────────────────
# Maps slides to generated plot images using Slide() helper.
# Layouts: "main_plot" (full-width) | "double_plot" (two side-by-side images)
# Reference format: "type/Plot Name" — auto-converts to filename.

# Slide order matches oc_template.pptx (slides 4-18).

POWERPOINT_EXPORT_MAP = [
    # Driver input
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("main_plot",   "waveform/Brake Powers"),
    Slide("double_plot", "bar/Brake Energies Bar",           "bar/Ratio Time in SM Zones"),
    Slide("main_plot",   "waveform/TPG"),
    # Power unit
    Slide("main_plot",   "waveform/Power Unit"),
    Slide("double_plot", "scatter/Gear Ratios",              "scatter/Engine Power"),
    Slide("double_plot", "bar/Cumulative Metrics",           "scatter/Engine Efficiency"),
    # Handling
    Slide("double_plot", "scatter/Long Acceleration",        "scatter/Lat Acceleration"),
    Slide("double_plot", "scatter/GG Plot",                  "scatter/Understeer Plot"),
    Slide("double_plot", "scatter/Yaw Rate Response",        "scatter/Lateral Acceleration Response"),
    Slide("double_plot", "scatter/Braking Efficiency",       "scatter/rLLTD vs vCar"),
    # Ride
    Slide("main_plot",   "waveform/Ride Heights Waveform"),
    Slide("double_plot", "scatter/Front Ride vCar",          "scatter/Rear Ride vCar"),
    Slide("double_plot", "scatter/Ride Height Compare",      "scatter/Roll angle gLat"),
    Slide("main_plot",   "waveform/Corner Radius"),
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
        powerpoint_template=POWERPOINT_TEMPLATE,
        powerpoint_output=POWERPOINT_OUTPUT,
        export_map=POWERPOINT_EXPORT_MAP,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
    )


if __name__ == "__main__":
    main()
