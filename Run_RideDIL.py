"""Ride/DIL workflow — edit RUNS and plot definitions to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot, HeatmapPlot,
)

WORKFLOW_NAME = "ride_dil"
EVENT = "26P01BCN"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────


RUNS = [
    # {
    #     "name": "D1-R2",
    #     "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R02PARTIAL.txt",
    #     "color": "#FF0000",
    #     #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
    #     "type": "CAR",
    # },
    {
        "name": "D1-R3",
        "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R03PARTIAL.txt",
        "color": "#12B700",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
    {
        "name": "D1-R4",
        "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R04PARTIAL.txt",
        "color": "#001EFF",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
    {
        "name": "D1-R5",
        "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R05PARTIAL.txt",
        "color": "#A600FF",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
    # {
    #     "name": "D1-R7",
    #     "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R07PARTIAL.txt",
    #     "color": "#FF00FF",
    #     #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
    #     "type": "CAR",
    # },
    # {
    #     "name": "BOT R",
    #     "file": r"26R07BCN_260614_MAC26-03_BOT_GP_R02.txt",
    #     "color": "#D80000",
    #     #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
    #     "type": "CAR",
    # },
    # {
    #     "name": "DLS - BSL",
    #     "file": r"Barcelona_260612_GMDiL-08_PAG_R19PARTIAL_1.txt",
    #     "color": "#0400FF",
    #     "nlap": 1, # selects the run with the lowest nRun value (best lap) for each plot type
    #     "type": "DLS",
    # },
    # {
    #     "name": "DLS - COR",
    #     "file": r"Barcelona_260612_GMDiL-08_PAG_R19PARTIAL_1.txt",
    #     "color": "#0400FF",
    #     "nlap": 1, # selects the run with the lowest nRun value (best lap) for each plot type
    #     "type": "DLS",
    # },
]

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = False
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = _OUTPUT_DIR / "DIL_Ride_Report.pptx"
# Slide number (1-based) where the first POWERPOINT_EXPORT_MAP entry is placed.
# Leaves cover / intro slides untouched.
POWERPOINT_START_SLIDE = 4

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────


WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="DIL TELEM",
        channels=('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
        axis_limits=((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-180, 180), ((None, None), (None, None))),
        reference_lines=(None, None, (-350, 0, 350), None, (0,), None),
        subplot_heights=(0.15, 0.2, 0.3, 0.5, 0.3, 0.3),
    ),
    # WaveformPlot(
    #     name="Prod Forces",
    #     channels=(('vCar', 'NGear'), 'FPushrodFL', 'FPushrodFR', 'FPushrodRL', 'FPushrodRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Damper Displacements",
    #     channels=(('vCar', 'NGear'), 'xDamperFL', 'xDamperFR', 'xDamperRL', 'xDamperRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Damper Variations",
    #     channels=(('vCar', 'NGear'), 'xDamperVarFL', 'xDamperVarFR', 'xDamperVarRL', 'xDamperVarRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Prod Force Variations",
    #     channels=(('vCar', 'NGear'), 'FProdVarFL', 'FProdVarFR', 'FProdVarRL', 'FProdVarRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
                best_fit=[('y', -7500, None), ('y', None, -9000)]),
    ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
                best_fit=[('y', None, 13000)]), #, ('y', 16500, None)
    ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)]),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = [
    # nperseg=256 @ 100 Hz resample -> 2.56 s Welch window, ~0.39 Hz resolution.
    # Tuned for grip-limited corner gating: at 100 Hz, most corners (2-8 s)
    # contribute at least one Welch periodogram. Larger nperseg (512/1024) was
    # rejected for the gated modes because short corners (Roll/Warp on tight
    # sections) failed the segment-length requirement and were skipped.
    # Ride modes of interest (1-20 Hz) are still well resolved.
    # ── Ride modes from pushrod forces ────────────────────────────────────────
    PsdPlot("Heave Mode PSD - ungated",  "FPRodHeave", nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(5.5),       lorentz_fit=5.5,         gate=None),
    PsdPlot("Pitch Mode PSD - ungated",  "FPRodPitch", nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(9),         lorentz_fit=9.0,         gate=None),
    PsdPlot("Roll Mode PSD - ungated",   "FPRodRoll",  nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(5.5,10.5),  lorentz_fit=[5.5, 10.5], gate=None),
    PsdPlot("Warp Mode PSD - ungated",   "FPRodWarp",  nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(25),        lorentz_fit=25.0,        gate=None),

    PsdPlot("Heave Mode PSD - abs",  "FPRodHeave", nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(5.5),       lorentz_fit=5.5,         gate=None, log_scale=False),
    PsdPlot("Pitch Mode PSD - abs",  "FPRodPitch", nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(9),         lorentz_fit=9.0,         gate=None, log_scale=False),
    PsdPlot("Roll Mode PSD - abs",   "FPRodRoll",  nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(5.5,10.5),  lorentz_fit=[5.5, 10.5], gate=None, log_scale=False),
    PsdPlot("Warp Mode PSD - abs",   "FPRodWarp",  nperseg=512, axis_limits=[(0, 30), (1e4, None)], annotate_at=(25),        lorentz_fit=25.0,        gate=None, log_scale=False),

    PsdPlot("FPushrod FL PSD - ungated",  "FPushrodFL", nperseg=512, axis_limits=[(0, 20), (1e4, None)], annotate_at=(4, 9), gate=None, log_scale=False),
    PsdPlot("FPushrod FR PSD - ungated",  "FPushrodFR", nperseg=512, axis_limits=[(0, 20), (1e4, None)], annotate_at=(6), gate=None, log_scale=False),
    PsdPlot("FPushrod RL PSD - ungated",   "FPushrodRL",  nperseg=512, axis_limits=[(0, 20), (1e4, None)], annotate_at=(5, 9), gate=None, log_scale=False),
    PsdPlot("FPushrod RR PSD - ungated",   "FPushrodRR",  nperseg=512, axis_limits=[(0, 20), (1e4, None)], annotate_at=(5, 9), gate=None, log_scale=False),

    # ── Vertical chassis accelerations (ungated -> use larger nperseg for resolution) ──
    PsdPlot("Front Vertical Acceleration PSD", "gVertF", nperseg=512, axis_limits=[(0, 30), (None, None)], annotate_at=(9, 16),   lorentz_fit=[9.0, 16.0]),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR", nperseg=512, axis_limits=[(0, 30), (None, None)], annotate_at=(5.5, 9), lorentz_fit=[5.5, 9.0]),
    PsdPlot("Front Vertical Acceleration PSD - ABS", "gVertF", nperseg=512, axis_limits=[(0, 30), (None, None)], annotate_at=(9, 16),   lorentz_fit=[9.0, 16.0], log_scale=False),
    PsdPlot("Rear Vertical Acceleration PSD - ABS",  "gVertR", nperseg=512, axis_limits=[(0, 30), (None, None)], annotate_at=(5.5, 9), lorentz_fit=[5.5, 9.0],  log_scale=False),


    PsdPlot("hRideF PSD", "hRideF (raw)", nperseg=512, axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 9, 15)),
    PsdPlot("hRideR PSD", "hRideR (raw)", nperseg=512, axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 14.5)),

    PsdPlot("hRideF PSD - abs", "hRideF (high)", nperseg=512, axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 9, 15), log_scale=False),
    PsdPlot("hRideR PSD - abs", "hRideR (high)", nperseg=512, axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 14.5), log_scale=False),

    PsdPlot("xDamperFL", "xDamperFL_High", nperseg=512, axis_limits=[(None, None), (None, None)], log_scale=False),
    PsdPlot("xDamperFR", "xDamperFR_High", nperseg=512, axis_limits=[(None, None), (None, None)], log_scale=False),
    PsdPlot("xDamperRL", "xDamperRL_High", nperseg=512, axis_limits=[(None, None), (None, None)], log_scale=False),
    PsdPlot("xDamperRR", "xDamperRR_High", nperseg=512, axis_limits=[(None, None), (None, None)], log_scale=False),

    PsdPlot("FPushrodFL PSD", "FProdFL_High", nperseg=512, axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    PsdPlot("FPushrodFR PSD", "FProdFR_High", nperseg=512, axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    PsdPlot("FPushrodRL PSD", "FProdRL_High", nperseg=512, axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    PsdPlot("FPushrodRR PSD", "FProdRR_High", nperseg=512, axis_limits=[(0, 20), (1e4, None)], log_scale=False),
]

# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────
HISTOGRAM_PLOT_DEFINITIONS = [
]

# ─── BAR PLOTS ────────────────────────────────────────────────────────────────


BAR_PLOT_DEFINITIONS = [
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
    Slide("double_plot", "psd/Heave Mode PSD",                  "psd/Pitch Mode PSD"),
    Slide("double_plot", "psd/Roll Mode PSD",                   "psd/Warp Mode PSD"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Rear Vertical Acceleration PSD"),
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
