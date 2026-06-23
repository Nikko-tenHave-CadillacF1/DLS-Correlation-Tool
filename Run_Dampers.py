"""Damper workflow — edit RUNS and plot definitions to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs
from engine import run_workflow, WaveformPlot, ScatterPlot, PsdPlot

WORKFLOW_NAME = "dampers"
EVENT = "26R06MCO"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    {"folder": ".", "filetype": ".parquet", "nlap": 1, "type": "DLS"},
]

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────

WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="Load Transfer Roll",
        channels=('rLLTD', 'aRoll', 'gLat_Abs'),
        axis_limits=((0, 100), (-1, 1), None),
        reference_lines=(None, None, 0),
        x_limits=(3920, 3980),
    ),
    WaveformPlot(
        name="rLLTD",
        channels=('rLLTD',),
        axis_limits=((35, 70),),
        x_limits=(3400, 3440),
    ),
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel'),
        axis_limits=(None, ((60, 400), (-1, 9)), (-160, 160)),
        subplot_heights=(0.4, 0.8, 0.4),
        x_limits=(1200, 1800),
    ),
    WaveformPlot(
        name="gVert",
        channels=('gVert', 'gVertF', 'gVertR'),
        axis_limits=((-2, 4), (-2, 4), (-2, 4)),
        reference_lines=(1, 1, 1),
        x_limits=(680, 820),
    ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────

SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("rLLTD vs. CosPhi", "rLLTD", "CosPhi"),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────

PSD_PLOT_DEFINITIONS = [
    PsdPlot("Front Vertical Acceleration PSD", "gVertF",       axis_limits=[(0, 20), (1e-4, None)], lorentz_fit=(5, 10)),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR",       axis_limits=[(0, 20), (1e-4, None)], lorentz_fit=(4, 8)),
    PsdPlot("Front Ride PSD",                  "hRideF (raw)", axis_limits=[(0, 20), (1e-4, None)], lorentz_fit=(3, 7)),
    PsdPlot("Rear Ride PSD",                   "hRideR (raw)", axis_limits=[(0, 20), (1e-4, None)], lorentz_fit=(3, 7)),

    PsdPlot("FPushrodFL PSD", "FProdFL_High",  axis_limits=[(0, 20), (1e4, None)]),
    PsdPlot("FPushrodFR PSD", "FProdFR_High",  axis_limits=[(0, 20), (1e4, None)]),
    PsdPlot("FPushrodRL PSD", "FProdRL_High",  axis_limits=[(0, 20), (1e4, None)]),
    PsdPlot("FPushrodRR PSD", "FProdRR_High",  axis_limits=[(0, 20), (1e4, None)]),

    PsdPlot("Heave Mode PSD - ungated",  "FPRodHeave", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(4, 7)),
    PsdPlot("Pitch Mode PSD - ungated",  "FPRodPitch", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(7, 11)),
    PsdPlot("Roll Mode PSD - ungated",   "FPRodRoll",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)]),
    PsdPlot("Warp Mode PSD - ungated",   "FPRodWarp",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(20, 30)),

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
        fig_size={"waveform": (9.5, 8)},
    )
