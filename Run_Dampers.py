"""Damper workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import get_workflow_dirs
from engine import run_workflow, WaveformPlot, ScatterPlot

WORKFLOW_NAME = "dampers"
EVENT = "26R04MIA"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    {"name": "Front Heave - 7A", "file": "VPG Baselines  MIA  26R04MIA v1b_-7A FRONT HEAVE_LTS_Iteration_8.parquet",    "color": "#D70000", "nlap": 1, "type": "DLS"},
    {"name": "Front Heave - 7B", "file": "VPG Baselines  MIA  26R04MIA v1b_-7B FRONT HEAVE_LTS_Iteration_8.parquet",    "color": "#059E00", "nlap": 1, "type": "DLS"},
    {"name": "Front Heave - 7C", "file": "VPG Baselines  MIA  26R04MIA v1b_- 7C FRONT HEAVE_LTS_Iteration_8.parquet",  "color": "#008CFF", "nlap": 1, "type": "DLS"},
]

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────

WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="Load Transfer Roll",
        channels=('rLLTD', 'aRoll', 'gLat_Abs'),
        axis_limits=((0, 100), (-1, 1), None),
        reference_lines=(None, None, 0),
        subplot_heights=(1, 1, 1),
        x_limits=(3920, 3980),
    ),
    WaveformPlot(
        name="rLLTD",
        channels=('rLLTD',),
        axis_limits=((35, 70),),
        reference_lines=(None,),
        subplot_heights=(1,),
        x_limits=(3400, 3440),
    ),
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel'),
        axis_limits=(None, ((60, 400), (-1, 9)), (-160, 160)),
        reference_lines=(None, None, None),
        subplot_heights=(0.4, 0.8, 0.4),
        x_limits=(1200, 1800),
    ),
    WaveformPlot(
        name="gVert",
        channels=('gVert', 'gVertF', 'gVertR'),
        axis_limits=((-2, 4), (-2, 4), (-2, 4)),
        reference_lines=(1, 1, 1),
        subplot_heights=(1, 1, 1),
        x_limits=(680, 820),
    ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────

SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("rLLTD vs. CosPhi", "rLLTD", "CosPhi", best_fit=0),
]

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_workflow(
        WORKFLOW_NAME,
        title="DAMPER PLOT ANALYSIS",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        fig_size={"waveform": (9.5, 8), "scatter": (10, 8), "psd": (10, 8), "histogram": (10, 8), "bar": (10, 6)},
    )
