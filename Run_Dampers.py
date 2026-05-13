"""Damper workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import DAMPER_INPUT_DIR as _INPUT_DIR, DAMPER_PLOTS_DIR
from plot_runtime import build_plot_groups, PlotJobConfig, run_from_config, parse_plot_cli
from plot_runtime import WaveformPlot, ScatterPlot
from channel_config import (
    CHANNEL_MAPPINGS, UNITS_MAP, CHANNEL_TRANSFORMS,
    DAMPER_CALCULATED, DAMPER_FILTERS,
)

ROOT_FOLDER = _INPUT_DIR

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

PLOT_DEFINITIONS = build_plot_groups(waveforms=WAVEFORM_PLOT_DEFINITIONS, scatters=SCATTER_PLOT_DEFINITIONS)

_FIG_SIZE = [(9.5, 8), (10, 8), (10, 8), (10, 8), (10, 6)]

if __name__ == "__main__":
    _cfg = PlotJobConfig(
        title="DAMPER PLOT ANALYSIS",
        root_folder=ROOT_FOLDER,
        output_dir=DAMPER_PLOTS_DIR,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=DAMPER_CALCULATED,
        filters=DAMPER_FILTERS,
        units_map=UNITS_MAP,
        fig_size=_FIG_SIZE,
        generate_message="Generating damper plots...",
    )
    run_from_config(_cfg, parse_plot_cli("Damper plot analysis"))
