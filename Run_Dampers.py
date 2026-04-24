"""Damper workflow entry point.

Keep this file focused on the plots used for dampers.
"""

import numpy as np

from data_layout import DAMPER_PLOTS_DIR, resolve_damper_input_dir
from plot_runtime import (
    build_plot_groups,
    build_plotter as runtime_build_plotter,
    run_plot_job,
)
from plot_shared_config import CHANNEL_MAPPINGS, UNITS_MAP


ROOT_FOLDER = resolve_damper_input_dir()

RUNS = [
    {
        "name": "Front Heave - 7A",
        "file": "VPG Baselines  MIA  26R04MIA v1b_-7A FRONT HEAVE_LTS_Iteration_8.parquet",
        "color": "#D70000",
        "nlap" : 1,
        "type": "DLS",
    },
    {
        "name": "Front Heave - 7B",
        "file": "VPG Baselines  MIA  26R04MIA v1b_-7B FRONT HEAVE_LTS_Iteration_8.parquet",
        "color": "#059E00",
        "nlap" : 1,
        "type": "DLS",
    },
    {
        "name": "Front Heave - 7C",
        "file": "VPG Baselines  MIA  26R04MIA v1b_- 7C FRONT HEAVE_LTS_Iteration_8.parquet",
        "color": "#008CFF",
        "nlap" : 1,
        "type": "DLS",
    },
]

CALCULATED_CHANNELS = {
    "gLat_Abs": lambda df: np.abs(df["gLat"]),
}

LOW_PASS_FILTERS = {
    "CosPhi": {"cutoff": 3, "order": 3},
    "rLLTD": {"cutoff": 10, "order": 2},
    "gVert": {"cutoff": 30, "order": 2},
    "gVertF": {"cutoff": 30, "order": 2},
    "gVertR": {"cutoff": 30, "order": 2},
    "all": {"cutoff": 10, "order": 3},
}

WAVEFORM_PLOT_DEFINITIONS = [
    [
        "Load Transfer Roll",
        ('rLLTD', 'aRoll', 'gLat_Abs'),
        ((0,100), (-1,1), None),
        (None, None, 0),
        (1, 1, 1),
        (3920, 3980)
    ],
    [
        "rLLTD",
        ('rLLTD',),
        ((35,70),),
        (None,),
        (1,),
        (3400, 3440)
    ],
    [
        "Driver Input",
        ('PMGUK', ('vCar', 'NGear'), 'aSteerWheel'),
        (None, ((60, 400), (-1, 9)), (-160, 160)),
        (None, None, None),
        (0.4, 0.8, 0.4),
        (1200, 1800) 
    ],
    [
        "gVert",
        ('gVert', 'gVertF', 'gVertR'),
        ((-2,4), (-2,4), (-2,4)),
        (1, 1, 1),
        (1, 1, 1),
        (680, 820)
    ],
]

SCATTER_PLOT_DEFINITIONS = [
    [
        "rLLTD vs. CosPhi",
        ("rLLTD", "CosPhi"),
        None,
        0,
        True,
        True,
    ],
]

PLOT_DEFINITIONS = build_plot_groups(WAVEFORM_PLOT_DEFINITIONS, SCATTER_PLOT_DEFINITIONS, [], [], [], [])


def build_plotter():
    """Build the configured plotter for the damper workflow."""
    FIG_SIZE = [(9.5, 8), (10, 8), (10, 8), (10, 8), (10, 6)]
    return runtime_build_plotter(
        root_folder=ROOT_FOLDER,
        output_dir=DAMPER_PLOTS_DIR,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS,
        units_map=UNITS_MAP,
        fig_size=FIG_SIZE,
    )


if __name__ == "__main__":
    # The shared runner handles console framing and any future export hooks.
    run_plot_job(
        title="DAMPER PLOT ANALYSIS",
        plotter=build_plotter(),
        generate_message="Generating damper plots...",
    )
