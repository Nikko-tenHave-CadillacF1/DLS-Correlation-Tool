"""
Box plot analysis entry point.

This script keeps only the box-plot configuration and the minimal runner needed
to generate the plots.
"""

import numpy as np

from data_layout import BOXPLOT_OUTPUT_DIR, resolve_boxplot_input_dir
from plot_runtime import (
    build_plot_groups,
    build_plotter as runtime_build_plotter,
    run_plot_job,
)
from plot_shared_config import CHANNEL_MAPPINGS, UNITS_MAP


ROOT_FOLDER = resolve_boxplot_input_dir()

RUNS = [
    {
        "name": "T01BCN - R4",
        "file": "26T01BCN_260129_MAC26-01_PER_R04PARTIAL.txt",
        "color": "#D70000",
        "type": "CAR",
    },
    {
        "name": "T01BCN - R5",
        "file": "26T01BCN_260129_MAC26-01_PER_R05PARTIAL.txt",
        "color": "#06B300",
        "type": "CAR",
    },
    {
        "name": "T01BCN - R6",
        "file": "26T01BCN_260129_MAC26-01_PER_R06PARTIAL.txt",
        "color": "#008CFF",
        "type": "CAR",
    },
    {
        "name": "T01BCN - R7",
        "file": "26T01BCN_260129_MAC26-01_PER_R07PARTIAL.txt",
        "color": "#EA00FF",
        "type": "CAR",
    },
]

CHANNEL_TRANSFORMS = {
    "CAR": None,
}

CALCULATED_CHANNELS = {
    "dmInjector (kg/s)": lambda df: df["dmInjector"] / 3600,
    "rLambda_avg (%)": lambda df: 100 * (df["rLambdaL"] + df["rLambdaR"]) / 2,
    "dmExhaust": lambda df: df["dmInjector"] * (1 + 13.23 * df["rLambda_avg (%)"] / 100) / 3600,
    "dmExhaust_Estimated": lambda df: df["dmInjector"] * (1 + 13.23 * (0.155 * df["vCar"] + 109.329) / 100) / 3600,
    "Error_dmExhaust": lambda df: df["dmExhaust"] - df["dmExhaust_Estimated"],
    "CosPhi_Calc": lambda df: df["gLong"] / np.sqrt(df["gLat"] ** 2 + df["gLong"] ** 2),
}

LOW_PASS_FILTERS = {
    "rLambdaL": {"cutoff": 5, "order": 2},
    "rLambdaR": {"cutoff": 5, "order": 2},
    "rLambda_avg": {"cutoff": 3, "order": 2},
    "dmInjector": {"cutoff": 5, "order": 2},
    "dmFFMFuel": {"cutoff": 5, "order": 2},
    "CosPhi": {"cutoff": 5, "order": 2},
    "CosPhi_Calc": {"cutoff": 5, "order": 2},
    "gLong": {"cutoff": 3, "order": 2},
    "gLat": {"cutoff": 5, "order": 2},
}

BOX_PLOT_DEFINITIONS = [
    [
        "Low Speed Corner Distribution",
        "vCar",
        "per_run",
        (None, None),
        [("gLong", "between", (-0.1, 0.1)), ("vCar", "<", 120)],
        {},
    ],
]

BOX_PLOT_SETTINGS = {
    "show_points": True,
    "jitter": 0.15,
    "point_alpha": 0.25,
    "point_size": 18,
    "show_fliers": False,
    "box_width": 0.65,
    "box_linewidth": 1.8,
    "box_edge_color": "#4A4A4A",
    "medianline_color": "#1A1A1A",
    "medianline_width": 2.5,
    "aggregated_box_color": "#2E7D99",
    "aggregated_box_alpha": 0.75,
    "per_run_box_alpha": 0.75,
    "figsize_single_channel": (10, 6),
    "figsize_multi_channel": (14, 10),
}

PLOT_DEFINITIONS = build_plot_groups([], [], [], [], [], BOX_PLOT_DEFINITIONS)


def build_plotter():
    return runtime_build_plotter(
        root_folder=ROOT_FOLDER,
        output_dir=BOXPLOT_OUTPUT_DIR,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS,
        units_map=UNITS_MAP,
        box_plot_settings=BOX_PLOT_SETTINGS,
    )


if __name__ == "__main__":
    run_plot_job(
        title="BOX PLOT ANALYSIS",
        plotter=build_plotter(),
        plot_method="generate_box_plots",
        generate_message="Generating box plots...",
    )
