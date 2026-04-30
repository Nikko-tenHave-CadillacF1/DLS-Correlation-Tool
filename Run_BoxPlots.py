"""Box plot workflow — edit RUNS and BOX_PLOT_DEFINITIONS to configure your analysis."""

from channel_config import BOXPLOT_INPUT_DIR as _INPUT_DIR, BOXPLOT_OUTPUT_DIR
from plot_runtime import build_plot_groups, PlotJobConfig, run_from_config, parse_plot_cli
from plot_runtime import BoxPlot
from channel_config import (
    CHANNEL_MAPPINGS, UNITS_MAP, CHANNEL_TRANSFORMS,
    BOXPLOT_CALCULATED, BOXPLOT_FILTERS, BOX_PLOT_SETTINGS,
)

ROOT_FOLDER = _INPUT_DIR

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    {"name": "T01BCN - R4", "file": "26T01BCN_260129_MAC26-01_PER_R04PARTIAL.txt", "color": "#D70000", "type": "CAR"},
    {"name": "T01BCN - R5", "file": "26T01BCN_260129_MAC26-01_PER_R05PARTIAL.txt", "color": "#06B300", "type": "CAR"},
    {"name": "T01BCN - R6", "file": "26T01BCN_260129_MAC26-01_PER_R06PARTIAL.txt", "color": "#008CFF", "type": "CAR"},
    {"name": "T01BCN - R7", "file": "26T01BCN_260129_MAC26-01_PER_R07PARTIAL.txt", "color": "#EA00FF", "type": "CAR"},
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
# aggregation_mode: "per_run" (one box per run) | "aggregated" (all runs merged)
# gate: filter data before plotting — ('channel', 'operator', value) or list of conditions

BOX_PLOT_DEFINITIONS = [
    BoxPlot(
        name="Low Speed Corner Distribution",
        channels="vCar",
        aggregation_mode="per_run",
        gate=[("gLong", "between", (-0.1, 0.1)), ("vCar", "<", 120)],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────

PLOT_DEFINITIONS = build_plot_groups(boxes=BOX_PLOT_DEFINITIONS)

if __name__ == "__main__":
    _cfg = PlotJobConfig(
        title="BOX PLOT ANALYSIS",
        root_folder=ROOT_FOLDER,
        output_dir=BOXPLOT_OUTPUT_DIR,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=BOXPLOT_CALCULATED,
        low_pass_filters=BOXPLOT_FILTERS,
        units_map=UNITS_MAP,
        box_plot_settings=BOX_PLOT_SETTINGS,
        plot_method="generate_box_plots",
        generate_message="Generating box plots...",
    )
    run_from_config(_cfg, parse_plot_cli("Box plot analysis"))
