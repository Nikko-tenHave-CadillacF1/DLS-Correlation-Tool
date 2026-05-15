"""Box plot workflow — edit RUNS and BOX_PLOT_DEFINITIONS to configure your analysis."""

from plot_runtime import build_plot_groups, workflow_config, run_from_config, parse_plot_cli
from plot_runtime import BoxPlot

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
    _cfg = workflow_config(
        "boxplots",
        title="BOX PLOT ANALYSIS",
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        plot_method="generate_box_plots",
        generate_message="Generating box plots...",
    )
    run_from_config(_cfg, parse_plot_cli("Box plot analysis"))
