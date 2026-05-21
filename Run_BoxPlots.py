"""Box plot workflow — edit RUNS and BOX_PLOT_DEFINITIONS to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs
from engine import run_workflow, BoxPlot

WORKFLOW_NAME = "boxplots"
EVENT = "26T01BCN"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

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

if __name__ == "__main__":
    run_workflow(
        WORKFLOW_NAME,
        title="BOX PLOT ANALYSIS",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        boxes=BOX_PLOT_DEFINITIONS,
    )
