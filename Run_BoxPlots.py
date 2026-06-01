"""Box plot workflow — edit RUNS and BOX_PLOT_DEFINITIONS to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs
from engine import run_workflow, BoxPlot, BoxPlotGrid, WaveformPlot

WORKFLOW_NAME = "boxplots"
EVENT = "26R05MTL"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

# RUNS = [
#     # PER (hot colours)
#     {"name": "PER P1R3", "file": "26R04MIA_260501_MAC26-01_PER_P1_R03PARTIAL.txt", "color": "#D70000", "type": "CAR"},
#     {"name": "PER SQ1R1", "file": "26R04MIA_260501_MAC26-01_PER_SQ_R01PARTIAL.txt", "color": "#FF6600", "type": "CAR"},
#     {"name": "PER Q1R3",  "file": "26R04MIA_260502_MAC26-01_PER_Q_R03.txt",         "color": "#FFD700", "type": "CAR"},
#     {"name": "PER SR", "file": "26R04MIA_260502_MAC26-01_PER_SR_R02.txt",        "color": "#FF3399", "type": "CAR"},
#     {"name": "PER GP", "file": "26R04MIA_260503_MAC26-01_PER_GP_R02.txt",        "color": "#CC0066", "type": "CAR"},
#     # BOT (cool colours)
#     {"name": "BOT P1R4", "file": "26R04MIA_260501_MAC26-02_BOT_P1_R04PARTIAL.txt", "color": "#008CFF", "type": "CAR"},
#     {"name": "BOT SQ1R1", "file": "26R04MIA_260501_MAC26-02_BOT_SQ_R01PARTIAL.txt", "color": "#00CC88", "type": "CAR"},
#     {"name": "BOT Q1R3",  "file": "26R04MIA_260502_MAC26-02_BOT_Q_R03.txt",         "color": "#4C00BF", "type": "CAR"},
#     {"name": "BOT SR", "file": "26R04MIA_260502_MAC26-02_BOT_SR_R02.txt",        "color": "#0055AA", "type": "CAR"},
#     {"name": "BOT GP", "file": "26R04MIA_260503_MAC26-02_BOT_GP_R02.txt",        "color": "#006666", "type": "CAR"},
# ]

# ─── 26R05MTL RUNS ──────────────────────────────────────────────────────────
RUNS = [
    # PER (hot colours)
    {"name": "PER P1R3", "file": "26R05MTL_260522_MAC26-01_PER_P1_R03PARTIAL.txt",  "color": "#D70000", "type": "CAR"},
    {"name": "PER SQ1R1", "file": "26R05MTL_260522_MAC26-01_PER_SQ_R01PARTIAL.txt",  "color": "#FF6600", "type": "CAR"},
    {"name": "PER Q1R1",  "file": "26R05MTL_260523_MAC26-01_PER_Q_R01PARTIAL.txt",   "color": "#FFD700", "type": "CAR"},
    {"name": "PER SR", "file": "26R05MTL_260523_MAC26-01_PER_SR_R02.txt",         "color": "#FF3399", "type": "CAR"},
    {"name": "PER GP", "file": "26R05MTL_260524_MAC26-01_PER_GP_R02.txt",         "color": "#CC0066", "type": "CAR"},
    # BOT (cool colours)
    {"name": "BOT P1R3", "file": "26R05MTL_260522_MAC26-03_BOT_P1_R03PARTIAL_1.txt","color": "#008CFF", "type": "CAR"},
    {"name": "BOT SQ1R1", "file": "26R05MTL_260522_MAC26-03_BOT_SQ_R01PARTIAL.txt",  "color": "#00CC88", "type": "CAR"},
    {"name": "BOT Q1R1",  "file": "26R05MTL_260523_MAC26-03_BOT_Q_R01PARTIAL.txt",   "color": "#4C00BF", "type": "CAR"},
    {"name": "BOT SR", "file": "26R05MTL_260523_MAC26-03_BOT_SR_R03.txt",         "color": "#0055AA", "type": "CAR"},
    {"name": "BOT GP", "file": "26R05MTL_260524_MAC26-03_BOT_GP_R02.txt",         "color": "#006666", "type": "CAR"},
]

WAVEFORM_DEFINITIONS = [
    WaveformPlot(
        name="[CHECK] Filtering",
        channels=('hRideF', 'hRideR', 'aSteerF', 'aRoll', 'nYaw'),
        axis_limits=(None, None, None, None, None),
        reference_lines=((0,), (0,), (0,), (0,), (0,)),
        subplot_heights=(0.6, 0.6, 0.6, 0.6, 0.6),
        # highlight_zones=('SM', '>', 0.5)
    ),
]

# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
# aggregation_mode: "per_run" (one box per run) | "aggregated" (all runs merged)
#                   "per_run_aggregated" (per-run boxes + aggregated box at end)
# gate: filter data before plotting — ('channel', 'operator', value) or list of conditions

BOX_PLOT_SETTINGS = {"show_points": False, "show_fliers": False, "title": EVENT}

# ─── Gate Dimensions ──────────────────────────────────────────────────────────
SPEED_BANDS = {
    "Low Speed": [("vCar", "<", 120)],
    "Medium Speed": [("vCar", ">", 120), ("vCar", "<", 200)],
    "High Speed": [("vCar", ">", 200)],
}

CORNER_PHASE = {
    "Entry": [("CosPhi_Calc", "between", (-0.7, -0.3))],
    "Mid":   [("CosPhi_Calc", "between", (-0.3, 0.3))],
    "Exit":  [("CosPhi_Calc", "between", (0.3, 0.7))],
}

BOX_PLOT_DEFINITIONS = [
    BoxPlotGrid(name="Typical Front Ride Heights", channels=("hRideF",), aggregation_mode="aggregated", rows=SPEED_BANDS, cols=CORNER_PHASE, render_mode="grid"),
    BoxPlotGrid(name="Typical Rear Ride Heights", channels=("hRideR",), aggregation_mode="aggregated", rows=SPEED_BANDS, cols=CORNER_PHASE, render_mode="grid"),
    BoxPlotGrid(name="Typical Yaw Rates",    channels=("nYaw",),   aggregation_mode="aggregated", rows=SPEED_BANDS, cols=CORNER_PHASE, render_mode="grid"),
    BoxPlotGrid(name="Typical Steering Angles", channels=("aSteerF",), aggregation_mode="aggregated", rows=SPEED_BANDS, cols=CORNER_PHASE, render_mode="grid"),
    BoxPlotGrid(name="Typical Roll Angles",  channels=("aRoll",),  aggregation_mode="aggregated", rows=SPEED_BANDS, cols=CORNER_PHASE, render_mode="grid"),
]

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_workflow(
        WORKFLOW_NAME,
        title="BOX PLOT ANALYSIS",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_DEFINITIONS,
        boxes=BOX_PLOT_DEFINITIONS,
        box_plot_settings=BOX_PLOT_SETTINGS,
    )
