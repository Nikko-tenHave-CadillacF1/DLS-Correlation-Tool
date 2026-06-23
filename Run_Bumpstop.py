"""Ride/DIL workflow — edit RUNS and plot definitions to configure your analysis."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot,
)

WORKFLOW_NAME = "bumpstops"
EVENT = None
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
RUNS = [
    {"folder": "4xStopChoc", "contains": "OG", "filetype": ".parquet", "nlap": 1, "type": "DLS"},
    {"folder": "4xStopChoc", "contains": "COMP", "filetype": ".parquet", "nlap": 1, "type": "DLS", "colors": ["#8000FF", "#EA00FF", "#FF00A6"]},
    # {"folder": "3xStopChoc", "contains": "MAC26", "filetype": ".txt", "type": "CAR"},
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
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
                best_fit=[('y', 11000, None)]),
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
    PsdPlot("Heave Mode PSD - ungated",  "FPRodHeave", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(4, 7)),
    PsdPlot("Pitch Mode PSD - ungated",  "FPRodPitch", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(7, 11)),
    PsdPlot("Roll Mode PSD - ungated",   "FPRodRoll",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)]),
    PsdPlot("Warp Mode PSD - ungated",   "FPRodWarp",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(20, 30)),

    PsdPlot("Heave Mode PSD - abs",  "FPRodHeave", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(4, 7),                 log_scale=False),
    PsdPlot("Pitch Mode PSD - abs",  "FPRodPitch", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(7, 11),                log_scale=False),
    PsdPlot("Roll Mode PSD - abs",   "FPRodRoll",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)],      log_scale=False),
    PsdPlot("Warp Mode PSD - abs",   "FPRodWarp",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(20, 30),               log_scale=False),

]

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
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
        powerpoint_start_slide=POWERPOINT_START_SLIDE,
        fig_size={"waveform": (20, 10), "default": (10, 8)},
    )
