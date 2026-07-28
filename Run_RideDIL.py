"""Ride/DIL workflow — edit RUNS and plot definitions to configure your analysis."""

from channel_config import get_workflow_dirs
from engine import (
    PsdPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)

WORKFLOW_NAME = "ride_dil"
EVENT = "26R11BUD"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────


RUNS = [
    {
        "name": "PER FP2R2",
        "file": r"26R11BUD_260724_MAC26-02_PER_P2_R02PARTIAL.txt",
        "color": "#E76000",
        "type": "CAR",
    },
    {
        "name": "FIT R18",
        "file": r"Budapest_260724_GMDiL-08_FIT_R18PARTIAL.txt",
        "color": "#005CE7",
        "type": "DIL",
    },
]

# ─── POWERPOINT EXPORT ───────────────────────────────────────────────────────────────────────
# Set POWERPOINT_OUTPUT to a Path to enable a blank 16:9 deck export.
NPERSEG = 256  # PSD segment length (samples) for Welch method. See PSD_PLOT_DEFINITIONS.
POWERPOINT_OUTPUT = None  # e.g. _OUTPUT_DIR / "DIL_Ride_Report.pptx"

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────


WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="DIL TELEM",
        channels=('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
        axis_limits=((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-180, 180), ((None, None), (None, None))),
        reference_lines=(None, None, (-350, 0, 350), None, (0,), None),
        subplot_heights=(0.15, 0.2, 0.3, 0.5, 0.3, 0.3),
    ),
    WaveformPlot(
        name="Prod Forces",
        channels=(('vCar', 'NGear'), 'FPushrodFL', 'FPushrodFR', 'FPushrodRL', 'FPushrodRR'),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
        reference_lines=(None, None, None, None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    ),
    WaveformPlot(
        name="Damper Displacements",
        channels=(('vCar', 'NGear'), 'xDamperFL', 'xDamperFR', 'xDamperRL', 'xDamperRR'),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
        reference_lines=(None, None, None, None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    ),
    WaveformPlot(
        name="Damper Variations",
        channels=(('vCar', 'NGear'), 'xDamperVarFL', 'xDamperVarFR', 'xDamperVarRL', 'xDamperVarRR'),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
        reference_lines=(None, None, None, None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    ),
    WaveformPlot(
        name="Prod Force Variations",
        channels=(('vCar', 'NGear'), 'FProdVarFL', 'FProdVarFR', 'FProdVarRL', 'FProdVarRR'),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
        reference_lines=(None, None, None, None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    ),
    WaveformPlot(
        name="Vertical Accelerations",
        channels=(('vCar', 'NGear'), 'gVert', 'gVertF', 'gVertR'),
        axis_limits=(((None, 400), (-1, 9)), None, None, None),
        reference_lines=(None, None, None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.8),
    ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
                best_fit=[('y', -7500, None), ('y', None, -9000)]),
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
    PsdPlot("Heave Mode PSD - abs",  "FPRodHeave", axis_limits=[(0, 20), (1e4, None)],                 log_scale=False, nperseg=NPERSEG), # lorentz_fit=(4, 7)
    PsdPlot("Pitch Mode PSD - abs",  "FPRodPitch", axis_limits=[(0, 20), (1e4, None)],                 log_scale=False, nperseg=NPERSEG), # lorentz_fit=(4, 7)
    PsdPlot("Roll Mode PSD - abs",   "FPRodRoll",  axis_limits=[(0, 20), (1e4, None)],       log_scale=False, nperseg=NPERSEG), # lorentz_fit=[(4, 7), (9, 12)],
    PsdPlot("Warp Mode PSD - abs",   "FPRodWarp",  axis_limits=[(0, 20), (1e4, None)],               log_scale=False, nperseg=NPERSEG),

    PsdPlot("Heave Mode PSD - gated",  "FPRodHeave", axis_limits=[(0, 20), (1e4, None)],                 log_scale=False, nperseg=NPERSEG,         gate=[('rThrottle', '<', 95)],), # lorentz_fit=(4, 7)
    PsdPlot("Pitch Mode PSD - gated",  "FPRodPitch", axis_limits=[(0, 20), (1e4, None)],                 log_scale=False, nperseg=NPERSEG, gate = [('rThrottle', '<', 95)]), # lorentz_fit=(4, 7)
    PsdPlot("Roll Mode PSD - gated",   "FPRodRoll",  axis_limits=[(0, 20), (1e4, None)],       log_scale=False, nperseg=NPERSEG, gate = [('rThrottle', '<', 95)]), # lorentz_fit=[(4, 7), (9, 12)],
    PsdPlot("Warp Mode PSD - gated",   "FPRodWarp",  axis_limits=[(0, 20), (1e4, None)],               log_scale=False, nperseg=NPERSEG, gate = [('rThrottle', '<', 95)]),


    # PsdPlot("FPushrod FL PSD - ungated",  "FPushrodFL", axis_limits=[(0, 20), (1e4, None)], annotate_at=(4, 9), log_scale=False, nperseg=NPERSEG),
    # PsdPlot("FPushrod FR PSD - ungated",  "FPushrodFR", axis_limits=[(0, 20), (1e4, None)], annotate_at=(6),    log_scale=False, nperseg=NPERSEG),
    # PsdPlot("FPushrod RL PSD - ungated",  "FPushrodRL", axis_limits=[(0, 20), (1e4, None)], annotate_at=(5, 9), log_scale=False, nperseg=NPERSEG),
    # PsdPlot("FPushrod RR PSD - ungated",  "FPushrodRR", axis_limits=[(0, 20), (1e4, None)], annotate_at=(5, 9), log_scale=False, nperseg=NPERSEG),

    # ── Vertical chassis accelerations ─────────────────────────────────────────
    PsdPlot("Front Vertical Acceleration PSD", "gVertF", axis_limits=[(0, 20), (None, None)],                            nperseg=NPERSEG), # lorentz_fit=[(5, 11)],
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR", axis_limits=[(0, 20), (None, None)],                            nperseg=NPERSEG), # lorentz_fit=[(5, 11)],
    PsdPlot("Front Vertical Acceleration PSD - ABS", "gVertF", axis_limits=[(0, 20), (None, None)],  log_scale=False, nperseg=NPERSEG), #lorentz_fit=[(5, 11)],
    PsdPlot("Rear Vertical Acceleration PSD - ABS",  "gVertR", axis_limits=[(0, 20), (None, None)],    log_scale=False, nperseg=NPERSEG), #lorentz_fit=(5, 11),

    PsdPlot("hRideF PSD", "hRideF (raw)", axis_limits=[(0, 20), (None, None)], nperseg=NPERSEG), # lorentz_fit=(3,8),
    PsdPlot("hRideR PSD", "hRideR (raw)", axis_limits=[(0, 20), (None, None)], nperseg=NPERSEG), # lorentz_fit=(4,9),

    PsdPlot("hRideF PSD - abs", "hRideF (high)", axis_limits=[(0, 20), (1e-4, None)], log_scale=False, nperseg=NPERSEG),
    PsdPlot("hRideR PSD - abs", "hRideR (high)", axis_limits=[(0, 20), (1e-4, None)],  log_scale=False, nperseg=NPERSEG),

    # PsdPlot("xDamperFL", "xDamperFL_High",  log_scale=False),
    # PsdPlot("xDamperFR", "xDamperFR_High",  log_scale=False),
    # PsdPlot("xDamperRL", "xDamperRL_High",  log_scale=False),
    # PsdPlot("xDamperRR", "xDamperRR_High",  log_scale=False),

    # PsdPlot("FPushrodFL PSD", "FPushrodFL_High",  axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    # PsdPlot("FPushrodFR PSD", "FPushrodFR_High",  axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    # PsdPlot("FPushrodRL PSD", "FPushrodRL_High",  axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    # PsdPlot("FPushrodRR PSD", "FPushrodRR_High",  axis_limits=[(0, 20), (1e4, None)], log_scale=False),
    PsdPlot("FPRodAvgF - PSD", "FPRodAvgF_High", axis_limits=[(0, 20), (1e4, None)], log_scale=False, nperseg=NPERSEG),
    PsdPlot("FPRodAvgR - PSD", "FPRodAvgR_High", axis_limits=[(0, 20), (1e4, None)], log_scale=False, nperseg=NPERSEG),
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

def main() -> None:
    run_workflow(
        WORKFLOW_NAME,
        title=f"{WORKFLOW_NAME.upper()} PLOT GENERATION",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        psds=PSD_PLOT_DEFINITIONS,
        powerpoint_output=POWERPOINT_OUTPUT,
        export_map=POWERPOINT_EXPORT_MAP,
        fig_size={"waveform": (20, 10), "default": (10, 8)},
    )


if __name__ == "__main__":
    main()
