"""Vibration modal analysis — fits 4-DOF body modes (Heave/Pitch/Roll/Warp)
to FPushrod (or damperpot) PSDs across one or more runs."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs
from tools.vibrations import run_fit, plot_comparison

WORKFLOW_NAME = "ride_dil"
EVENT = "26R07BCN"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
RUNS = [
    {"name": "MCO", "file": r"26R06MCO_260607_MAC26-01_PER_GP_R02.txt", "color": "#FF0000"},
    {"name": "MIA", "file": r"26R04MIA_260503_MAC26-01_PER_GP_R02PARTIAL.txt", "color": "#1900FF"},
]

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
FS = 100                  # sampling rate [Hz]

# Fit window — body modes only; exclude wheel-hop (~15 Hz+).
F_MIN = 2.0
F_MAX = 16.0

# Expected modal bands (lo, hi) [Hz] — used to seed DE and softly bias the fit.
# A scalar is treated as ±15 %. Set EXPECTED_FREQS = None to disable.
EXPECTED_FREQS = {
    "heave": (4, 6),
    "pitch": (8, 10),
    "roll":  (4, 6),
    "warp":  (8, 12),
}

NPERSEG = 1024            # Welch segment length, or "auto"
SHOW_INDIVIDUAL_PLOTS = True
DISPLACEMENT_MODE = False # True → fit damperpot displacements instead of forces

# "lorentzian_combined" → shared (f₀, ζ) front/rear per mode, per-DOF amplitude (fast).
# "body4dof"            → 13-parameter MCK fit producing a self-consistent body model.
METHOD = "lorentzian_combined"

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for run in RUNS:
        filepath = _INPUT_DIR / run["file"]
        print(f"\n{'#'*60}\n# RUN: {run['name']}\n{'#'*60}")
        fit_result = run_fit(
            filepath=filepath, fs=FS,
            fmin=F_MIN, fmax=F_MAX, nperseg=NPERSEG,
            show_plots=SHOW_INDIVIDUAL_PLOTS, output_dir=_OUTPUT_DIR,
            run_name=run["name"], displacement_mode=DISPLACEMENT_MODE,
            expected_freqs=EXPECTED_FREQS, method=METHOD, event=EVENT,
        )
        fit_result["name"] = run["name"]
        fit_result["color"] = run.get("color")
        fit_result["filepath"] = filepath
        results.append(fit_result)

    plot_comparison(
        results=results, fs=FS,
        fmin=F_MIN, fmax=F_MAX, nperseg=NPERSEG,
        event=EVENT, output_dir=_OUTPUT_DIR,
    )
