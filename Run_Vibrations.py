"""Vibration modal analysis — fits 4-DOF body modes (Heave/Pitch/Roll/Warp)
to FPushrod (or damperpot) PSDs across one or more runs."""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs
from tools.vibrations import run_fit, plot_comparison

WORKFLOW_NAME = "correlation"
EVENT = "26R07BCN"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────
# `type` selects the per-source pushrod sign convention (DLS / CAR / DIL).
# Auto-detected from the filename if omitted, but explicit is safer.
RUNS = [
    {"name": "DLS", "type": "DLS", "file": r"26R07BCN - HER FP1R2 - Correlation_DLS.parquet", "color": "#0011FF"},
    {"name": "CAR", "type": "CAR", "file": r"26R07BCN_260612_MAC26-01_HER_P1_R02PARTIAL.txt", "color": "#FF6200"},
    {"name": "DIL", "type": "DIL", "file": r"Barcelona_260608_GMDiL-08_PAG_R12PARTIAL_1.txt", "color": "#00FF00"},
]

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
FS = 100                  # sampling rate [Hz]

# Fit window — body modes only; exclude wheel-hop (~15 Hz+).
F_MIN = 2.0
F_MAX = 13.0

# Expected modal bands (lo, hi) [Hz] — used to seed DE and softly bias the fit.
# A scalar is treated as ±15 %. Set EXPECTED_FREQS = None to disable.
EXPECTED_FREQS = {
    "heave": (4, 6),
    "pitch": (8, 10),
    "roll":  (4, 6),
    "warp":  (8, 11),
}

NPERSEG = "auto"          # int or "auto" (Δf scales with run length, ≥50 averages)
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
            source_type=run.get("type"),
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
