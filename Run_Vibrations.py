"""Vibration modal analysis — edit RUNS and settings below and run directly.

Fits a 4-DOF body dynamics transfer function (Heave, Pitch, Roll, Warp)
to measured FPushrod Power Spectral Densities. Supports multiple runs
for overlaid comparison of normalised best fits.
"""

from bootstrap import ensure_dependencies
ensure_dependencies()

from pathlib import Path
from channel_config import get_workflow_dirs
from tools.vibrations import run_fit, plot_comparison

WORKFLOW_NAME = "ride_dil"
EVENT = "26P01BCN"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# ─── RUNS ─────────────────────────────────────────────────────────────────────

RUNS = [
    {
        "name": "D1-R2",
        "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R02PARTIAL.txt",
        "color": "#FF0000",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
    {
        "name": "D1-R3",
        "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R03PARTIAL.txt",
        "color": "#12B700",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
    {
        "name": "D1-R4",
        "file": r"26P01BCN_260616_MAC26-03_ZHO_D1_R04PARTIAL.txt",
        "color": "#001EFF",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
    {
        "name": "BOT R",
        "file": r"26R07BCN_260614_MAC26-03_BOT_GP_R02.txt",
        "color": "#000000",
        #"nrun": 1, # selects the run with the lowest nRun value (best lap) for each plot type
        "type": "CAR",
    },
]

# ======================================================
# USER SETTINGS
# ======================================================

# Sampling frequency [Hz]
FS = 100

# Vehicle parameters
TOTAL_MASS = None     # Total car mass [kg]
WHEELBASE = 3.4       # Wheelbase [m]
TRACK_FRONT = 1.8     # Front track width [m]
TRACK_REAR = 1.8      # Rear track width [m]
PITCH_INERTIA = None  # Pitch inertia Ip [kg·m²]
ROLL_INERTIA = None   # Roll inertia Ix [kg·m²]

# Frequency range for fitting [Hz] — body modes only (wheel-hop above ~15 Hz
# isn't captured by the 4-DOF body model, so it must not be in the fit window)
F_MIN = 2.0
F_MAX = 15.0

# Expected modal frequencies [Hz] — ranges (lo, hi) per body mode.
# Used both as a DE seed (midpoint) and as a HARD band constraint: the
# optimiser is penalised if a model mode escapes its declared range, so
# user knowledge of where each mode lives bypasses the limits of automatic
# peak detection. A plain float is also accepted (treated as f ± 15 %).
# Set EXPECTED_FREQS = None to disable both behaviours.
EXPECTED_FREQS = {
    "heave": (4.5, 6.5),
    "pitch": (8, 10.5),
    "roll":  (4, 6),
    "warp":  (9, 12),
}

# Welch PSD segment length (higher = finer frequency resolution, but noisier for short signals)
NPERSEG = 512

# Show individual run plots (set False to only show comparison)
SHOW_INDIVIDUAL_PLOTS = True
DISPLACEMENT_MODE = False  # Fit to damperpot displacements instead of pushrod forces

# Fit method:
#   "lorentzian" (default) — per-DOF sum-of-SDOF Lorentzians. Each output
#       DOF gets its own (f, zeta, A) per declared band, so asymmetric
#       front/rear damping or split mode frequencies are captured.
#   "body4dof" — original 13-parameter body MCK fit. Produces a
#       self-consistent mass/stiffness/damping model and mode shapes,
#       constrained to one (f, zeta) pair per mode shared across all DOFs.
METHOD = "lorentzian"
# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    results = []

    for run in RUNS:
        filepath = _INPUT_DIR / run["file"]
        print(f"\n{'#'*60}")
        print(f"# RUN: {run['name']}")
        print(f"{'#'*60}")

        fit_result = run_fit(
            filepath=filepath,
            fs=FS,
            track_front=TRACK_FRONT,
            track_rear=TRACK_REAR,
            fmin=F_MIN,
            fmax=F_MAX,
            nperseg=NPERSEG,
            total_mass=TOTAL_MASS,
            wheelbase=WHEELBASE,
            pitch_inertia=PITCH_INERTIA,
            roll_inertia=ROLL_INERTIA,
            show_plots=SHOW_INDIVIDUAL_PLOTS,
            output_dir=_OUTPUT_DIR,
            run_name=run["name"],
            displacement_mode=DISPLACEMENT_MODE,
            expected_freqs=EXPECTED_FREQS,
            method=METHOD,
            event=EVENT,
        )

        fit_result["name"] = run["name"]
        fit_result["color"] = run.get("color", None)
        fit_result["filepath"] = filepath
        results.append(fit_result)

    # Comparison overlay plot
    plot_comparison(
        results=results,
        fs=FS,
        track_front=TRACK_FRONT,
        track_rear=TRACK_REAR,
        fmin=F_MIN,
        fmax=F_MAX,
        nperseg=NPERSEG,
        event=EVENT,
        output_dir=_OUTPUT_DIR,
    )
