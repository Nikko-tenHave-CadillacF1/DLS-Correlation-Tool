"""Vibration modal analysis — edit settings below and run directly.

Fits a 4-DOF body dynamics transfer function (Heave, Pitch, Roll, Warp)
to measured FPushrod Power Spectral Densities.
"""

from bootstrap import ensure_dependencies
ensure_dependencies()

from pathlib import Path
from tools.vibrations import run_fit

# ======================================================
# USER SETTINGS — edit these directly
# ======================================================

# Path to the data file (relative to project root)
DATA_FILE = r"Data\inputs\ride_dil\26R07BCN\26R07BCN_260614_MAC26-01_PER_GP_R02PARTIAL.txt"

# Sampling frequency [Hz]
FS = 100

# Vehicle parameters
TOTAL_MASS = 770.0    # Total car mass [kg]
WHEELBASE = 3.4       # Wheelbase [m]
TRACK_FRONT = 1.8     # Front track width [m]
TRACK_REAR = 1.8      # Rear track width [m]
PITCH_INERTIA = 1000.0  # Pitch inertia Ip [kg·m²]
ROLL_INERTIA = 120.0    # Roll inertia Ix [kg·m²]

# Frequency range for fitting [Hz]
F_MIN = 1.0
F_MAX = 16.0

# Welch PSD segment length (higher = finer frequency resolution, but noisier for short signals)
NPERSEG = 1024

# Show plots after fitting
SHOW_PLOTS = True

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    filepath = project_root / DATA_FILE

    run_fit(
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
        show_plots=SHOW_PLOTS,
    )
