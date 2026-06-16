"""4-DOF Transfer Function Fitting to FPushrod PSDs.

Fits the parameters of a known 4-DOF body dynamics model (Heave, Pitch,
Roll, Warp) to measured FPushrod Power Spectral Densities using |H(jω)|²
shape matching.

Usage (standalone):
    python tools/vibrations.py Data/inputs/ride_dil/26R07BCN/26R07BCN_260612_MAC26-03_BOT_P2_R01.txt

Preferred usage (plug-and-play):
    Edit Run_Vibrations.py settings and run it directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import scipy.signal as signal
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel_config import RESAMPLE_RATE
from engine.logger import log
from engine.datafunctions import sanitize_numeric_series, _apply_butterworth_filter_to_data
from engine.dataplotter import DataPlotter

# ======================================================
# CONSTANTS
# ======================================================
FORCE_CHANNELS = [
    "FPushrodFL",
    "FPushrodFR",
    "FPushrodRL",
    "FPushrodRR",
]

PARAM_NAMES = [
    "mF", "mR", "Ir", "mu",
    "cFH", "cR", "cRH", "cW",
    "kFH", "kR", "kRH", "kW",
]
PARAM_UNITS = [
    "kg", "kg", "kg·m²", "kg",
    "Ns/m", "Nms/rad", "Ns/m", "Nms/rad",
    "N/m", "Nm/rad", "N/m", "Nm/rad",
]
DOF_LABELS = ["Heave Front (z_F)", "Roll Front (th_F)",
              "Heave Rear (z_R)", "Roll Rear (th_R)"]

# Initial parameter guess (based on 770 kg car, 3.4 m wheelbase, 45% front)
# Pitch inertia Ip = 1000 kg·m² → mu = Ip/L = 1000/3.4 ≈ 294 kg
# Roll inertia total = 120 kg·m² → IF + IR = 120
DEFAULT_P0 = np.array([
    300,      # mF  [kg] (45% of 770)
    500,      # mR  [kg] (55% of 770)
    25.0,       # Ir  [kg·m²] roll inertia (symmetric front/rear)
    100.0,      # mu = Ip/L [kg] (Ip = 1000 kg·m², L = 3.4 m)
    1200.0,     # cFH [Ns/m] front heave damping
    250.0,      # cR  [Nms/rad] roll damping (symmetric front/rear)
    3000.0,     # cRH [Ns/m] rear heave damping
    2000.0,     # cW  [Nms/rad] warp damping (anti-roll bar)
    200000.0,   # kFH [N/m] front heave stiffness
    20000.0,    # kR  [Nm/rad] roll stiffness (symmetric front/rear)
    600000.0,   # kRH [N/m] rear heave stiffness
    300000.0,   # kW  [Nm/rad] warp (anti-roll) stiffness
])

# Parameter bounds for optimisation [lower, upper] in physical units
# mF/mR bounds enforce 44-46% front weight distribution (770 kg total)
# IF/IR bounds: total roll inertia ~120 kg·m²
# mu bounds: Ip = 1000 kg·m², L = 3.4 m → mu = 294 kg (allow ±20%)
# Stiffness bounds keep all modes within 1-19 Hz fitting range
# Damping bounds ensure underdamped peaks (ζ < ~0.5)
BOUNDS_PHYSICAL = np.array([
    [10,    800],       # mF
    [10,    800],       # mR
    [5,    50],         # Ir  (roll inertia, symmetric)
    [10,    500],       # mu = Ip/L
    [500,   5000],     # cFH (heave damping)
    [100,    1000],      # cR  (roll damping, symmetric)
    [1000,   10000],     # cRH (heave damping)
    [500,    5000],     # cW  (warp damping)
    [10000,  500000],   # kFH (heave stiffness)
    [5000,   50000],   # kR  (roll stiffness, symmetric)
    [100000, 800000],   # kRH (heave stiffness)
    [100000, 1000000],  # kW  (warp coupling)
])


# ======================================================
# 1. DATA LOADING
# ======================================================

def load_force_data(filepath: Path, fs: float = RESAMPLE_RATE) -> np.ndarray:
    """Load FPushrod corner forces from CSV and apply 2.5 Hz high-pass filter.

    Uses engine utilities for numeric sanitization and Butterworth filtering.

    Corrects sign convention: front pushrods are negative by convention,
    rear are positive. Negate rear channels so all corners share the same
    sign direction (positive = compression into chassis).
    """
    df = pd.read_csv(filepath, sep=",", skiprows=[0, 2], header=0, low_memory=False)

    # Sanitize using engine helper
    for ch in FORCE_CHANNELS:
        df[ch] = sanitize_numeric_series(df[ch])

    # Linear interpolation (cap at 100 consecutive NaN samples)
    df[FORCE_CHANNELS] = df[FORCE_CHANNELS].interpolate(
        method="linear", limit=100, axis=0
    )

    # Drop rows that still have NaN after interpolation
    df = df.dropna(subset=FORCE_CHANNELS)

    # Apply 2.5 Hz high-pass filter using engine utility
    for ch in FORCE_CHANNELS:
        filtered, success = _apply_butterworth_filter_to_data(
            df[ch].values, cutoff=2, order=2, sample_rate=fs, btype="high"
        )
        if success:
            df[ch] = filtered
        else:
            log.warning("High-pass filter failed for channel '%s'.", ch)

    F_corner = df[FORCE_CHANNELS].astype(float).values.T  # [4 x N]

    # Flip rear pushrod sign to match front convention
    F_corner[2] *= -1  # RL
    F_corner[3] *= -1  # RR

    return F_corner



# ======================================================
# 2. CORNER-TO-BODY TRANSFORMATION
# ======================================================
def build_T(track_front: float, track_rear: float) -> np.ndarray:
    """Transformation matrix: corner → body coordinates.

    Body DOFs: [z_F, θ_F, z_R, θ_R]
    Corner order: [FL, FR, RL, RR]
    """
    return np.array([
        [0.5,          0.5,           0,             0            ],
        [1/track_front, -1/track_front, 0,           0            ],
        [0,            0,             0.5,           0.5          ],
        [0,            0,             1/track_rear, -1/track_rear ],
    ])


# ======================================================
# 3. PSD COMPUTATION
# ======================================================
def compute_body_psds(F_corner: np.ndarray, T: np.ndarray,
                      fs: float, nperseg: int = 1024):
    """Transform corner forces to body coordinates and compute PSDs."""
    F_body = T @ F_corner
    F_body = signal.detrend(F_body, axis=1)
    freqs, psds = signal.welch(F_body, fs, nperseg=nperseg, axis=1)
    return freqs, psds


# ======================================================
# 4. MODEL DEFINITION: M, C, K MATRICES
# ======================================================
def build_MCK(params: np.ndarray):
    """Construct M, C, K matrices from the 14-element parameter vector.

    Mass matrix (with pitch inertia coupling via mu):
        The off-diagonal coupling is POSITIVE for a typical race car where
        Ip < m*a*b (dynamic index < 1). This ensures heave mode has higher
        effective mass than pitch mode, giving ω_heave < ω_pitch.

        M = [[mF-mu,  0,    +mu,   0 ],
             [0,      Ir,    0,    0 ],
             [+mu,    0,    mR-mu, 0 ],
             [0,      0,     0,    Ir]]

    Damping matrix (with symmetric roll damping cR and warp coupling cW):
        C = [[cFH,    0,       0,     0      ],
             [0,      cR+cW,   0,    -cW     ],
             [0,      0,       cRH,   0      ],
             [0,     -cW,      0,     cR+cW  ]]

    Stiffness matrix (with symmetric roll stiffness kR and warp coupling kW):
        K = [[kFH,  0,       0,    0      ],
             [0,    kR+kW,   0,   -kW     ],
             [0,    0,       kRH,  0      ],
             [0,   -kW,      0,    kR+kW  ]]
    """
    mF, mR, Ir, mu = params[0:4]
    cFH, cR, cRH, cW = params[4:8]
    kFH, kR, kRH, kW = params[8:12]

    M = np.array([
        [mF - mu,  0,      +mu,      0   ],
        [0,        Ir,     0,       0   ],
        [+mu,       0,      mR - mu, 0   ],
        [0,        0,      0,       Ir  ],
    ])

    C = np.array([
        [cFH,  0,         0,    0       ],
        [0,    cR + cW,   0,   -cW      ],
        [0,    0,         cRH,  0       ],
        [0,   -cW,        0,    cR + cW ],
    ])

    K = np.array([
        [kFH,  0,         0,    0       ],
        [0,    kR + kW,   0,   -kW      ],
        [0,    0,         kRH,  0       ],
        [0,   -kW,        0,    kR + kW ],
    ])

    return M, C, K


# ======================================================
# 5. TRANSFER FUNCTION |H(jω)|²
# ======================================================
def compute_H_mag_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute |H(jω)|² at each frequency.

    H(jω) = (-ω²M + jωC + K)⁻¹  (receptance matrix)

    Returns: [4, nf] array — sum of |H_ij|² over input DOFs for each output DOF i.
    """
    nf = len(freqs_hz)
    H_sq = np.zeros((4, nf))

    for k, f in enumerate(freqs_hz):
        omega = 2.0 * np.pi * f
        Z = K - omega**2 * M + 1j * omega * C
        try:
            H = np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            H_sq[:, k] = 1e-30
            continue
        H_sq[:, k] = np.sum(np.abs(H)**2, axis=1)

    return H_sq


# ======================================================
# 6. COST FUNCTION
# ======================================================
def _normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise a 1-D array to [0, 1] by its maximum."""
    peak = np.max(arr)
    return arr / peak if peak > 0 else arr


def _apply_roll_weighting(H_sq: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """Apply w^4 weighting to roll DOFs (1, 3) for force PSD shape matching.

    The receptance |H(jw)|^2 peaks scale as 1/w^4, making low-frequency
    modes dominate after normalisation.  Multiplying by w^4 converts to an
    accelerance-equivalent whose peaks are frequency-independent (proportional
    only to 1/zeta^2), matching measured force PSD shapes under the flat road
    acceleration assumption typical for race circuits.
    """
    H_weighted = H_sq.copy()
    omega4 = (2.0 * np.pi * freqs_hz) ** 4
    H_weighted[1] *= omega4
    H_weighted[3] *= omega4
    return H_weighted


def cost_function(log_params: np.ndarray, freqs_fit: np.ndarray,
                  measured_psds_fit: np.ndarray,
                  total_mass: float = None, wheelbase: float = None,
                  pitch_inertia: float = None, roll_inertia: float = None) -> float:
    """Shape-matching cost: normalised |H(jω)|² vs normalised measured PSDs.

    Both model and measurement are normalised per-DOF to [0, 1] so that only
    the spectral shape matters — absolute magnitudes are irrelevant. This
    preserves all natural frequency and damping information.

    Optional penalty constraints:
      - mF + mR ≈ total_mass
      - Front weight distribution 44-46%
      - mu ≈ pitch_inertia / wheelbase
      - IF + IR ≈ roll_inertia
    """
    params = np.exp(log_params)
    M, C, K = build_MCK(params)

    try:
        eigvals = np.linalg.eigvalsh(M)
        if np.any(eigvals <= 0):
            return 1e15
    except np.linalg.LinAlgError:
        return 1e15

    H_sq = compute_H_mag_sq(freqs_fit, M, C, K)

    # Apply w^4 weighting to roll DOFs for balanced peak fitting
    H_sq = _apply_roll_weighting(H_sq, freqs_fit)

    # Normalised shape comparison (sum of squared differences per DOF)
    total_cost = 0.0
    for dof in range(4):
        total_cost += np.sum((_normalise(measured_psds_fit[dof]) - _normalise(H_sq[dof]))**2)

    nf = len(freqs_fit)

    # Hard constraint: total mass (mF + mR must equal total_mass)
    if total_mass is not None:
        mF, mR = params[0], params[1]
        mass_error = (mF + mR - total_mass) / total_mass
        total_cost += 500.0 * nf * mass_error**2

        # Hard constraint: front weight distribution (44-46%)
        front_pct = mF / total_mass
        if front_pct < 0.44 or front_pct > 0.46:
            dist_error = front_pct - 0.45
            total_cost += 500.0 * nf * dist_error**2

    # Hard constraint: pitch inertia (Ip = mu * L)
    if pitch_inertia is not None and wheelbase is not None:
        mu = params[3]
        Ip_model = mu * wheelbase
        Ip_error = (Ip_model - pitch_inertia) / pitch_inertia
        total_cost += 500.0 * nf * Ip_error**2

    # Hard constraint: total roll inertia (2 * Ir)
    if roll_inertia is not None:
        Ir = params[2]
        roll_error = (2 * Ir - roll_inertia) / roll_inertia
        total_cost += 500.0 * nf * roll_error**2

    return total_cost


# --- Decoupled sub-system cost functions ---
# Heave/Pitch indices in full param vector: mF(0), mR(1), mu(3), cFH(4), cRH(6), kFH(8), kRH(10)
_HEAVE_PITCH_IDX = [0, 1, 3, 4, 6, 8, 10]
# Roll/Warp indices: Ir(2), cR(5), cW(7), kR(9), kW(11)
_ROLL_WARP_IDX = [2, 5, 7, 9, 11]


def _cost_heave_pitch(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                      measured_psds_fit: np.ndarray,
                      total_mass: float = None, wheelbase: float = None,
                      pitch_inertia: float = None) -> float:
    """Cost function for heave/pitch subsystem only (DOFs 0, 2)."""
    sub_params = np.exp(log_sub_params)
    mF, mR, mu, cFH, cRH, kFH, kRH = sub_params

    # 2x2 subsystem matrices
    M_hp = np.array([[mF - mu, mu], [mu, mR - mu]])
    C_hp = np.diag([cFH, cRH])
    K_hp = np.diag([kFH, kRH])

    try:
        eigvals = np.linalg.eigvalsh(M_hp)
        if np.any(eigvals <= 0):
            return 1e15
    except np.linalg.LinAlgError:
        return 1e15

    nf = len(freqs_fit)
    H_sq = np.zeros((2, nf))
    for k, f in enumerate(freqs_fit):
        omega = 2.0 * np.pi * f
        Z = K_hp - omega**2 * M_hp + 1j * omega * C_hp
        try:
            H = np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            H_sq[:, k] = 1e-30
            continue
        H_sq[:, k] = np.sum(np.abs(H)**2, axis=1)

    # Compare DOFs 0 (z_F) and 2 (z_R) from measured data
    cost = 0.0
    cost += np.sum((_normalise(measured_psds_fit[0]) - _normalise(H_sq[0]))**2)
    cost += np.sum((_normalise(measured_psds_fit[2]) - _normalise(H_sq[1]))**2)

    # Optional constraints
    if total_mass is not None:
        mass_error = (mF + mR - total_mass) / total_mass
        cost += 500.0 * nf * mass_error**2
        front_pct = mF / total_mass
        if front_pct < 0.44 or front_pct > 0.46:
            cost += 500.0 * nf * (front_pct - 0.45)**2

    if pitch_inertia is not None and wheelbase is not None:
        Ip_model = mu * wheelbase
        cost += 500.0 * nf * ((Ip_model - pitch_inertia) / pitch_inertia)**2

    return cost


def _cost_roll_warp(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                    measured_psds_fit: np.ndarray,
                    roll_inertia: float = None) -> float:
    """Cost function for roll/warp subsystem only (DOFs 1, 3)."""
    sub_params = np.exp(log_sub_params)
    Ir, cR, cW, kR, kW = sub_params

    # 2x2 subsystem matrices (fully symmetric M, C and K)
    M_rw = np.diag([Ir, Ir])
    C_rw = np.array([[cR + cW, -cW], [-cW, cR + cW]])
    K_rw = np.array([[kR + kW, -kW], [-kW, kR + kW]])

    nf = len(freqs_fit)
    H_sq = np.zeros((2, nf))
    for k, f in enumerate(freqs_fit):
        omega = 2.0 * np.pi * f
        Z = K_rw - omega**2 * M_rw + 1j * omega * C_rw
        try:
            H = np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            H_sq[:, k] = 1e-30
            continue
        H_sq[:, k] = np.sum(np.abs(H)**2, axis=1)

    # Apply w^4 weighting (accelerance) so roll and warp peaks are balanced
    omega4 = (2.0 * np.pi * freqs_fit) ** 4
    H_sq[0] *= omega4
    H_sq[1] *= omega4

    # Compare DOFs 1 (θ_F) and 3 (θ_R) from measured data
    cost = 0.0
    cost += np.sum((_normalise(measured_psds_fit[1]) - _normalise(H_sq[0]))**2)
    cost += np.sum((_normalise(measured_psds_fit[3]) - _normalise(H_sq[1]))**2)

    # Optional constraint
    if roll_inertia is not None:
        roll_error = (2 * Ir - roll_inertia) / roll_inertia
        cost += 500.0 * nf * roll_error**2

    return cost


# ======================================================
# 7. MODAL ANALYSIS
# ======================================================
def extract_modes(M: np.ndarray, C: np.ndarray, K: np.ndarray):
    """Extract natural frequencies, damping ratios, and mode shapes.

    Uses state-space eigenvalue decomposition:
        A = [[0, I], [-M⁻¹K, -M⁻¹C]]
    """
    n = M.shape[0]
    M_inv = np.linalg.inv(M)
    A = np.zeros((2*n, 2*n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -M_inv @ K
    A[n:, n:] = -M_inv @ C

    eigvals, eigvecs = np.linalg.eig(A)

    mask = eigvals.imag > 0
    lam = eigvals[mask]
    vecs = eigvecs[:n, mask]

    order = np.argsort(np.abs(lam))
    lam = lam[order]
    vecs = vecs[:, order]

    omega_n = np.abs(lam)
    fn = omega_n / (2.0 * np.pi)
    zeta = -np.real(lam) / omega_n

    return fn, zeta, vecs


def classify_mode(shape: np.ndarray) -> str:
    """Classify mode shape [z_F, θ_F, z_R, θ_R] as Heave/Pitch/Roll/Warp.

    Uses complex phase angle between DOF pairs for robust classification
    even with non-proportional damping (complex eigenvectors).
    """
    heave_content = np.abs(shape[0]) + np.abs(shape[2])
    roll_content = np.abs(shape[1]) + np.abs(shape[3])

    if heave_content > roll_content:
        # Heave vs Pitch: check relative phase of z_F and z_R
        phase_diff = np.angle(shape[0] * np.conj(shape[2]))
        return "Heave" if abs(phase_diff) < np.pi / 2 else "Pitch"
    else:
        # Roll vs Warp: check relative phase of θ_F and θ_R
        phase_diff = np.angle(shape[1] * np.conj(shape[3]))
        return "Roll" if abs(phase_diff) < np.pi / 2 else "Warp"


# ======================================================
# 8. PLOTTING (uses engine style constants)
# ======================================================

_PLOT_FONT = DataPlotter.PLOT_FONT
_GRID_MAJOR = DataPlotter.GRID_STYLE["major"]
_GRID_MINOR = DataPlotter.GRID_STYLE["minor"]
_INK = "#1A1A1A"


def _configure_style():
    """Apply plot style matching the rest of the codebase."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = _PLOT_FONT["family"] if _PLOT_FONT["family"] in available else _PLOT_FONT["fallback"][0]

    plt.rcParams.update({
        "font.family": font,
        "font.sans-serif": [_PLOT_FONT["family"]] + _PLOT_FONT["fallback"],
        "axes.titlesize": _PLOT_FONT["title_size"],
        "axes.titleweight": "bold",
        "axes.labelsize": _PLOT_FONT["label_size"],
        "axes.labelweight": "bold",
        "axes.edgecolor": _INK,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "xtick.labelsize": _PLOT_FONT["tick_size"],
        "ytick.labelsize": _PLOT_FONT["tick_size"],
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "text.color": _INK,
        "legend.fontsize": _PLOT_FONT["legend_size"],
        "figure.titlesize": _PLOT_FONT["figure_title_size"],
        "figure.titleweight": "bold",
    })


def _style_axis(ax, grid_axis="both"):
    """Apply standard grid + spine styling to an axis."""
    ax.grid(True, which="major", axis=grid_axis, **_GRID_MAJOR)
    ax.grid(True, which="minor", axis=grid_axis, **_GRID_MINOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_legend(ax, loc="upper right"):
    """Add a consistently styled legend."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
        loc=loc,
        fancybox=True, framealpha=0.92, edgecolor="#3C3C3C",
        borderpad=0.55, handlelength=1.8,
        prop={"family": _PLOT_FONT["family"], "weight": "bold", "size": _PLOT_FONT["legend_size"]},
    )
    legend.get_frame().set_linewidth(1.4)


def generate_plots(freqs, psds, freqs_fit, psds_fit, M, C, K, T,
                   fn, zeta, mode_shapes, fmin, fmax,
                   output_dir: Path = None):
    """Generate all diagnostic plots and save to output_dir/plots/."""
    _configure_style()

    plots_dir = (output_dir / "plots" / "vibrations") if output_dir else Path(".") / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_modes = len(fn)
    H_sq_fit = compute_H_mag_sq(freqs_fit, M, C, K)

    # Apply w^4 weighting to roll DOFs for balanced peak fitting
    H_sq_fit = _apply_roll_weighting(H_sq_fit, freqs_fit)

    # PLOT 1: Measured vs Fitted PSDs
    fig1, axes1 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for dof in range(4):
        ax = axes1[dof]
        ax.plot(freqs_fit, _normalise(psds_fit[dof]), color="#2000BF",
                linewidth=1.8, alpha=0.85, label="Measured")
        ax.plot(freqs_fit, _normalise(H_sq_fit[dof]), color="#D70000",
                linewidth=1.8, alpha=0.85, label="Fitted |H|²")
        for f_n in fn:
            if fmin <= f_n <= fmax:
                ax.axvline(f_n, color="#00AA55", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(norm.)",
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax)
    axes1[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout(pad=0.3, h_pad=0.0)
    fig1.savefig(plots_dir / "vibrations_fit.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
    plt.close(fig1)

    # PLOT 2: Full-band PSDs with mode frequencies
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for dof in range(4):
        ax = axes2[dof]
        valid = freqs > 0.5
        ax.semilogy(freqs[valid], psds[dof, valid], color="#2000BF", linewidth=1.6, alpha=0.85)
        for i, f_n in enumerate(fn):
            mode_label = classify_mode(mode_shapes[:, i])
            ax.axvline(f_n, color="#D70000", linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"{mode_label} {f_n:.2f} Hz" if dof == 0 else "")
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(PSD)",
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax)
    axes2[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout(pad=0.3, h_pad=0.0)
    fig2.savefig(plots_dir / "vibrations_psd_modes.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
    plt.close(fig2)

    # PLOT 3: Mode shapes (body coordinates)
    if n_modes > 0:
        fig3, axes3 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4), sharey=True)
        if n_modes == 1:
            axes3 = [axes3]

        dof_short = ["z_F", "th_F", "z_R", "th_R"]
        for i in range(n_modes):
            shape = np.real(mode_shapes[:, i])
            shape = shape / np.max(np.abs(shape))
            axes3[i].barh(range(4), shape,
                          color=["#2E86AB" if s >= 0 else "#E05263" for s in shape],
                          edgecolor=_INK, linewidth=0.5)
            axes3[i].set_yticks(range(4))
            axes3[i].set_yticklabels(dof_short)
            mode_name = classify_mode(mode_shapes[:, i])
            axes3[i].set_title(f"{mode_name}\n{fn[i]:.2f} Hz  z={zeta[i]:.4f}", fontsize=10)
            axes3[i].axvline(0, color=_INK, linewidth=0.6)
            axes3[i].set_xlim(-1.2, 1.2)
            _style_axis(axes3[i], grid_axis="x")
        plt.tight_layout(pad=0.25)
        fig3.savefig(plots_dir / "vibrations_mode_shapes_body.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
        plt.close(fig3)

    # PLOT 4: Transfer function magnitude (normalised)
    freqs_plot = np.linspace(0.5, fmax + 2.0, 500)
    H_sq_full = compute_H_mag_sq(freqs_plot, M, C, K)
    fig4, axes4 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for dof in range(4):
        ax = axes4[dof]
        ax.plot(freqs_plot, _normalise(H_sq_full[dof]), color="#D70000", linewidth=1.8, alpha=0.85)
        for f_n in fn:
            ax.axvline(f_n, color="#00AA55", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(norm.)",
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
    axes4[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout(pad=0.3, h_pad=0.0)
    fig4.savefig(plots_dir / "vibrations_transfer_function.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
    plt.close(fig4)

    # PLOT 5: Mode shapes (corner coordinates)
    if n_modes > 0:
        T_inv = np.linalg.inv(T)
        fig5, axes5 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4), sharey=True)
        if n_modes == 1:
            axes5 = [axes5]

        corner_labels = ["FL", "FR", "RL", "RR"]
        for i in range(n_modes):
            body_shape = np.real(mode_shapes[:, i])
            corner_shape = T_inv @ body_shape
            corner_shape = corner_shape / np.max(np.abs(corner_shape))
            axes5[i].barh(range(4), corner_shape,
                          color=["#2E86AB" if s >= 0 else "#E05263" for s in corner_shape],
                          edgecolor=_INK, linewidth=0.5)
            axes5[i].set_yticks(range(4))
            axes5[i].set_yticklabels(corner_labels)
            mode_name = classify_mode(mode_shapes[:, i])
            axes5[i].set_title(f"{mode_name}\n{fn[i]:.2f} Hz", fontsize=10)
            axes5[i].axvline(0, color=_INK, linewidth=0.6)
            axes5[i].set_xlim(-1.2, 1.2)
            _style_axis(axes5[i], grid_axis="x")
        plt.tight_layout(pad=0.25)
        fig5.savefig(plots_dir / "vibrations_mode_shapes_corner.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
        plt.close(fig5)

    log.info("  Plots saved to: %s", plots_dir)


def _generate_diagnosis_plot(freqs_fit, psds_fit, M, C, K, fn, zeta, mode_shapes,
                             fmin, fmax, run_name, output_dir):
    """Generate a single per-run diagnosis figure evaluating fit quality.

    Shows measured vs fitted normalised PSDs for all 4 DOFs with mode
    frequencies, residual error, and identified mode annotations.
    Saved as vibrations_diag_<run_name>.png.
    """
    _configure_style()
    plots_dir = (output_dir / "plots" / "vibrations") if output_dir else Path(".") / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)

    H_sq_fit = compute_H_mag_sq(freqs_fit, M, C, K)
    H_sq_fit = _apply_roll_weighting(H_sq_fit, freqs_fit)
    n_modes = len(fn)

    # 4 DOF rows + 1 residual row
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5]})


    # Per-DOF: measured vs fitted
    for dof in range(4):
        ax = axes[dof]
        meas = _normalise(psds_fit[dof])
        model = _normalise(H_sq_fit[dof])
        ax.plot(freqs_fit, meas, color="#2000BF", linewidth=1.6, alpha=0.85, label="Measured")
        ax.plot(freqs_fit, model, color="#D70000", linewidth=1.6, alpha=0.85, label="Fitted")
        ax.fill_between(freqs_fit, meas, model, color="#D70000", alpha=0.12)
        for i, f_n in enumerate(fn):
            if fmin <= f_n <= fmax:
                mode_label = classify_mode(mode_shapes[:, i])
                ax.axvline(f_n, color="#00AA55", linestyle="--", linewidth=0.9, alpha=0.7,
                           label=f"{mode_label} {f_n:.1f} Hz" if dof == 0 else "")
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(norm.)",
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax)

    # Residual row: per-frequency sum-of-squared-error across DOFs
    ax_res = axes[4]
    residual = np.zeros_like(freqs_fit)
    for dof in range(4):
        residual += (_normalise(psds_fit[dof]) - _normalise(H_sq_fit[dof]))**2
    ax_res.fill_between(freqs_fit, 0, residual, color="#D70000", alpha=0.3)
    ax_res.plot(freqs_fit, residual, color="#D70000", linewidth=1.0, alpha=0.7)
    ax_res.set_ylabel("Residual\n(SSE)",
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
    ax_res.yaxis.set_label_coords(-0.035, 0.5)
    ax_res.set_ylim(bottom=0)
    _style_axis(ax_res, grid_axis="y")

    axes[-1].set_xlabel("Frequency [Hz]")

    # Annotation box with mode summary
    mode_text_lines = []
    for i in range(n_modes):
        mtype = classify_mode(mode_shapes[:, i])
        z_str = f"z={zeta[i]:.3f}" if not np.isnan(zeta[i]) else ""
        mode_text_lines.append(f"{mtype}: {fn[i]:.2f} Hz  {z_str}")
    if mode_text_lines:
        mode_text = "\n".join(mode_text_lines)
        axes[0].text(0.98, 0.95, mode_text, transform=axes[0].transAxes,
                     fontsize=8.5, fontweight="bold", va="top", ha="right",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor="#3C3C3C", alpha=0.9))

    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.95))
    safe_name = run_name.replace(" ", "_").replace("/", "-")
    fig.savefig(plots_dir / f"vibrations_diag_{safe_name}.png",
                dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("  Diagnosis plot saved: vibrations_diag_%s.png", safe_name)


# ======================================================
# 9. MAIN PIPELINE
# ======================================================
def run_fit(filepath: Path, fs: float = RESAMPLE_RATE, track_front: float = 1.8,
            track_rear: float = 1.8, fmin: float = 1.0, fmax: float = 19.0,
            nperseg: int = 1024, total_mass: float = None,
            wheelbase: float = None, pitch_inertia: float = None,
            roll_inertia: float = None, show_plots: bool = True,
            output_dir: Path = None, run_name: str = None):
    """Run the full 4-DOF transfer function fitting pipeline.

    Fits a normalised |H(jω)|² shape to normalised measured PSDs per body DOF.
    Absolute magnitudes are irrelevant — only spectral shape is matched,
    which preserves all natural frequency and damping ratio information.

    Always generates a per-run diagnosis plot to output_dir/plots/ for
    evaluating fit quality.

    Returns: (params_fit, fn, zeta, mode_shapes)
    """
    label = run_name or filepath.stem
    log.info("Loading: %s", filepath.name)
    F_corner = load_force_data(filepath, fs)
    n_samples = F_corner.shape[1]
    log.info("  %d samples (%.1f s), fit range %.1f–%.1f Hz", n_samples, n_samples/fs, fmin, fmax)

    T = build_T(track_front, track_rear)
    freqs, psds = compute_body_psds(F_corner, T, fs, nperseg=nperseg)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[freq_mask]
    psds_fit = psds[:, freq_mask]

    # Optimisation in log-space — decoupled into two independent subsystems
    bounds_log = np.log(BOUNDS_PHYSICAL)

    # Heave/Pitch subsystem (7 params): mF, mR, mu, cFH, cRH, kFH, kRH
    hp_bounds = bounds_log[_HEAVE_PITCH_IDX]
    log.info("  Optimising heave/pitch subsystem (7 params)...")
    result_hp = differential_evolution(
        _cost_heave_pitch,
        bounds=list(zip(hp_bounds[:, 0], hp_bounds[:, 1])),
        args=(freqs_fit, psds_fit, total_mass, wheelbase, pitch_inertia),
        seed=42,
        popsize=40,
        polish=True,
        disp=False,
        updating="deferred",
        workers=-1,
    )
    log.info("  Heave/pitch done (cost=%.4f)", result_hp.fun)

    # Roll/Warp subsystem (5 params): Ir, cR, cW, kR, kW
    rw_bounds = bounds_log[_ROLL_WARP_IDX]
    log.info("  Optimising roll/warp subsystem (5 params)...")
    result_rw = differential_evolution(
        _cost_roll_warp,
        bounds=list(zip(rw_bounds[:, 0], rw_bounds[:, 1])),
        args=(freqs_fit, psds_fit, roll_inertia),
        seed=42,
        popsize=40,
        polish=True,
        disp=False,
        updating="deferred",
        workers=-1,
    )
    log.info("  Roll/warp done (cost=%.4f)", result_rw.fun)

    # Reassemble full 12-element parameter vector
    params_fit = np.zeros(12)
    for i, idx in enumerate(_HEAVE_PITCH_IDX):
        params_fit[idx] = np.exp(result_hp.x[i])
    for i, idx in enumerate(_ROLL_WARP_IDX):
        params_fit[idx] = np.exp(result_rw.x[i])
    M, C, K = build_MCK(params_fit)
    fn, zeta, mode_shapes = extract_modes(M, C, K)

    # Compact results output
    log.info("  %-8s %-12s %-10s | %-6s %-12s %s", "Mode", "Freq [Hz]", "Damp", "Param", "Value", "Unit")
    for i in range(max(len(fn), len(PARAM_NAMES))):
        mode_str = ""
        if i < len(fn):
            mtype = classify_mode(mode_shapes[:, i])
            z_str = f"{zeta[i]:.4f}" if not np.isnan(zeta[i]) else "N/A"
            mode_str = f"  {mtype:<8} {fn[i]:<12.3f} {z_str:<10}"
        else:
            mode_str = f"  {'':8} {'':12} {'':10}"
        param_str = ""
        if i < len(PARAM_NAMES):
            param_str = f" | {PARAM_NAMES[i]:<6} {params_fit[i]:<12.1f} {PARAM_UNITS[i]}"
        log.info(mode_str + param_str)

    # Always generate per-run diagnosis plot
    _generate_diagnosis_plot(
        freqs_fit, psds_fit, M, C, K, fn, zeta, mode_shapes,
        fmin, fmax, label, output_dir,
    )

    if show_plots:
        generate_plots(freqs, psds, freqs_fit, psds_fit, M, C, K, T,
                       fn, zeta, mode_shapes, fmin, fmax,
                       output_dir=output_dir)

    return params_fit, fn, zeta, mode_shapes


# ======================================================
# CLI (standalone fallback — prefer Run_Vibrations.py)
# ======================================================

# Default colour cycle for comparison plots
_DEFAULT_COLORS = [
    "#FF8000", "#2000BF", "#D70000", "#008CFF",
    "#00CC88", "#CC0066", "#FFD700", "#4C00BF",
]


def plot_comparison(results: list, fs: float = 100.0,
                    track_front: float = 1.8, track_rear: float = 1.8,
                    fmin: float = 1.0, fmax: float = 19.0,
                    nperseg: int = 1024, event: str = "",
                    output_dir: Path = None):
    """Overlay normalised best-fit |H(jw)|^2 for multiple runs.

    Args:
        results: List of dicts with keys: name, color, params, fn, zeta,
                 mode_shapes, filepath.
        fs, track_front, track_rear, fmin, fmax, nperseg: shared settings.
        event: Event name for plot title.
        output_dir: Directory to save plots into (a plots/ subfolder is created).
    """
    _configure_style()

    if not results:
        log.warning("No results to compare.")
        return

    plots_dir = (output_dir / "plots" / "vibrations") if output_dir else Path(".") / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)

    freqs_plot = np.linspace(fmin, fmax, 500)
    T = build_T(track_front, track_rear)

    # --- PLOT A: Fitted |H(jw)|^2 comparison ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        M, C, K = build_MCK(res["params"])
        H_sq = compute_H_mag_sq(freqs_plot, M, C, K)

        for dof in range(4):
            ax = axes[dof]
            ax.plot(freqs_plot, _normalise(H_sq[dof]), color=color,
                    linewidth=1.8, alpha=0.85, label=res["name"])

    for dof in range(4):
        ax = axes[dof]
        ax.set_ylabel(DOF_LABELS[dof],
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
    axes[-1].set_xlabel("Frequency [Hz]")

    # --- Mode info box (figure-level, overlaps subplots as needed) ---
    # Group by mode type for easy cross-run comparison; colour identifies run.
    from collections import OrderedDict
    mode_groups = OrderedDict()  # {mode_type: [(fn, zeta, color, run_name), ...]}
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        for i in range(len(res["fn"])):
            f_n = res["fn"][i]
            if fmin <= f_n <= fmax:
                mtype = classify_mode(res["mode_shapes"][:, i])
                mode_groups.setdefault(mtype, []).append((f_n, res["zeta"][i], color, res["name"]))

    if mode_groups:
        legend_fs = _PLOT_FONT["legend_size"]
        # Scale font if many entries would overflow
        n_entries = sum(1 + len(v) for v in mode_groups.values())
        fs = legend_fs if n_entries <= 12 else (legend_fs - 1 if n_entries <= 16 else legend_fs - 2)

        # Find max run name length for aligned columns
        max_name = max(len(rname) for entries in mode_groups.values()
                       for _, _, _, rname in entries)

        info_lines = []
        for mtype, entries in mode_groups.items():
            info_lines.append((f"{mtype}", _INK))
            for f_n, z_val, color, rname in entries:
                z_str = f"{z_val:.3f}" if not np.isnan(z_val) else "  n/a"
                padded = rname.ljust(max_name)
                info_lines.append((f"  {padded}  f={f_n:.2f} Hz  z={z_str}", color))

        text_areas = [
            TextArea(
                text,
                textprops=dict(
                    color=color,
                    fontsize=fs,
                    fontweight="bold",
                    family="monospace",
                ),
            )
            for text, color in info_lines
        ]
        vpacker = VPacker(children=text_areas, pad=6, sep=2)
        ab = AnnotationBbox(
            vpacker,
            xy=(0.99, 0.98),
            xycoords="figure fraction",
            box_alignment=(1.0, 1.0),
            bboxprops=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.92,
                edgecolor="#3C3C3C",
                linewidth=1.4,
            ),
            frameon=True,
            pad=0,
        )
        ab.set_zorder(10)
        fig.add_artist(ab)
    plt.tight_layout(pad=0.3, h_pad=0.0)
    fig.savefig(plots_dir / "vibrations_comparison_fit.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # --- PLOT B: Measured PSD comparison ---
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        F_corner = load_force_data(res["filepath"], fs)
        freqs, psds = compute_body_psds(F_corner, T, fs, nperseg=nperseg)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        for dof in range(4):
            ax = axes2[dof]
            ax.plot(freqs[freq_mask], _normalise(psds[dof, freq_mask]),
                    color=color, linewidth=1.6, alpha=0.85, label=res["name"])

    for dof in range(4):
        ax = axes2[dof]
        ax.set_ylabel(DOF_LABELS[dof],
                      fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax)
    axes2[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout(pad=0.3, h_pad=0.0)
    fig2.savefig(plots_dir / "vibrations_comparison_psd.png", dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
    plt.close(fig2)

    # --- Summary table ---
    log.info("  %-20s %-12s %-12s %-12s %-12s", "Run", "Heave [Hz]", "Pitch [Hz]", "Roll [Hz]", "Warp [Hz]")
    for res in results:
        fn = res["fn"]
        ms = res["mode_shapes"]
        modes = [classify_mode(ms[:, i]) for i in range(len(fn))]
        def find_freq(label):
            for i, m in enumerate(modes):
                if m == label:
                    return f"{fn[i]:.2f}"
            return "\u2014"
        log.info("  %-20s %-12s %-12s %-12s %-12s",
                 res['name'], find_freq('Heave'), find_freq('Pitch'),
                 find_freq('Roll'), find_freq('Warp'))

    log.info("  Plots saved to: %s", plots_dir)


# ======================================================
# CLI (standalone fallback — prefer Run_Vibrations.py)
# ======================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="4-DOF Transfer Function Fitting to FPushrod PSDs.",
    )
    parser.add_argument(
        "data_file",
        help="Path to a CSV data file containing FPushrod channels "
             "(relative to project root or absolute).",
    )
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--track-front", type=float, default=1.8)
    parser.add_argument("--track-rear", type=float, default=1.8)
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=19.0)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--total-mass", type=float, default=None,
                        help="Total car mass [kg] for constraint")
    parser.add_argument("--wheelbase", type=float, default=None,
                        help="Wheelbase [m] for pitch inertia constraint")
    parser.add_argument("--pitch-inertia", type=float, default=None,
                        help="Total pitch inertia Ip [kg·m²]")
    parser.add_argument("--roll-inertia", type=float, default=None,
                        help="Total roll inertia Ix [kg·m²]")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        sys.exit(1)

    run_fit(
        filepath=data_path,
        fs=args.fs,
        track_front=args.track_front,
        track_rear=args.track_rear,
        fmin=args.fmin,
        fmax=args.fmax,
        nperseg=args.nperseg,
        total_mass=args.total_mass,
        wheelbase=args.wheelbase,
        pitch_inertia=args.pitch_inertia,
        roll_inertia=args.roll_inertia,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
