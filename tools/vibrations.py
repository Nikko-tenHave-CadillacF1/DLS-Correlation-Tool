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

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.optimize import differential_evolution, minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    "mF", "mR", "IF", "IR", "mu",
    "cFH", "cFR", "cRH", "cRR",
    "kFH", "kFR", "kRH", "kRR", "kW",
]
PARAM_UNITS = [
    "kg", "kg", "kg·m²", "kg·m²", "kg",
    "Ns/m", "Nms/rad", "Ns/m", "Nms/rad",
    "N/m", "Nm/rad", "N/m", "Nm/rad", "Nm/rad",
]
DOF_LABELS = ["Heave Front (z_F)", "Roll Front (θ_F)",
              "Heave Rear (z_R)", "Roll Rear (θ_R)"]

# Initial parameter guess (based on 770 kg car, 3.4 m wheelbase, 45% front)
# Pitch inertia Ip = 1000 kg·m² → mu = Ip/L = 1000/3.4 ≈ 294 kg
# Roll inertia total = 120 kg·m² → IF + IR = 120
DEFAULT_P0 = np.array([
    346.5,      # mF  [kg] (45% of 770)
    423.5,      # mR  [kg] (55% of 770)
    50.0,       # IF  [kg·m²] front roll inertia
    70.0,       # IR  [kg·m²] rear roll inertia
    294.0,      # mu = Ip/L [kg] (Ip = 1000 kg·m², L = 3.4 m)
    4000.0,     # cFH [Ns/m] front heave damping
    2500.0,     # cFR [Nms/rad] front roll damping
    5000.0,     # cRH [Ns/m] rear heave damping
    3000.0,     # cRR [Nms/rad] rear roll damping
    300000.0,   # kFH [N/m] front heave stiffness
    180000.0,   # kFR [Nm/rad] front roll stiffness
    400000.0,   # kRH [N/m] rear heave stiffness
    200000.0,   # kRR [Nm/rad] rear roll stiffness
    100000.0,   # kW  [Nm/rad] warp (anti-roll) stiffness
])

# Parameter bounds for optimisation [lower, upper] in physical units
# mF/mR bounds enforce 44-46% front weight distribution (770 kg total)
# IF/IR bounds: total roll inertia ~120 kg·m²
# mu bounds: Ip = 1000 kg·m², L = 3.4 m → mu = 294 kg (allow ±20%)
# Stiffness bounds keep all modes within 1-19 Hz fitting range
# Damping bounds ensure underdamped peaks (ζ < ~0.5)
BOUNDS_PHYSICAL = np.array([
    [330,    360],       # mF  (44-46% of 770 kg)
    [410,    440],       # mR  (54-56% of 770 kg)
    [20,     90],        # IF  (front roll inertia, part of 120 total)
    [30,     100],       # IR  (rear roll inertia, part of 120 total)
    [235,    360],       # mu = Ip/L (Ip ~800-1200 kg·m², L=3.4m)
    [1000,   25000],     # cFH (heave ζ ≈ 0.04-0.45)
    [300,    3000],      # cFR (roll ζ ≈ 0.05-0.58)
    [1000,   25000],     # cRH (heave ζ ≈ 0.03-0.44)
    [300,    3000],      # cRR (roll ζ ≈ 0.04-0.53)
    [80000,  2000000],   # kFH (heave ~1.8-7.7 Hz)
    [20000,  500000],    # kFR (roll ~3.2-15.9 Hz)
    [80000,  2000000],   # kRH (heave ~1.7-7.3 Hz)
    [20000,  500000],    # kRR (roll ~2.7-13.5 Hz)
    [5000,   300000],    # kW  (warp coupling)
])


# ======================================================
# 1. DATA LOADING
# ======================================================
def load_force_data(filepath: Path) -> np.ndarray:
    """Load FPushrod corner forces from CSV.

    Applies the same preprocessing as the main dataplotter pipeline:
      - Coerce non-numeric values to NaN
      - Replace int64 sentinels and infinities with NaN
      - Linear interpolation (up to 100 consecutive NaN samples)
      - Drop any remaining rows with NaN in force channels

    Corrects sign convention: front pushrods are negative by convention,
    rear are positive. Negate rear channels so all corners share the same
    sign direction (positive = compression into chassis).
    """
    df = pd.read_csv(filepath, sep=",", skiprows=[0, 2], header=0, low_memory=False)

    # Sanitize: coerce non-numeric, replace sentinels/inf with NaN
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    for ch in FORCE_CHANNELS:
        df[ch] = pd.to_numeric(df[ch], errors="coerce")
        df[ch] = df[ch].replace([int64_min, int64_max, -np.inf, np.inf], np.nan)

    # Linear interpolation (cap at 100 consecutive NaN samples, matching dataplotter)
    df[FORCE_CHANNELS] = df[FORCE_CHANNELS].interpolate(
        method="linear", limit=100, axis=0
    )

    # Drop rows that still have NaN after interpolation
    df = df.dropna(subset=FORCE_CHANNELS)

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

        M = [[mF+mu,  0,    +mu,   0 ],
             [0,      IF,    0,    0 ],
             [+mu,    0,    mR+mu, 0 ],
             [0,      0,     0,    IR]]

    Damping matrix (diagonal):
        C = diag(cFH, cFR, cRH, cRR)

    Stiffness matrix (with warp coupling kW):
        K = [[kFH,  0,        0,    0       ],
             [0,    kFR+kW,   0,   -kW      ],
             [0,    0,        kRH,  0       ],
             [0,   -kW,       0,    kRR+kW  ]]
    """
    mF, mR, IF_, IR_, mu = params[0:5]
    cFH, cFR, cRH, cRR = params[5:9]
    kFH, kFR, kRH, kRR, kW = params[9:14]

    M = np.array([
        [mF + mu,  0,      mu,      0   ],
        [0,        IF_,    0,       0   ],
        [mu,       0,      mR + mu, 0   ],
        [0,        0,      0,       IR_ ],
    ])

    C = np.diag([cFH, cFR, cRH, cRR])

    K = np.array([
        [kFH,  0,          0,    0        ],
        [0,    kFR + kW,   0,   -kW       ],
        [0,    0,          kRH,  0        ],
        [0,   -kW,         0,    kRR + kW ],
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
def cost_function(log_params: np.ndarray, freqs_fit: np.ndarray,
                  measured_psds_fit: np.ndarray,
                  total_mass: float = None, wheelbase: float = None,
                  pitch_inertia: float = None, roll_inertia: float = None) -> float:
    """Fit error: model |H|² vs measured PSDs (dB shape-only, mean-removed).

    Includes soft penalty constraints:
      - mF + mR ≈ total_mass (if provided)
      - Front weight distribution 44-46%
      - mu ≈ pitch_inertia / wheelbase (if both provided)
      - IF + IR ≈ roll_inertia (if provided)
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

    total_cost = 0.0
    for dof in range(4):
        meas_db = 10.0 * np.log10(np.maximum(measured_psds_fit[dof], 1e-30))
        model_db = 10.0 * np.log10(np.maximum(H_sq[dof], 1e-30))
        meas_db = meas_db - np.mean(meas_db)
        model_db = model_db - np.mean(model_db)
        total_cost += np.sum((meas_db - model_db)**2)

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
        mu = params[4]
        Ip_model = mu * wheelbase
        Ip_error = (Ip_model - pitch_inertia) / pitch_inertia
        total_cost += 500.0 * nf * Ip_error**2

    # Hard constraint: total roll inertia (IF + IR)
    if roll_inertia is not None:
        IF_, IR_ = params[2], params[3]
        roll_error = (IF_ + IR_ - roll_inertia) / roll_inertia
        total_cost += 500.0 * nf * roll_error**2

    return total_cost


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
    """Classify mode shape [z_F, θ_F, z_R, θ_R] as Heave/Pitch/Roll/Warp."""
    s = np.real(shape)
    s = s / np.max(np.abs(s)) if np.max(np.abs(s)) > 0 else s

    heave_content = abs(s[0]) + abs(s[2])
    roll_content = abs(s[1]) + abs(s[3])

    if heave_content > roll_content:
        return "Heave" if s[0] * s[2] > 0 else "Pitch"
    else:
        return "Roll" if s[1] * s[3] > 0 else "Warp"


# ======================================================
# 8. PLOTTING
# ======================================================
def generate_plots(freqs, psds, freqs_fit, psds_fit, M, C, K, T,
                   fn, zeta, mode_shapes, fmin, fmax):
    """Generate all diagnostic plots."""
    import matplotlib.pyplot as plt

    n_modes = len(fn)
    H_sq_fit = compute_H_mag_sq(freqs_fit, M, C, K)

    # PLOT 1: Measured vs Fitted PSDs
    fig1, axes1 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig1.suptitle("|H(jω)|² Fit to FPushrod Body-Coordinate PSDs", fontsize=13)
    for dof in range(4):
        ax = axes1[dof]
        meas_db = 10.0 * np.log10(np.maximum(psds_fit[dof], 1e-30))
        meas_db_zm = meas_db - np.mean(meas_db)
        model_db = 10.0 * np.log10(np.maximum(H_sq_fit[dof], 1e-30))
        model_db_zm = model_db - np.mean(model_db)
        ax.plot(freqs_fit, meas_db_zm, 'b-', linewidth=1.0, alpha=0.7, label="Measured")
        ax.plot(freqs_fit, model_db_zm, 'r-', linewidth=1.5, label="Fitted |H|²")
        for f_n in fn:
            if fmin <= f_n <= fmax:
                ax.axvline(f_n, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n[dB shape]", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes1[-1].set_xlabel("Frequency [Hz]")
    fig1.tight_layout()

    # PLOT 2: Full-band PSDs with mode frequencies
    fig2, axes2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig2.suptitle("FPushrod Body-Coordinate PSDs with Identified Modes", fontsize=13)
    for dof in range(4):
        ax = axes2[dof]
        valid = freqs > 0.5
        ax.semilogy(freqs[valid], psds[dof, valid], 'b-', linewidth=0.8)
        for i, f_n in enumerate(fn):
            mode_label = classify_mode(mode_shapes[:, i])
            ax.axvline(f_n, color='red', linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f"{mode_label} {f_n:.2f} Hz" if dof == 0 else "")
        ax.set_ylabel(DOF_LABELS[dof], fontsize=9)
        ax.grid(True, alpha=0.3)
    axes2[0].legend(fontsize=8, loc="upper right")
    axes2[-1].set_xlabel("Frequency [Hz]")
    fig2.tight_layout()

    # PLOT 3: Mode shapes (body coordinates)
    if n_modes > 0:
        fig3, axes3 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4), sharey=True)
        if n_modes == 1:
            axes3 = [axes3]
        dof_short = ["z_F", "θ_F", "z_R", "θ_R"]
        for i in range(n_modes):
            shape = np.real(mode_shapes[:, i])
            shape = shape / np.max(np.abs(shape))
            axes3[i].barh(range(4), shape,
                          color=['steelblue' if s >= 0 else 'coral' for s in shape])
            axes3[i].set_yticks(range(4))
            axes3[i].set_yticklabels(dof_short)
            mode_name = classify_mode(mode_shapes[:, i])
            axes3[i].set_title(f"{mode_name}\n{fn[i]:.2f} Hz\nζ={zeta[i]:.4f}", fontsize=9)
            axes3[i].axvline(0, color='k', linewidth=0.5)
            axes3[i].set_xlim(-1.2, 1.2)
        fig3.suptitle("Identified Mode Shapes (Body Coordinates)", fontsize=12)
        fig3.tight_layout()

    # PLOT 4: Transfer function magnitude
    freqs_plot = np.linspace(0.5, fmax + 2.0, 500)
    H_sq_full = compute_H_mag_sq(freqs_plot, M, C, K)
    fig4, axes4 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig4.suptitle("Fitted Transfer Function |H(jω)|² (Model)", fontsize=13)
    for dof in range(4):
        ax = axes4[dof]
        h_db = 10.0 * np.log10(np.maximum(H_sq_full[dof], 1e-30))
        ax.plot(freqs_plot, h_db, 'r-', linewidth=1.2)
        for f_n in fn:
            ax.axvline(f_n, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n[dB]", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes4[-1].set_xlabel("Frequency [Hz]")
    fig4.tight_layout()

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
                          color=['steelblue' if s >= 0 else 'coral' for s in corner_shape])
            axes5[i].set_yticks(range(4))
            axes5[i].set_yticklabels(corner_labels)
            mode_name = classify_mode(mode_shapes[:, i])
            axes5[i].set_title(f"{mode_name}\n{fn[i]:.2f} Hz", fontsize=9)
            axes5[i].axvline(0, color='k', linewidth=0.5)
            axes5[i].set_xlim(-1.2, 1.2)
        fig5.suptitle("Mode Shapes — Corner Coordinates (FL, FR, RL, RR)", fontsize=12)
        fig5.tight_layout()

    plt.show()


# ======================================================
# 9. MAIN PIPELINE
# ======================================================
def run_fit(filepath: Path, fs: float = 100.0, track_front: float = 1.8,
            track_rear: float = 1.8, fmin: float = 1.0, fmax: float = 19.0,
            nperseg: int = 1024, total_mass: float = None,
            wheelbase: float = None, pitch_inertia: float = None,
            roll_inertia: float = None, show_plots: bool = True):
    """Run the full 4-DOF transfer function fitting pipeline.

    Args:
        total_mass: Total car mass [kg]. Used as constraint on mF + mR.
        wheelbase: Wheelbase length [m]. Used to constrain mu = Ip/L.
        pitch_inertia: Total pitch inertia Ip [kg·m²]. Constrains mu.
        roll_inertia: Total roll inertia Ix [kg·m²]. Constrains IF + IR.

    Returns: (params_fit, fn, zeta, mode_shapes)
    """
    print("=" * 60)
    print("4-DOF TRANSFER FUNCTION FIT TO FPushrod PSDs")
    print("=" * 60)

    # Load and transform data
    print(f"\nLoading: {filepath}")
    F_corner = load_force_data(filepath)
    n_samples = F_corner.shape[1]
    print(f"  Samples: {n_samples} ({n_samples/fs:.1f} s)")
    if total_mass is not None:
        print(f"  Total mass constraint: {total_mass} kg")
    if wheelbase is not None:
        print(f"  Wheelbase: {wheelbase} m")
    if pitch_inertia is not None:
        print(f"  Pitch inertia constraint: {pitch_inertia} kg·m²")
    if roll_inertia is not None:
        print(f"  Roll inertia constraint: {roll_inertia} kg·m²")

    T = build_T(track_front, track_rear)
    freqs, psds = compute_body_psds(F_corner, T, fs, nperseg=nperseg)
    print(f"  PSD frequency resolution: {freqs[1]-freqs[0]:.3f} Hz")

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[freq_mask]
    psds_fit = psds[:, freq_mask]
    print(f"  Fitting range: {fmin}–{fmax} Hz ({np.sum(freq_mask)} bins)")

    # Optimisation bounds (log-space)
    bounds_log = np.log(BOUNDS_PHYSICAL)

    # Global optimisation
    print("\nRunning global optimisation (differential evolution)...")
    result_de = differential_evolution(
        cost_function,
        bounds=list(zip(bounds_log[:, 0], bounds_log[:, 1])),
        args=(freqs_fit, psds_fit, total_mass, wheelbase, pitch_inertia, roll_inertia),
        seed=42,
        maxiter=300,
        tol=1e-6,
        polish=False,
        disp=True,
        updating="deferred",
        workers=-1,
    )
    print(f"  DE cost: {result_de.fun:.2f}")

    # Local refinement (bounded)
    print("\nRefining with L-BFGS-B...")
    result = minimize(
        cost_function,
        result_de.x,
        args=(freqs_fit, psds_fit, total_mass, wheelbase, pitch_inertia, roll_inertia),
        method="L-BFGS-B",
        bounds=list(zip(bounds_log[:, 0], bounds_log[:, 1])),
        options={"maxiter": 50000, "ftol": 1e-8, "disp": True},
    )
    print(f"  Final cost: {result.fun:.2f}")

    # Extract results
    params_fit = np.exp(result.x)
    M, C, K = build_MCK(params_fit)

    print(f"\n{'='*60}")
    print("FITTED PARAMETERS")
    print(f"{'='*60}")
    print(f"{'Parameter':<8} {'Value':<15} {'Unit'}")
    print(f"{'-'*40}")
    for name, val, unit in zip(PARAM_NAMES, params_fit, PARAM_UNITS):
        print(f"  {name:<6} {val:<15.2f} {unit}")

    # Modal analysis
    fn, zeta, mode_shapes = extract_modes(M, C, K)

    print(f"\n{'='*60}")
    print("IDENTIFIED MODES")
    print(f"{'='*60}")
    print(f"{'Mode':<6} {'Type':<8} {'Freq [Hz]':<12} {'Damping ζ':<12}")
    print(f"{'-'*40}")
    for i in range(len(fn)):
        name = classify_mode(mode_shapes[:, i])
        z_str = f"{zeta[i]:.4f}" if not np.isnan(zeta[i]) else "N/A"
        print(f"  {i+1:<4} {name:<8} {fn[i]:<12.3f} {z_str:<12}")

    if show_plots:
        generate_plots(freqs, psds, freqs_fit, psds_fit, M, C, K, T,
                       fn, zeta, mode_shapes, fmin, fmax)

    return params_fit, fn, zeta, mode_shapes


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
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
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
