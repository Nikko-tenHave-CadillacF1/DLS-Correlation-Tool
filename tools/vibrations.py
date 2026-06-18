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

DISPLACEMENT_CHANNELS = [
    "xDamperPotFL",
    "xDamperPotFR",
    "xDamperPotRL",
    "xDamperPotRR",
]

PARAM_NAMES = [
    "mF", "mR", "IrF", "mu",
    "cFH", "cR", "cRH", "cW",
    "kFH", "kR", "kRH", "kW",
    "IrR",
]
PARAM_UNITS = [
    "kg", "kg", "kg·m²", "kg",
    "Ns/m", "Nms/rad", "Ns/m", "Nms/rad",
    "N/m", "Nm/rad", "N/m", "Nm/rad",
    "kg·m²",
]
DOF_LABELS = ["Heave Front (z_F)", "Roll Front (th_F)",
              "Heave Rear (z_R)", "Roll Rear (th_R)"]

# Expected body-mode natural frequencies for a typical Formula car [Hz].
# Used to derive stiffness bounds so each mode's freq band is bracketed by
# the bounds: f = sqrt(k/I)/(2π) ⇒ k = (2π f)² I.
_EXPECTED_FREQS = {
    "heave": (2.0, 6.0),    # heave front (z_F) and rear (z_R) ride frequencies
    "roll":  (3.0, 6.0),    # symmetric roll mode
    "warp":  (5.0, 11.0),   # anti-roll-bar-stiffened warp mode
}

# Parameter bounds for optimisation [lower, upper] in physical units.
# Stiffness bounds derived from _EXPECTED_FREQS so the bracketed natural
# frequency lies inside [F_MIN, F_MAX] of a typical body-mode fit window.
BOUNDS_PHYSICAL = np.array([
    [200,    500],       # mF   [kg]   front sprung mass (~45% of total)
    [300,    600],       # mR   [kg]   rear sprung mass  (~55% of total)
    [10,      80],       # IrF  [kg·m²] front axle roll inertia
    [50,     500],       # mu   [kg]   = Ip/L pitch inertia coupling
    [500,   5000],       # cFH  [Ns/m]  front heave damping
    [100,   2000],       # cR   [Nms/rad] roll damping (per-axle, symmetric)
    [500,   8000],       # cRH  [Ns/m]  rear heave damping
    [200,   5000],       # cW   [Nms/rad] warp damping (ARB)
    [50000,  500000],    # kFH  [N/m]   front heave stiffness
    [5000,   80000],     # kR   [Nm/rad] roll stiffness (per-axle, symmetric)
    [80000,  800000],    # kRH  [N/m]   rear heave stiffness
    [50000,  2000000],   # kW   [Nm/rad] warp (ARB) stiffness
    [10,      80],       # IrR  [kg·m²] rear axle roll inertia
])


# ======================================================
# 1. DATA LOADING
# ======================================================

def _load_channels(filepath: Path, channels: list, fs: float) -> np.ndarray:
    """Load named channels from CSV, sanitise, interpolate, and 2 Hz high-pass."""
    df = pd.read_csv(filepath, sep=",", skiprows=[0, 2], header=0, low_memory=False)
    for ch in channels:
        df[ch] = sanitize_numeric_series(df[ch])
    df[channels] = df[channels].interpolate(method="linear", limit=100, axis=0)
    df = df.dropna(subset=channels)
    for ch in channels:
        filtered, ok = _apply_butterworth_filter_to_data(
            df[ch].values, cutoff=2, order=2, sample_rate=fs, btype="high"
        )
        if ok:
            df[ch] = filtered
        else:
            log.warning("High-pass filter failed for channel '%s'.", ch)
    return df[channels].astype(float).values.T  # [n_channels x N]


def load_force_data(filepath: Path, fs: float = RESAMPLE_RATE) -> np.ndarray:
    """Load FPushrod corner forces. Rear pushrods are negated so all corners
    share the same sign convention (positive = compression into chassis)."""
    F_corner = _load_channels(filepath, FORCE_CHANNELS, fs)
    F_corner[2] *= -1  # RL
    F_corner[3] *= -1  # RR
    return F_corner


def load_displacement_data(filepath: Path, fs: float = RESAMPLE_RATE) -> np.ndarray:
    """Load xDamperPot corner displacements."""
    return _load_channels(filepath, DISPLACEMENT_CHANNELS, fs)

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
        [1,            -1,            0,             0            ],
        [0,            0,             0.5,           0.5          ],
        [0,            0,             1,             -1           ],
    ])


# ======================================================
# 3. PSD COMPUTATION
# ======================================================
def compute_body_psds(F_corner: np.ndarray, T: np.ndarray,
                      fs: float, nperseg: int = 1024):
    """Transform corner forces to body coordinates and compute PSDs."""
    F_body = T @ F_corner
    freqs, psds = signal.welch(F_body, fs, nperseg=nperseg, axis=1)
    return freqs, psds


# ======================================================
# 4. MODEL DEFINITION: M, C, K MATRICES
# ======================================================
def build_MCK(params: np.ndarray):
    """Construct M, C, K matrices from the 13-element parameter vector.

    Mass matrix (with pitch inertia coupling via mu, asymmetric roll inertias):
        The off-diagonal coupling is POSITIVE for a typical race car where
        Ip < m*a*b (dynamic index < 1). This ensures heave mode has higher
        effective mass than pitch mode, giving ω_heave < ω_pitch.

        M = [[mF-mu,  0,    +mu,   0  ],
             [0,      IrF,   0,    0  ],
             [+mu,    0,    mR-mu, 0  ],
             [0,      0,     0,    IrR]]

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
    mF, mR, IrF, mu = params[0:4]
    cFH, cR, cRH, cW = params[4:8]
    kFH, kR, kRH, kW = params[8:12]
    IrR = params[12]

    M = np.array([
        [mF - mu,  0,      +mu,      0   ],
        [0,        IrF,    0,        0   ],
        [+mu,      0,      mR - mu,  0   ],
        [0,        0,      0,        IrR ],
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
# 5. TRANSFER FUNCTION |H(jω)|²  (vectorised over frequency)
# ======================================================
def compute_H_mag_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute |H(jω)|² at each frequency.

    H(jω) = (K - ω²M + jωC)⁻¹  — receptance matrix.

    Returns: [n_dof, nf] array — sum of |H_ij|² over input DOFs for each
    output DOF i (i.e. the row sum, which is what an input-uncorrelated
    PSD predicts at the output).
    """
    H_ij_sq = _compute_H_ij_sq(freqs_hz, M, C, K)  # (n_dof, n_dof, nf)
    return H_ij_sq.sum(axis=1)                     # sum over input cols


def _compute_H_ij_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Vectorised |H_ij(jω)|² for a stack of frequencies.

    Returns (n_dof, n_dof, nf).
    """
    omega = 2.0 * np.pi * freqs_hz                              # (nf,)
    Z = (K[None] - (omega**2)[:, None, None] * M[None]
         + 1j * omega[:, None, None] * C[None])                 # (nf, n, n)
    H = np.linalg.inv(Z)                                        # (nf, n, n)
    return np.transpose(np.abs(H)**2, (1, 2, 0))                # (n, n, nf)


# ======================================================
# 6. COST FUNCTION
# ======================================================
def _normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise a 1-D array to [0, 1] by its maximum."""
    peak = np.max(arr)
    return arr / peak if peak > 0 else arr


# --- Decoupled sub-system cost functions ---
# Heave/Pitch indices in full param vector: mF(0), mR(1), mu(3), cFH(4), cRH(6), kFH(8), kRH(10)
_HEAVE_PITCH_IDX = [0, 1, 3, 4, 6, 8, 10]
# Roll/Warp indices: IrF(2), cR(5), cW(7), kR(9), kW(11), IrR(12)
_ROLL_WARP_IDX = [2, 5, 7, 9, 11, 12]


def _normalise_expected_freqs(expected_freqs: dict | None) -> dict:
    """Normalise the user-supplied expected_freqs dict to a dict of
    (lo, hi, mid) tuples, one per mode.

    Each value may be a scalar f (treated as f ± 15 %) or a (lo, hi) tuple.
    Missing keys fall back to the defaults in _EXPECTED_FREQS.
    """
    out = {}
    for mode in ("heave", "pitch", "roll", "warp"):
        if expected_freqs and mode in expected_freqs:
            v = expected_freqs[mode]
            if isinstance(v, (tuple, list)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
            else:
                f = float(v)
                lo, hi = f * 0.85, f * 1.15
        else:
            # Defaults: roll/warp from _EXPECTED_FREQS, heave/pitch sensible defaults
            if mode in _EXPECTED_FREQS:
                lo, hi = _EXPECTED_FREQS[mode]
            elif mode == "heave":
                lo, hi = 3.0, 7.0
            else:  # pitch
                lo, hi = 6.0, 11.0
        out[mode] = (lo, hi, 0.5 * (lo + hi))
    return out


def _seed_from_expected_freqs(expected_freqs: dict) -> tuple:
    """Derive a 13-element parameter seed from user-supplied expected modal
    frequencies and return its (heave/pitch, roll/warp) log-space slices,
    ready to pass to DE as ``x0``.

    Uses the geometric mean of each parameter's bounds for masses, inertias
    and dampings (they're weakly constrained by frequency), then back-solves
    stiffnesses via k = (2π f)² · I using the decoupled approximation.
    """
    fr = _normalise_expected_freqs(expected_freqs)
    fH = fr["heave"][2]
    fP = fr["pitch"][2]
    fR = fr["roll"][2]
    fW = fr["warp"][2]

    # Mass/inertia seeds: a small mu keeps the decoupled-stiffness back-solve
    # accurate (the heave/pitch coupling vanishes as mu → 0). Use 2× the lower
    # bound rather than the geo-mean (which would over-couple the seed).
    mF, mR, IrF = 320.0, 450.0, 30.0
    IrR = 30.0
    mu = BOUNDS_PHYSICAL[3, 0] * 2.0
    cFH = np.sqrt(BOUNDS_PHYSICAL[4, 0] * BOUNDS_PHYSICAL[4, 1])
    cR  = np.sqrt(BOUNDS_PHYSICAL[5, 0] * BOUNDS_PHYSICAL[5, 1])
    cRH = np.sqrt(BOUNDS_PHYSICAL[6, 0] * BOUNDS_PHYSICAL[6, 1])
    cW  = np.sqrt(BOUNDS_PHYSICAL[7, 0] * BOUNDS_PHYSICAL[7, 1])

    kFH = (2.0 * np.pi * fH)**2 * (mF - mu)
    kRH = (2.0 * np.pi * fP)**2 * (mR - mu)
    kR = (2.0 * np.pi * fR)**2 * IrF
    kW = ((2.0 * np.pi * fW)**2 * IrF - kR) / 2.0

    seed = np.array([mF, mR, IrF, mu, cFH, cR, cRH, cW, kFH, kR, kRH, kW, IrR])
    # Pull strictly inside the bounds so log() is safe and DE has wiggle room
    seed = np.clip(seed, BOUNDS_PHYSICAL[:, 0] * 1.001,
                         BOUNDS_PHYSICAL[:, 1] * 0.999)
    log_seed = np.log(seed)
    return log_seed[_HEAVE_PITCH_IDX], log_seed[_ROLL_WARP_IDX]


def _shape_residual(meas_norm: np.ndarray, model: np.ndarray,
                    weights: np.ndarray = None) -> float:
    """Amplitude-weighted normalised-shape SSE (peak-biased).

    Optional per-frequency ``weights`` localise the fit to specific
    frequency bands (e.g. body-mode neighbourhoods), removing the
    track-noise input-spectrum bias on damping.
    """
    model_norm = _normalise(model)
    err = meas_norm * (model_norm - meas_norm)**2
    if weights is not None:
        err = err * weights
    return float(np.sum(err))


def _band_window(freqs: np.ndarray, ranges: list,
                 floor: float = 0.0) -> np.ndarray:
    """Per-frequency weights forming a union of Gaussians centred on each
    declared (lo, hi) band. sigma = half-width, so band edges sit at ~0.61
    weight and the curve falls to ~0.13 one half-width beyond the edge.
    Bands are combined by element-wise max so overlap isn't double-counted.

    Used to localise the shape-fit to resonance neighbourhoods, so the
    optimiser can't broaden a peak (raise zeta) just to fit the LF
    track-noise tail. Floor lets the LF tail still bias the fit slightly
    if you want.
    """
    if not ranges:
        return np.ones_like(freqs, dtype=float)
    w = np.full_like(freqs, floor, dtype=float)
    for r in ranges:
        if r is None:
            continue
        lo, hi = r[0], r[1]
        mid = 0.5 * (lo + hi)
        sigma = max(0.5 * (hi - lo), 1e-6)
        w = np.maximum(w, np.exp(-0.5 * ((freqs - mid) / sigma)**2))
    return w


def _band_penalty(model_freqs: np.ndarray, ranges: list, weight: float) -> float:
    """Hard band-constraint cost: each model frequency must lie inside its
    assigned (lo, hi) range. Out-of-band gives a quadratic penalty in the
    fractional overshoot, scaled by ``weight``.

    1-to-1 sorted assignment so two modes can't both satisfy the same band.
    """
    if not ranges:
        return 0.0
    sorted_model = np.sort(model_freqs)
    sorted_ranges = sorted(ranges, key=lambda r: 0.5 * (r[0] + r[1]))
    n = min(len(sorted_model), len(sorted_ranges))
    cost = 0.0
    for fm, (lo, hi) in zip(sorted_model[:n], sorted_ranges[:n]):
        band = max(hi - lo, 1.0)
        if fm < lo:
            cost += weight * ((lo - fm) / band)**2
        elif fm > hi:
            cost += weight * ((fm - hi) / band)**2
    return cost


def _cost_heave_pitch(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                      meas_norm: np.ndarray,
                      total_mass: float = None, wheelbase: float = None,
                      pitch_inertia: float = None,
                      heave_range: tuple = None,
                      pitch_range: tuple = None,
                      weights: np.ndarray = None) -> float:
    """Cost for the heave/pitch subsystem (output DOFs z_F, z_R).

    meas_norm: pre-normalised measured PSDs (shape [4, nf]) — DOFs 0 and 2 used.
    heave_range / pitch_range: optional (lo, hi) Hz bands. When set, model
    mode frequencies that escape their band are heavily penalised.
    weights: optional per-frequency window restricting the shape-fit to
    resonance neighbourhoods (see ``_band_window``).
    """
    mF, mR, mu, cFH, cRH, kFH, kRH = np.exp(log_sub_params)

    M_hp = np.array([[mF - mu, mu], [mu, mR - mu]])
    if np.any(np.linalg.eigvalsh(M_hp) <= 0):
        return 1e15
    C_hp = np.diag([cFH, cRH])
    K_hp = np.diag([kFH, kRH])

    H_sq = _compute_H_ij_sq(freqs_fit, M_hp, C_hp, K_hp).sum(axis=1)  # (2, nf)

    cost = (_shape_residual(meas_norm[0], H_sq[0], weights)
            + _shape_residual(meas_norm[2], H_sq[1], weights))

    nf = len(freqs_fit)

    # Hard band constraints on the analytic mode frequencies.
    # Coupled 2-DOF (mu coupling) closed form via 2x2 eigenvalue:
    M_inv_K = np.linalg.solve(M_hp, K_hp)
    eigvals = np.linalg.eigvals(M_inv_K).real
    eigvals = np.clip(eigvals, 0.0, None)
    model_fns = np.sqrt(eigvals) / (2.0 * np.pi)
    ranges = [r for r in (heave_range, pitch_range) if r is not None]
    if ranges:
        cost += _band_penalty(model_fns, ranges, weight=100.0 * nf)

    if total_mass is not None:
        cost += 500.0 * nf * ((mF + mR - total_mass) / total_mass)**2
        front_pct = mF / total_mass
        if not (0.44 <= front_pct <= 0.46):
            cost += 500.0 * nf * (front_pct - 0.45)**2
    if pitch_inertia is not None and wheelbase is not None:
        cost += 500.0 * nf * ((mu * wheelbase - pitch_inertia) / pitch_inertia)**2
    return cost


def _cost_roll_warp(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                    meas_norm: np.ndarray, meas_raw: np.ndarray,
                    roll_inertia: float = None,
                    disp_norm: np.ndarray = None,
                    roll_range: tuple = None,
                    warp_range: tuple = None,
                    weights: np.ndarray = None) -> float:
    """Cost for the roll/warp subsystem (output DOFs θ_F, θ_R).

    Three terms (last two optional):
      1. Amplitude-weighted shape match to measured force PSDs (DOFs 1, 3).
      2. Model-predicted displacement PSD vs measured displacement PSD:
            S_x_pred[i] = Σ_j |H_ij|² · S_F_meas[j]
         Directly constrains the TF shape, independent of input spectrum.
      3. Hard band constraints: model mode frequencies must lie inside their
         declared (lo, hi) ranges; quadratic penalty in fractional overshoot.

    weights: optional per-frequency window restricting the shape-fit to
    resonance neighbourhoods (see ``_band_window``).
    """
    IrF, cR, cW, kR, kW, IrR = np.exp(log_sub_params)

    M_rw = np.diag([IrF, IrR])
    C_rw = np.array([[cR + cW, -cW], [-cW, cR + cW]])
    K_rw = np.array([[kR + kW, -kW], [-kW, kR + kW]])

    H_ij_sq = _compute_H_ij_sq(freqs_fit, M_rw, C_rw, K_rw)  # (2, 2, nf)
    H_sq = H_ij_sq.sum(axis=1)                               # (2, nf)

    cost = (_shape_residual(meas_norm[1], H_sq[0], weights)
            + _shape_residual(meas_norm[3], H_sq[1], weights))

    if disp_norm is not None:
        # S_F input is raw measured force PSD on DOFs 1 and 3
        pred_thf = H_ij_sq[0, 0] * meas_raw[1] + H_ij_sq[0, 1] * meas_raw[3]
        pred_thr = H_ij_sq[1, 0] * meas_raw[1] + H_ij_sq[1, 1] * meas_raw[3]
        cost += _shape_residual(disp_norm[1], pred_thf, weights)
        cost += _shape_residual(disp_norm[3], pred_thr, weights)

    # Analytic undamped natural frequencies for asymmetric 2-DOF system:
    # eigenvalues of M_rw⁻¹ K_rw, closed form via 2x2 trace/det.
    T = (kR + kW) * (1.0 / IrF + 1.0 / IrR)
    D = kR * (kR + 2.0 * kW) / (IrF * IrR)
    disc = max(T * T - 4.0 * D, 0.0)
    om1_sq = 0.5 * (T - np.sqrt(disc))
    om2_sq = 0.5 * (T + np.sqrt(disc))
    model_fns = np.array([np.sqrt(om1_sq), np.sqrt(om2_sq)]) / (2.0 * np.pi)

    nf = len(freqs_fit)
    ranges = [r for r in (roll_range, warp_range) if r is not None]
    if ranges:
        cost += _band_penalty(model_fns, ranges, weight=100.0 * nf)

    if roll_inertia is not None:
        cost += 500.0 * nf * ((IrF + IrR - roll_inertia) / roll_inertia)**2
    return cost


# ======================================================
# 6b. LORENTZIAN PER-DOF FIT (sum-of-SDOF)
# ======================================================
# Drops the shared-pole constraint of the 4-DOF body MCK: each output DOF
# gets its own (f, zeta, A) per declared band. Better at tracking real-world
# asymmetry (different effective f/zeta per corner) at the cost of losing a
# single self-consistent body-MCK output.

# Which mode bands belong to which body DOF
_DOF_BAND_LABELS = [
    ("heave", "pitch"),  # z_F
    ("roll",  "warp"),   # th_F
    ("heave", "pitch"),  # z_R
    ("roll",  "warp"),   # th_R
]
_MODE_ORDER = ["Heave", "Pitch", "Roll", "Warp"]


def _lorentz_psd(freqs_hz: np.ndarray, f0: float, zeta: float, A: float) -> np.ndarray:
    """SDOF receptance |H|^2: A / [(w0^2 - w^2)^2 + (2 zeta w0 w)^2]."""
    omega = 2.0 * np.pi * freqs_hz
    omega0 = 2.0 * np.pi * f0
    denom = (omega0**2 - omega**2)**2 + (2.0 * zeta * omega0 * omega)**2
    return A / np.maximum(denom, 1e-30)


def _eval_lorentz_sum(freqs_hz: np.ndarray, dof_params: np.ndarray) -> np.ndarray:
    """Sum of N Lorentzians. dof_params shape (N, 3): (f0, zeta, A) per row."""
    out = np.zeros_like(freqs_hz, dtype=float)
    for f0, z, A in dof_params:
        out += _lorentz_psd(freqs_hz, f0, z, A)
    return out


def _cost_lorentz_dof(packed: np.ndarray, freqs_fit: np.ndarray,
                      meas_norm_dof: np.ndarray, n_bands: int,
                      weights: np.ndarray) -> float:
    params = packed.reshape(n_bands, 3).copy()
    params[:, 2] = np.exp(params[:, 2])  # amplitude in log-space
    model = _eval_lorentz_sum(freqs_fit, params)
    return _shape_residual(meas_norm_dof, model, weights)


def _fit_lorentz_dof(freqs_fit: np.ndarray, meas_norm_dof: np.ndarray,
                     ranges: list, weights: np.ndarray) -> np.ndarray:
    """Fit N Lorentzians (one per range) to a single DOF's normalised PSD.
    Returns (N, 3) of (f0_Hz, zeta, A) per band, sorted by f0.
    """
    n_bands = len(ranges)
    bounds = []
    for lo, hi in ranges:
        bounds += [(lo, hi), (0.02, 0.50), (np.log(1e-12), np.log(1e6))]
    result = differential_evolution(
        _cost_lorentz_dof,
        bounds=bounds,
        args=(freqs_fit, meas_norm_dof, n_bands, weights),
        seed=42, popsize=30, polish=True, disp=False,
        updating="deferred", workers=1,
        init="sobol", tol=1e-8, mutation=(0.5, 1.5),
    )
    out = result.x.reshape(n_bands, 3).copy()
    out[:, 2] = np.exp(out[:, 2])
    return out[np.argsort(out[:, 0])]


def _summarise_lorentz(params: np.ndarray) -> tuple:
    """Collapse the (4 DOFs x 2 bands x 3 params) array into per-mode
    (fn, zeta) summary arrays, ordered Heave/Pitch/Roll/Warp. Average is
    weighted by each Lorentzian's peak height ~ A / (2 zeta w0^2)^2.
    """
    fn = np.zeros(4)
    zeta = np.zeros(4)
    for mi, mlabel in enumerate(_MODE_ORDER):
        f_vals, z_vals, w_vals = [], [], []
        target = mlabel.lower()
        for d in range(4):
            for bi, blabel in enumerate(_DOF_BAND_LABELS[d]):
                if blabel == target:
                    f, z, A = params[d, bi]
                    f_vals.append(f); z_vals.append(z)
                    om0 = 2.0 * np.pi * f
                    peak = A / max((2.0 * z * om0**2)**2, 1e-30)
                    w_vals.append(peak)
        f_vals = np.asarray(f_vals); z_vals = np.asarray(z_vals); w_vals = np.asarray(w_vals)
        if w_vals.sum() > 0:
            fn[mi] = np.average(f_vals, weights=w_vals)
            zeta[mi] = np.average(z_vals, weights=w_vals)
        else:
            fn[mi] = f_vals.mean(); zeta[mi] = z_vals.mean()
    return fn, zeta


# ======================================================
# Unified model PSD evaluator (dispatches on fit method)
# ======================================================
def eval_fit_psds(result: dict, freqs_hz: np.ndarray) -> np.ndarray:
    """Return (4, nf) un-normalised model PSD per body DOF for a fit result.

    Dispatches on result['method']: 'body4dof' builds MCK then evaluates
    |H|^2; 'lorentzian' sums the per-DOF SDOF terms.
    """
    method = result["method"]
    params = result["params"]
    if method == "lorentzian":
        return np.stack([_eval_lorentz_sum(freqs_hz, params[d]) for d in range(4)])
    if method == "body4dof":
        M, C, K = build_MCK(params)
        return compute_H_mag_sq(freqs_hz, M, C, K)
    raise ValueError(f"Unknown fit method: {method!r}")


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

# Shared colour palette for vibration plots (kept stable so plots are
# directly comparable across runs in a technical report).
_MEAS_COLOR = "#2000BF"   # measured PSD trace
_FIT_COLOR  = "#D70000"   # model fit trace
_MODE_COLOR = "#00AA55"   # mode-frequency vertical guide lines
_RESID_COLOR = "#D70000"  # residual SSE shading
_POS_BAR = "#2E86AB"      # positive mode-shape bar
_NEG_BAR = "#E05263"      # negative mode-shape bar
_SAVE_KW = dict(pad_inches=0.15, facecolor="white", bbox_inches="tight")


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")


def _add_suptitle(fig, event: str, run_name: str, plot_type: str,
                  method: str = None, extras: str = None) -> None:
    """Two-line figure header matching the wider plot library.

    Top line: bold uppercase plot type (always present).
    Bottom line: "Event - Run [- method] [| extras]" (skipped if all empty).
    """
    fig.suptitle(plot_type.upper(),
                 fontsize=_PLOT_FONT["figure_title_size"],
                 fontweight="bold", color=_INK, y=0.995)
    sub_parts = []
    if event:
        sub_parts.append(str(event))
    if run_name:
        sub_parts.append(str(run_name))
    if method:
        sub_parts.append(f"method={method}")
    line = "  -  ".join(sub_parts)
    if extras:
        line = f"{line}    |    {extras}" if line else extras
    if line:
        fig.text(0.5, 0.965, line, ha="center", va="top",
                 fontsize=_PLOT_FONT["label_size"], color=_INK)


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
                   output_dir: Path = None,
                   event: str = "", run_name: str = "",
                   output_dpi: int = 300):
    """Generate the full body-4DOF figure set and save to output_dir/plots/.

    Only used by the ``body4dof`` method (mode shapes and a single TF
    require the body MCK and the modal eigenvectors).
    """
    _configure_style()

    plots_dir = (output_dir / "plots" / "vibrations") if output_dir else Path(".") / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_modes = len(fn)
    H_sq_fit = compute_H_mag_sq(freqs_fit, M, C, K)
    safe = _safe_name(run_name) if run_name else "fit"

    # PLOT 1: Measured vs Fitted PSDs
    fig1, axes1 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for dof in range(4):
        ax = axes1[dof]
        ax.plot(freqs_fit, _normalise(psds_fit[dof]), color=_MEAS_COLOR,
                linewidth=1.8, alpha=0.85, label="Measured" if dof == 0 else None)
        ax.plot(freqs_fit, _normalise(H_sq_fit[dof]), color=_FIT_COLOR,
                linewidth=1.8, alpha=0.85, label="Fitted" if dof == 0 else None)
        for f_n in fn:
            if fmin <= f_n <= fmax:
                ax.axvline(f_n, color=_MODE_COLOR, linestyle="--",
                           linewidth=0.9, alpha=0.7)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(norm.)",
                      fontsize=9.5, fontweight="bold", rotation=0,
                      ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax)
    axes1[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig1, event, run_name, "Modal Fit - Measured vs Fitted",
                  method="body4dof", extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.955))
    fig1.savefig(plots_dir / f"vibrations_fit_{safe}.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig1)

    # PLOT 2: Full-band PSDs with mode frequencies
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for dof in range(4):
        ax = axes2[dof]
        valid = freqs > 0.5
        ax.semilogy(freqs[valid], psds[dof, valid], color=_MEAS_COLOR,
                    linewidth=1.6, alpha=0.85)
        for i, f_n in enumerate(fn):
            mode_label = classify_mode(mode_shapes[:, i])
            ax.axvline(f_n, color=_FIT_COLOR, linestyle="--",
                       linewidth=1.0, alpha=0.7,
                       label=f"{mode_label} {f_n:.2f} Hz" if dof == 0 else None)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(PSD)",
                      fontsize=9.5, fontweight="bold", rotation=0,
                      ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax)
    axes2[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig2, event, run_name, "Body PSDs with Mode Frequencies",
                  method="body4dof")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.955))
    fig2.savefig(plots_dir / f"vibrations_psd_modes_{safe}.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig2)

    # PLOT 3: Mode shapes (body coordinates)
    if n_modes > 0:
        fig3, axes3 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4.6), sharey=True)
        if n_modes == 1:
            axes3 = [axes3]

        dof_short = ["z_F", "th_F", "z_R", "th_R"]
        for i in range(n_modes):
            shape = np.real(mode_shapes[:, i])
            shape = shape / np.max(np.abs(shape))
            axes3[i].barh(range(4), shape,
                          color=[_POS_BAR if s >= 0 else _NEG_BAR for s in shape],
                          edgecolor=_INK, linewidth=0.5)
            axes3[i].set_yticks(range(4))
            axes3[i].set_yticklabels(dof_short)
            mode_name = classify_mode(mode_shapes[:, i])
            axes3[i].set_title(f"{mode_name}\n{fn[i]:.2f} Hz  z={zeta[i]:.4f}",
                               fontsize=10)
            axes3[i].axvline(0, color=_INK, linewidth=0.6)
            axes3[i].set_xlim(-1.2, 1.2)
            _style_axis(axes3[i], grid_axis="x")
        _add_suptitle(fig3, event, run_name, "Mode Shapes - Body Coords",
                      method="body4dof")
        plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.86))
        fig3.savefig(plots_dir / f"vibrations_mode_shapes_body_{safe}.png",
                     dpi=output_dpi, **_SAVE_KW)
        plt.close(fig3)

    # PLOT 4: Transfer function magnitude (normalised)
    freqs_plot = np.linspace(0.5, fmax + 2.0, 500)
    H_sq_full = compute_H_mag_sq(freqs_plot, M, C, K)
    fig4, axes4 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for dof in range(4):
        ax = axes4[dof]
        ax.plot(freqs_plot, _normalise(H_sq_full[dof]),
                color=_FIT_COLOR, linewidth=1.8, alpha=0.85)
        for f_n in fn:
            ax.axvline(f_n, color=_MODE_COLOR, linestyle="--",
                       linewidth=0.9, alpha=0.7)
        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(norm.)",
                      fontsize=9.5, fontweight="bold", rotation=0,
                      ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
    axes4[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig4, event, run_name, "Fitted Transfer Function",
                  method="body4dof")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.955))
    fig4.savefig(plots_dir / f"vibrations_transfer_function_{safe}.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig4)

    # PLOT 5: Mode shapes (corner coordinates)
    if n_modes > 0:
        T_inv = np.linalg.inv(T)
        fig5, axes5 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4.6), sharey=True)
        if n_modes == 1:
            axes5 = [axes5]

        corner_labels = ["FL", "FR", "RL", "RR"]
        for i in range(n_modes):
            body_shape = np.real(mode_shapes[:, i])
            corner_shape = T_inv @ body_shape
            corner_shape = corner_shape / np.max(np.abs(corner_shape))
            axes5[i].barh(range(4), corner_shape,
                          color=[_POS_BAR if s >= 0 else _NEG_BAR
                                 for s in corner_shape],
                          edgecolor=_INK, linewidth=0.5)
            axes5[i].set_yticks(range(4))
            axes5[i].set_yticklabels(corner_labels)
            mode_name = classify_mode(mode_shapes[:, i])
            axes5[i].set_title(f"{mode_name}\n{fn[i]:.2f} Hz", fontsize=10)
            axes5[i].axvline(0, color=_INK, linewidth=0.6)
            axes5[i].set_xlim(-1.2, 1.2)
            _style_axis(axes5[i], grid_axis="x")
        _add_suptitle(fig5, event, run_name, "Mode Shapes - Corner Coords",
                      method="body4dof")
        plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.86))
        fig5.savefig(plots_dir / f"vibrations_mode_shapes_corner_{safe}.png",
                     dpi=output_dpi, **_SAVE_KW)
        plt.close(fig5)

    log.info("  Plots saved to: %s", plots_dir)


def _generate_diagnosis_plot(result: dict, freqs_fit, psds_fit,
                             fmin, fmax, run_name, output_dir,
                             event: str = "", output_dpi: int = 300):
    """Per-run diagnosis figure: measured vs fitted normalised PSDs for
    all 4 DOFs plus residual SSE. Saved as vibrations_diag_<run_name>.png.

    Lorentzian fits annotate the per-DOF (f, zeta) of each band inline
    (since each DOF has its own pole locations). Body-MCK fits draw a
    single shared (f, zeta) per mode in the top-axis legend.
    """
    _configure_style()
    plots_dir = (output_dir / "plots" / "vibrations") if output_dir else Path(".") / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)

    H_sq_fit = eval_fit_psds(result, freqs_fit)
    fn = result["fn"]
    zeta = result["zeta"]
    mode_labels = result["mode_labels"]
    method = result["method"]
    is_lorentz = (method == "lorentzian")

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5]})

    # Per-DOF: measured vs fitted
    for dof in range(4):
        ax = axes[dof]
        meas = _normalise(psds_fit[dof])
        model = _normalise(H_sq_fit[dof])
        ax.plot(freqs_fit, meas, color=_MEAS_COLOR, linewidth=1.6, alpha=0.85,
                label="Measured" if dof == 0 else None)
        ax.plot(freqs_fit, model, color=_FIT_COLOR, linewidth=1.6, alpha=0.85,
                label="Fitted" if dof == 0 else None)
        ax.fill_between(freqs_fit, meas, model, color=_FIT_COLOR, alpha=0.12)

        if is_lorentz:
            # Per-DOF Lorentzian: draw vertical lines at THIS DOF's poles and
            # annotate (f, zeta) inline so per-corner asymmetry is visible.
            dof_params = result["params"][dof]   # (n_bands, 3) -> (f0, zeta, A)
            dof_band_labels = _DOF_BAND_LABELS[dof]
            ann_lines = []
            for bi in range(dof_params.shape[0]):
                f0, z, _ = dof_params[bi]
                if fmin <= f0 <= fmax:
                    ax.axvline(f0, color=_MODE_COLOR, linestyle="--",
                               linewidth=0.9, alpha=0.7)
                ann_lines.append(f"{dof_band_labels[bi].capitalize()}: "
                                 f"{f0:.2f} Hz  z={z:.3f}")
            ax.text(0.985, 0.93, "\n".join(ann_lines),
                    transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold",
                    family="monospace", color=_INK,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.22",
                              facecolor="white", alpha=0.92,
                              edgecolor="#3C3C3C", linewidth=0.8))
        else:
            # body4dof: shared poles -> single (f, zeta) per mode, in the
            # top-axis legend.
            for i, f_n in enumerate(fn):
                if fmin <= f_n <= fmax:
                    z_str = f" z={zeta[i]:.2f}" if not np.isnan(zeta[i]) else ""
                    ax.axvline(f_n, color=_MODE_COLOR, linestyle="--",
                               linewidth=0.9, alpha=0.7,
                               label=(f"{mode_labels[i]} {f_n:.1f} Hz{z_str}"
                                      if dof == 0 else None))

        ax.set_ylabel(f"{DOF_LABELS[dof]}\n(norm.)",
                      fontsize=9.5, fontweight="bold", rotation=0,
                      ha="right", va="center")
        ax.yaxis.set_label_coords(-0.035, 0.5)
        ax.set_ylim(bottom=0)
        _style_axis(ax, grid_axis="y")
        if dof == 0:
            _add_legend(ax, loc="upper left" if is_lorentz else "upper right")

    # Residual row: per-frequency sum-of-squared-error across DOFs
    ax_res = axes[4]
    residual = np.zeros_like(freqs_fit)
    for dof in range(4):
        residual += (_normalise(psds_fit[dof]) - _normalise(H_sq_fit[dof]))**2
    ax_res.fill_between(freqs_fit, 0, residual, color=_RESID_COLOR, alpha=0.3)
    ax_res.plot(freqs_fit, residual, color=_RESID_COLOR, linewidth=1.0, alpha=0.7)
    total_sse = float(np.trapezoid(residual, freqs_fit))
    ax_res.text(0.985, 0.92, f"integral SSE = {total_sse:.3f}",
                transform=ax_res.transAxes, fontsize=8.5, fontweight="bold",
                family="monospace", color=_INK, va="top", ha="right")
    ax_res.set_ylabel("Residual\n(SSE)",
                      fontsize=9.5, fontweight="bold", rotation=0,
                      ha="right", va="center")
    ax_res.yaxis.set_label_coords(-0.035, 0.5)
    ax_res.set_ylim(bottom=0)
    ax_res.yaxis.set_major_locator(plt.MaxNLocator(3))
    _style_axis(ax_res, grid_axis="y")
    axes[-1].set_xlabel("Frequency [Hz]")

    _add_suptitle(fig, event, run_name, "Modal Fit - Diagnosis",
                  method=method,
                  extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.955))
    safe_name = _safe_name(run_name)
    fig.savefig(plots_dir / f"vibrations_diag_{safe_name}.png",
                dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  Diagnosis plot saved: vibrations_diag_%s.png", safe_name)


# ======================================================
# 9. MAIN PIPELINE
# ======================================================
def run_fit(filepath: Path, fs: float = RESAMPLE_RATE, track_front: float = 1.8,
            track_rear: float = 1.8, fmin: float = 1.0, fmax: float = 12.0,
            nperseg: int = 1024, total_mass: float = None,
            wheelbase: float = None, pitch_inertia: float = None,
            roll_inertia: float = None, show_plots: bool = True,
            output_dir: Path = None, run_name: str = None,
            displacement_mode: bool = False,
            expected_freqs: dict = None,
            method: str = "lorentzian",
            event: str = "",
            output_dpi: int = 300) -> dict:
    """Fit a modal model to measured body PSDs and return a result dict.

    method:
      * ``"lorentzian"`` (default) — per-DOF sum-of-SDOF fit. Each output
        DOF gets its own (f, zeta, A) per declared band, so asymmetric
        front/rear damping or split mode frequencies are captured. No
        single body-MCK is produced.
      * ``"body4dof"`` — original 13-parameter MCK fit. Produces a
        self-consistent body model with mass/stiffness/damping and mode
        shapes; constrained to one (f, zeta) pair per body mode shared
        across all DOFs.

    expected_freqs: dict with keys "heave", "pitch", "roll", "warp". Each
    value may be a scalar (treated as f +/- 15 %) or an explicit (lo, hi)
    tuple. Used as DE seed and (for body4dof) as a hard band constraint.

    [fmin, fmax] should bracket the body modes only (typically 2-12 Hz).

    Returns a dict with keys:
      method, params, fn, zeta, mode_labels, mode_shapes (may be None)
    """
    label = run_name or filepath.stem
    log.info("Loading: %s", filepath.name)
    if displacement_mode:
        log.info("  Displacement mode: fitting damperpot displacement PSDs")
        primary = load_displacement_data(filepath, fs)
    else:
        primary = load_force_data(filepath, fs)
    n_samples = primary.shape[1]
    log.info("  %d samples (%.1f s), fit %.1f-%.1f Hz, method=%s",
             n_samples, n_samples / fs, fmin, fmax, method)

    T = build_T(track_front, track_rear)
    freqs, psds = compute_body_psds(primary, T, fs, nperseg=nperseg)
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[fit_mask]
    psds_fit = psds[:, fit_mask]
    meas_norm = np.stack([_normalise(p) for p in psds_fit])

    fr = _normalise_expected_freqs(expected_freqs)
    log.info("  Expected freq bands: %s",
             ", ".join(f"{m}={fr[m][0]:.1f}-{fr[m][1]:.1f}Hz"
                       for m in ("heave", "pitch", "roll", "warp")))

    if method == "lorentzian":
        result = _run_fit_lorentzian(freqs_fit, meas_norm, fr)
    elif method == "body4dof":
        result = _run_fit_body4dof(filepath, fs, nperseg, T, fit_mask, freqs_fit,
                                   meas_norm, psds_fit, fr,
                                   total_mass, wheelbase, pitch_inertia,
                                   roll_inertia, expected_freqs,
                                   displacement_mode)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'lorentzian' or 'body4dof'.")

    _log_modes(result)
    _generate_diagnosis_plot(result, freqs_fit, psds_fit, fmin, fmax,
                             label, output_dir,
                             event=event, output_dpi=output_dpi)
    if show_plots and method == "body4dof":
        # generate_plots produces body-MCK-specific diagnostics
        M, C, K = build_MCK(result["params"])
        generate_plots(freqs, psds, freqs_fit, psds_fit, M, C, K, T,
                       result["fn"], result["zeta"], result["mode_shapes"],
                       fmin, fmax, output_dir=output_dir,
                       event=event, run_name=label, output_dpi=output_dpi)

    return result


def _log_modes(result: dict) -> None:
    """Log a summary table of fitted mode frequencies and damping."""
    log.info("  %-8s %-12s %-10s", "Mode", "Freq [Hz]", "Damp")
    for i in range(len(result["fn"])):
        z = result["zeta"][i]
        z_str = f"{z:.4f}" if not np.isnan(z) else "N/A"
        log.info("  %-8s %-12.3f %-10s",
                 result["mode_labels"][i], result["fn"][i], z_str)
    if result["method"] == "body4dof":
        log.info("  Body params:")
        for i, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
            log.info("    %-6s %-12.1f %s", name, result["params"][i], unit)


def _run_fit_lorentzian(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                        fr: dict) -> dict:
    """Per-DOF Lorentzian sum-of-SDOF fit. Returns a result dict."""
    bands_hp = [fr["heave"][:2], fr["pitch"][:2]]
    bands_rw = [fr["roll"][:2],  fr["warp"][:2]]
    weights_hp = _band_window(freqs_fit, bands_hp)
    weights_rw = _band_window(freqs_fit, bands_rw)
    dof_bands  = [bands_hp, bands_rw, bands_hp, bands_rw]
    dof_weights = [weights_hp, weights_rw, weights_hp, weights_rw]

    log.info("  Fitting per-DOF Lorentzians...")
    params = np.zeros((4, 2, 3))
    for d in range(4):
        params[d] = _fit_lorentz_dof(freqs_fit, meas_norm[d],
                                     dof_bands[d], dof_weights[d])
        labels = _DOF_BAND_LABELS[d]
        log.info("    %s: %s", DOF_LABELS[d],
                 ", ".join(f"{labels[i]}={params[d,i,0]:.2f}Hz z={params[d,i,1]:.3f}"
                           for i in range(2)))

    fn, zeta = _summarise_lorentz(params)
    return {
        "method": "lorentzian",
        "params": params,
        "fn": fn,
        "zeta": zeta,
        "mode_labels": list(_MODE_ORDER),
        "mode_shapes": None,
    }


def _run_fit_body4dof(filepath, fs, nperseg, T, fit_mask, freqs_fit,
                      meas_norm, psds_fit, fr,
                      total_mass, wheelbase, pitch_inertia, roll_inertia,
                      expected_freqs, displacement_mode) -> dict:
    """Original 13-parameter body MCK fit. Returns a result dict."""
    # Optional displacement cross-check for roll/warp TF shape
    disp_norm = None
    if not displacement_mode:
        try:
            x_corner = load_displacement_data(filepath, fs)
            _, disp_psds = compute_body_psds(x_corner, T, fs, nperseg=nperseg)
            disp_norm = np.stack([_normalise(p) for p in disp_psds[:, fit_mask]])
            log.info("  Displacement PSDs loaded for roll/warp TF cross-check")
        except Exception:
            log.info("  No displacement data - roll/warp fit uses force PSDs only")

    bounds_log = np.log(BOUNDS_PHYSICAL)
    de_kwargs = dict(seed=42, popsize=40, polish=True, disp=False,
                     updating="deferred", workers=-1,
                     init="sobol", tol=1e-7, mutation=(0.5, 1.5))

    if expected_freqs:
        hp_x0, rw_x0 = _seed_from_expected_freqs(expected_freqs)
    else:
        hp_x0 = rw_x0 = None
    heave_range = fr["heave"][:2]
    pitch_range = fr["pitch"][:2]
    roll_range  = fr["roll"][:2]
    warp_range  = fr["warp"][:2]

    hp_weights = _band_window(freqs_fit, [heave_range, pitch_range])
    rw_weights = _band_window(freqs_fit, [roll_range, warp_range])

    hp_bounds = bounds_log[_HEAVE_PITCH_IDX]
    log.info("  Optimising heave/pitch subsystem (7 params)...")
    result_hp = differential_evolution(
        _cost_heave_pitch,
        bounds=list(zip(hp_bounds[:, 0], hp_bounds[:, 1])),
        args=(freqs_fit, meas_norm, total_mass, wheelbase, pitch_inertia,
              heave_range, pitch_range, hp_weights),
        x0=hp_x0,
        **de_kwargs,
    )
    log.info("  Heave/pitch done (cost=%.4f)", result_hp.fun)

    rw_bounds = bounds_log[_ROLL_WARP_IDX]
    log.info("  Optimising roll/warp subsystem (6 params)...")
    result_rw = differential_evolution(
        _cost_roll_warp,
        bounds=list(zip(rw_bounds[:, 0], rw_bounds[:, 1])),
        args=(freqs_fit, meas_norm, psds_fit, roll_inertia, disp_norm,
              roll_range, warp_range, rw_weights),
        x0=rw_x0,
        **de_kwargs,
    )
    log.info("  Roll/warp done (cost=%.4f)", result_rw.fun)

    params_fit = np.zeros(13)
    params_fit[_HEAVE_PITCH_IDX] = np.exp(result_hp.x)
    params_fit[_ROLL_WARP_IDX] = np.exp(result_rw.x)
    M, C, K = build_MCK(params_fit)
    fn, zeta, mode_shapes = extract_modes(M, C, K)
    mode_labels = [classify_mode(mode_shapes[:, i]) for i in range(len(fn))]
    return {
        "method": "body4dof",
        "params": params_fit,
        "fn": fn,
        "zeta": zeta,
        "mode_labels": mode_labels,
        "mode_shapes": mode_shapes,
    }


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
                    output_dir: Path = None,
                    output_dpi: int = 300):
    """Overlay normalised best-fit |H(jw)|^2 for multiple runs.

    Args:
        results: List of dicts with keys: name, color, params, fn, zeta,
                 mode_labels, mode_shapes, filepath, method.
        fs, track_front, track_rear, fmin, fmax, nperseg: shared settings.
        event: Event name for the figure header.
        output_dir: Directory to save plots into (a plots/ subfolder is created).
        output_dpi: Output DPI for saved figures.
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
        H_sq = eval_fit_psds(res, freqs_plot)

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
        labels = res["mode_labels"]
        for i in range(len(res["fn"])):
            f_n = res["fn"][i]
            if fmin <= f_n <= fmax:
                mtype = labels[i]
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
            xy=(0.99, 0.945),
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

    # Methods used (if mixed, list them; if uniform, name it)
    methods = sorted({r.get("method", "?") for r in results})
    method_str = methods[0] if len(methods) == 1 else "mixed (" + ",".join(methods) + ")"
    _add_suptitle(fig, event, f"{len(results)} runs",
                  "Modal Fit - Comparison", method=method_str,
                  extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.95))
    fig.savefig(plots_dir / "vibrations_comparison_fit.png",
                dpi=output_dpi, **_SAVE_KW)
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
    _add_suptitle(fig2, event, f"{len(results)} runs",
                  "Measured Body PSDs - Comparison",
                  extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.955))
    fig2.savefig(plots_dir / "vibrations_comparison_psd.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig2)

    # --- Summary table ---
    log.info("  %-20s %-12s %-12s %-12s %-12s", "Run", "Heave [Hz]", "Pitch [Hz]", "Roll [Hz]", "Warp [Hz]")
    for res in results:
        fn = res["fn"]
        modes = res["mode_labels"]
        def find_freq(label):
            for i, m in enumerate(modes):
                if m == label:
                    return f"{fn[i]:.2f}"
            return "-"
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
    parser.add_argument("--displacement-mode", action="store_true",
                        help="Fit to damperpot displacements instead of pushrod forces")
    parser.add_argument("--expected-freqs", type=float, nargs=4, default=None,
                        metavar=("HEAVE", "PITCH", "ROLL", "WARP"),
                        help="Expected modal frequencies [Hz] to seed the DE optimiser.")
    parser.add_argument("--method", choices=("lorentzian", "body4dof"),
                        default="lorentzian",
                        help="Fit method: 'lorentzian' (per-DOF sum-of-SDOF) "
                             "or 'body4dof' (13-param body MCK).")
    parser.add_argument("--event", type=str, default="",
                        help="Event tag for figure headers (e.g. 26P01BCN).")
    parser.add_argument("--output-dpi", type=int, default=300,
                        help="Saved figure DPI (default 300, report-ready).")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        sys.exit(1)

    expected_freqs = None
    if args.expected_freqs is not None:
        h, p, r, w = args.expected_freqs
        expected_freqs = {"heave": h, "pitch": p, "roll": r, "warp": w}

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
        displacement_mode=args.displacement_mode,
        expected_freqs=expected_freqs,
        method=args.method,
        event=args.event,
        output_dpi=args.output_dpi,
    )


if __name__ == "__main__":
    main()
