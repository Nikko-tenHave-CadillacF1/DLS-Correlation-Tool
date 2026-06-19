"""4-DOF Transfer Function Fitting to FPushrod PSDs.

Fits Heave/Pitch/Roll/Warp body modes to measured FPushrod PSDs using
|H(jω)|² shape matching. Preferred entry point: Run_Vibrations.py.
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
FORCE_CHANNELS = ["FPushrodFL", "FPushrodFR", "FPushrodRL", "FPushrodRR"]
DISPLACEMENT_CHANNELS = ["xDamperPotFL", "xDamperPotFR", "xDamperPotRL", "xDamperPotRR"]

PARAM_NAMES = ["mF", "mR", "IrF", "mu", "cFH", "cR", "cRH", "cW",
               "kFH", "kR", "kRH", "kW", "IrR"]
PARAM_UNITS = ["kg", "kg", "kg·m²", "kg", "Ns/m", "Nms/rad", "Ns/m", "Nms/rad",
               "N/m", "Nm/rad", "N/m", "Nm/rad", "kg·m²"]
DOF_LABELS = ["Heave Front (z_F)", "Roll Front (th_F)",
              "Heave Rear (z_R)", "Roll Rear (th_R)"]
# Plot row order: heaves first, rolls second, so front/rear pairs sit
# adjacent for easier visual comparison.
_PLOT_DOF_ORDER = [0, 2, 1, 3]   # [z_F, z_R, th_F, th_R]
_MODE_ORDER = ["Heave", "Pitch", "Roll", "Warp"]

# Expected body-mode natural frequency bands [Hz].
_EXPECTED_FREQS = {
    "heave": (2.0,  6.0),
    "pitch": (6.0, 11.0),
    "roll":  (3.0,  6.0),
    "warp":  (5.0, 11.0),
}

# Parameter bounds for optimisation [lower, upper] in physical units.
BOUNDS_PHYSICAL = np.array([
    [200,    500],       # mF   [kg]
    [300,    600],       # mR   [kg]
    [10,      80],       # IrF  [kg·m²]
    [50,     500],       # mu   [kg]
    [500,   5000],       # cFH  [Ns/m]
    [100,   2000],       # cR   [Nms/rad]
    [500,   8000],       # cRH  [Ns/m]
    [200,   5000],       # cW   [Nms/rad]
    [50000,  500000],    # kFH  [N/m]
    [5000,   80000],     # kR   [Nm/rad]
    [80000,  800000],    # kRH  [N/m]
    [50000,  2000000],   # kW   [Nm/rad]
    [10,      80],       # IrR  [kg·m²]
])


# ======================================================
# DATA LOADING
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
    """Load FPushrod corner forces (rear negated for consistent sign convention)."""
    F_corner = _load_channels(filepath, FORCE_CHANNELS, fs)
    F_corner[2] *= -1  # RL
    F_corner[3] *= -1  # RR
    return F_corner


def load_displacement_data(filepath: Path, fs: float = RESAMPLE_RATE) -> np.ndarray:
    """Load xDamperPot corner displacements."""
    return _load_channels(filepath, DISPLACEMENT_CHANNELS, fs)

# ======================================================
# CORNER-TO-BODY TRANSFORMATION & PSD
# ======================================================
# Corner → body transformation. DOFs: [z_F, θ_F, z_R, θ_R], corners: [FL, FR, RL, RR].
# Track widths cancel out — pitch/roll DOFs are dimensionless (±1), not physical angles.
T_BODY = np.array([
    [0.5,  0.5,  0,    0  ],
    [1,   -1,    0,    0  ],
    [0,    0,    0.5,  0.5],
    [0,    0,    1,   -1  ],
])


def compute_body_psds(F_corner: np.ndarray, T: np.ndarray,
                      fs: float, nperseg: int = 1024):
    """Transform corner forces to body coordinates and compute PSDs."""
    F_body = T @ F_corner
    freqs, psds = signal.welch(F_body, fs, nperseg=nperseg, axis=1)
    return freqs, psds


def compute_coherence_weights(F_corner: np.ndarray, T: np.ndarray,
                              fs: float, nperseg: int = 512,
                              smooth_bins: int = 5) -> np.ndarray:
    """Smoothed amplitude coherence √γ² between front/rear per subsystem → (4, nf).

    Rows 0,2 use √γ²(z_F, z_R); rows 1,3 use √γ²(θ_F, θ_R).
    """
    F_body = T @ F_corner
    _, coh_hp = signal.coherence(F_body[0], F_body[2], fs=fs, nperseg=nperseg)
    _, coh_rw = signal.coherence(F_body[1], F_body[3], fs=fs, nperseg=nperseg)
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins) / smooth_bins
        coh_hp = np.convolve(coh_hp, kernel, mode="same")
        coh_rw = np.convolve(coh_rw, kernel, mode="same")
    amp_hp = np.sqrt(np.clip(coh_hp, 0.0, 1.0))
    amp_rw = np.sqrt(np.clip(coh_rw, 0.0, 1.0))
    return np.stack([amp_hp, amp_rw, amp_hp, amp_rw])


def auto_nperseg(n_samples: int, fs: float, min_averages: int = 50,
                 max_nperseg: int = 4096) -> int:
    """Largest power-of-2 NPERSEG yielding ≥ `min_averages` Welch segments (50% overlap)."""
    limit = min(int(2 * n_samples / (min_averages + 1)), max_nperseg)
    if limit < 64:
        return 64
    nperseg = 1
    while nperseg * 2 <= limit:
        nperseg *= 2
    return nperseg


# ======================================================
# 4-DOF MODEL: M, C, K MATRICES
# ======================================================
def build_MCK(params: np.ndarray):
    """Construct M, C, K from the 13-element parameter vector."""
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
# TRANSFER FUNCTION |H(jω)|²
# ======================================================
def compute_H_mag_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """|H(jω)|² row-sum per output DOF. Shape: [n_dof, nf]."""
    return _compute_H_ij_sq(freqs_hz, M, C, K).sum(axis=1)


def _compute_H_ij_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Vectorised |H_ij(jω)|² → (n_dof, n_dof, nf)."""
    omega = 2.0 * np.pi * freqs_hz                              # (nf,)
    Z = (K[None] - (omega**2)[:, None, None] * M[None]
         + 1j * omega[:, None, None] * C[None])                 # (nf, n, n)
    H = np.linalg.inv(Z)                                        # (nf, n, n)
    return np.transpose(np.abs(H)**2, (1, 2, 0))                # (n, n, nf)


# ======================================================
# COST FUNCTIONS
# ======================================================
def _normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise 1-D array to [0, 1] by its maximum."""
    peak = np.max(arr)
    return arr / peak if peak > 0 else arr


def _scale_model_to_data(measured: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Least-squares scale model to match measured per-DOF.

    For each DOF row: alpha = dot(meas, model) / dot(model, model).
    Returns scaled model array (same shape as input).
    """
    scaled = np.empty_like(model)
    for i in range(model.shape[0]):
        denom = np.dot(model[i], model[i])
        alpha = np.dot(measured[i], model[i]) / denom if denom > 0 else 1.0
        scaled[i] = alpha * model[i]
    return scaled


# Heave/Pitch indices in param vector
_HEAVE_PITCH_IDX = [0, 1, 3, 4, 6, 8, 10]
# Roll/Warp indices
_ROLL_WARP_IDX = [2, 5, 7, 9, 11, 12]


def _normalise_expected_freqs(expected_freqs: dict | None) -> dict:
    """Normalise user-supplied expected_freqs to {mode: (lo, hi, mid)} dict."""
    out = {}
    for mode in _MODE_ORDER:
        key = mode.lower()
        v = (expected_freqs or {}).get(key)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
        elif v is not None:
            f = float(v)
            lo, hi = f * 0.85, f * 1.15
        else:
            lo, hi = _EXPECTED_FREQS[key]
        out[key] = (lo, hi, 0.5 * (lo + hi))
    return out


def _seed_from_expected_freqs(expected_freqs: dict) -> tuple:
    """Derive 13-element DE seed from expected modal frequencies."""
    fr = _normalise_expected_freqs(expected_freqs)
    fH, fP, fR, fW = fr["heave"][2], fr["pitch"][2], fr["roll"][2], fr["warp"][2]
    zH = zP = zR = zW = 0.15   # generic body-mode damping prior
    return _seed_from_modal_fit(fH, fP, fR, fW, zH, zP, zR, zW)


def _seed_from_modal_fit(fH: float, fP: float, fR: float, fW: float,
                         zH: float, zP: float, zR: float, zW: float) -> tuple:
    """Build a 13-element body4dof seed from per-mode (f, ζ) pairs.

    Uses nominal masses/inertias and back-solves the stiffness and damping
    coefficients from each mode's natural frequency and damping ratio. The
    resulting seed lands the DE close to the true basin even on cars where
    bound-midpoint damping seeds would be far off.
    """
    mF, mR, IrF, IrR = 320.0, 450.0, 30.0, 30.0
    mu = BOUNDS_PHYSICAL[3, 0] * 2.0

    omH, omP, omR, omW = (2.0 * np.pi * f for f in (fH, fP, fR, fW))
    kFH = omH**2 * (mF - mu)
    kRH = omP**2 * (mR - mu)
    kR  = omR**2 * IrF
    kW  = (omW**2 * IrF - kR) / 2.0

    cFH = 2.0 * zH * np.sqrt(max(kFH * (mF - mu), 1.0))
    cRH = 2.0 * zP * np.sqrt(max(kRH * (mR - mu), 1.0))
    cR  = 2.0 * zR * np.sqrt(max(kR  * IrF, 1.0))
    cW  = 2.0 * zW * np.sqrt(max(kW  * IrF, 1.0))

    seed = np.array([mF, mR, IrF, mu, cFH, cR, cRH, cW, kFH, kR, kRH, kW, IrR])
    seed = np.clip(seed, BOUNDS_PHYSICAL[:, 0] * 1.001,
                         BOUNDS_PHYSICAL[:, 1] * 0.999)
    log_seed = np.log(seed)
    return log_seed[_HEAVE_PITCH_IDX], log_seed[_ROLL_WARP_IDX]


def _shape_residual(meas_norm: np.ndarray, model: np.ndarray,
                    weights: np.ndarray = None) -> float:
    """√-amplitude-weighted normalised-shape SSE (peak-biased but valley-aware).

    Both ``meas_norm`` and ``model`` are re-normalised to peak=1 internally
    so the residual is shape-only and invariant to caller-side scaling.
    """
    meas = _normalise(meas_norm)
    model_norm = _normalise(model)
    err = np.sqrt(meas) * (model_norm - meas)**2
    if weights is not None:
        err = err * weights
    return float(np.sum(err))


def _expanded_frequency_bounds(freqs_fit: np.ndarray, ranges: list,
                               expansion: float = 1.0,
                               min_margin: float = 0.5) -> list:
    """Frequency bounds wider than expected bands for DE search."""
    fit_lo = float(np.min(freqs_fit))
    fit_hi = float(np.max(freqs_fit))
    out = []
    for lo, hi in ranges:
        width = max(float(hi) - float(lo), 1e-6)
        margin = max(expansion * width, min_margin)
        bound_lo = max(fit_lo, float(lo) - margin)
        bound_hi = min(fit_hi, float(hi) + margin)
        if bound_lo >= bound_hi:
            bound_lo, bound_hi = fit_lo, fit_hi
        out.append((bound_lo, bound_hi))
    return out

def _cost_heave_pitch(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                      meas_norm: np.ndarray,
                      total_mass: float = None, wheelbase: float = None,
                      pitch_inertia: float = None,
                      weights: np.ndarray = None) -> float:
    """Cost for the heave/pitch 2-DOF subsystem (DOFs z_F, z_R)."""
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
    if total_mass is not None:
        cost += 500.0 * nf * ((mF + mR - total_mass) / total_mass)**2
    if pitch_inertia is not None and wheelbase is not None:
        cost += 500.0 * nf * ((mu * wheelbase - pitch_inertia) / pitch_inertia)**2
    return cost


def _cost_roll_warp(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                    meas_norm: np.ndarray, meas_raw: np.ndarray,
                    roll_inertia: float = None,
                    disp_norm: np.ndarray = None,
                    weights: np.ndarray = None) -> float:
    """Cost for the roll/warp 2-DOF subsystem (DOFs θ_F, θ_R)."""
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

    nf = len(freqs_fit)

    if roll_inertia is not None:
        cost += 500.0 * nf * ((IrF + IrR - roll_inertia) / roll_inertia)**2
    return cost


# ======================================================
# LORENTZIAN FIT (shared-pole front/rear)
# ======================================================


def _eval_band_shapes(freqs_hz: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Vectorised SDOF shapes 1/[(ω₀²-ω²)² + (2ζω₀ω)²] → (n_bands, nf)."""
    omega = 2.0 * np.pi * freqs_hz
    omega0 = 2.0 * np.pi * fz[:, 0:1]                # (n_bands, 1)
    zeta = fz[:, 1:2]                                # (n_bands, 1)
    denom = (omega0**2 - omega**2)**2 + (2.0 * zeta * omega0 * omega)**2
    return 1.0 / np.maximum(denom, 1e-30)


def _eval_lorentz_sum(freqs_hz: np.ndarray, dof_params: np.ndarray) -> np.ndarray:
    """Sum of N Lorentzians. dof_params shape (N, 3): (f0, zeta, A) per row."""
    shapes = _eval_band_shapes(freqs_hz, dof_params[:, :2])
    return dof_params[:, 2] @ shapes


def _cost_lorentz_combined(packed: np.ndarray, freqs_fit: np.ndarray,
                           meas_norm: np.ndarray, n_bands: int,
                           n_traces: int, weights: np.ndarray) -> float:
    """N-trace Lorentzian + per-trace noise floor (fully vectorised).

    Packed layout: per band [f0, zeta, log_A_0, ..., log_A_{T-1}],
    followed by a [log_B_0, ..., log_B_{T-1}] tail. Each amplitude scales
    directly into the (peak-normalised) measurement space; each baseline
    absorbs broadband floor (wheel-hop leakage, drift, sensor noise) so
    the Lorentzian widths reflect actual modal damping rather than
    fighting the off-band floor.
    """
    cols = 2 + n_traces
    n_band_params = n_bands * cols
    params = packed[:n_band_params].reshape(n_bands, cols)
    baselines = np.exp(packed[n_band_params:])             # (n_traces,)
    amps = np.exp(params[:, 2:])                            # (n_bands, n_traces)
    shapes = _eval_band_shapes(freqs_fit, params[:, :2])    # (n_bands, nf)
    models = amps.T @ shapes + baselines[:, None]           # (n_traces, nf)

    diff = models - meas_norm
    err = diff * diff
    if weights is not None:
        err = err * weights
    return float(err.sum())


def _peak_pick_x0(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                  ranges: list, n_traces: int) -> np.ndarray:
    """Seed DE from smoothed peak frequencies and half-power damping estimates.

    log-amplitude seed = log(meas_peak / shape_peak), where shape_peak is
    the SDOF Lorentzian gain 1/(2ζω₀²)² at resonance. Per-trace baseline
    is seeded from the 5th-percentile PSD level of that trace.
    """
    cols = 2 + n_traces
    n_band_params = len(ranges) * cols
    x0 = np.zeros(n_band_params + n_traces)
    mean_trace = np.mean(meas_norm, axis=0)
    if len(mean_trace) >= 5:
        kernel = np.ones(5) / 5.0
        mean_smooth = np.convolve(mean_trace, kernel, mode="same")
    else:
        mean_smooth = mean_trace
    for i, (lo, hi) in enumerate(ranges):
        mask = (freqs_fit >= lo) & (freqs_fit <= hi)
        if np.any(mask):
            band_f = freqs_fit[mask]
            band_p = mean_smooth[mask]
            idx = int(np.argmax(band_p))
            f_peak = float(band_f[idx])
            zeta0 = _half_power_zeta(band_f, band_p, idx)
            meas_peak = float(max(band_p[idx], 1e-6))
        else:
            f_peak = 0.5 * (lo + hi)
            zeta0 = 0.10
            meas_peak = 1.0
        omega0 = 2.0 * np.pi * f_peak
        shape_peak = 1.0 / (2.0 * zeta0 * omega0**2)**2
        log_amp = float(np.log(meas_peak / shape_peak))
        base = i * cols
        x0[base] = f_peak
        x0[base + 1] = zeta0
        x0[base + 2:base + cols] = log_amp
    for t in range(n_traces):
        floor_seed = float(max(np.percentile(meas_norm[t], 5.0), 1e-9))
        x0[n_band_params + t] = float(np.log(floor_seed))
    return x0


def _half_power_zeta(freqs: np.ndarray, psd: np.ndarray, peak_idx: int,
                     z_min: float = 0.05, z_max: float = 0.30) -> float:
    """Estimate ζ from the -3 dB bandwidth around `peak_idx` in a smoothed PSD."""
    peak = float(psd[peak_idx])
    if peak <= 0.0:
        return 0.10
    half = 0.5 * peak
    lo_idx = peak_idx
    while lo_idx > 0 and psd[lo_idx] > half:
        lo_idx -= 1
    hi_idx = peak_idx
    while hi_idx < len(psd) - 1 and psd[hi_idx] > half:
        hi_idx += 1
    bw = float(freqs[hi_idx] - freqs[lo_idx])
    f0 = float(freqs[peak_idx])
    if f0 <= 0.0 or bw <= 0.0:
        return 0.10
    return float(np.clip(bw / (2.0 * f0), z_min, z_max))


_ZETA_LO, _ZETA_HI = 0.02, 0.70


def _de_lorentz(bounds, args, x0, seed, popsize=12):
    """DE with project-standard settings (reduced popsize for speed)."""
    return differential_evolution(
        _cost_lorentz_combined, bounds=bounds, args=args,
        x0=x0, seed=seed, polish=True, disp=False,
        updating="immediate", workers=1, popsize=popsize,
        init="sobol", tol=1e-3, mutation=(0.5, 1.5),
    )


def _hits_zeta_bound(x: np.ndarray, n_bands: int, n_traces: int,
                     margin: float = 0.02) -> bool:
    """True if any fitted ζ sits within `margin` of either ζ bound."""
    cols = 2 + n_traces
    n_band_params = n_bands * cols
    zetas = x[:n_band_params].reshape(n_bands, cols)[:, 1]
    span = _ZETA_HI - _ZETA_LO
    return bool(np.any(zetas <= _ZETA_LO + margin * span)
                or np.any(zetas >= _ZETA_HI - margin * span))


def _fit_lorentz_combined(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                          ranges: list, weights: np.ndarray):
    """Fit shared-pole Lorentzians + per-trace baseline to N traces.

    Returns (params, baselines) where params is (N_bands, 2 + n_traces)
    with rows (f0, ζ, A_0, ..., A_{n_traces-1}) sorted by f0, and
    baselines is (n_traces,) of additive floor values.
    """
    from scipy.optimize import minimize

    n_traces = meas_norm.shape[0]
    n_bands = len(ranges)
    freq_bounds = _expanded_frequency_bounds(freqs_fit, ranges)
    bounds = []
    for lo, hi in freq_bounds:
        bounds.append((lo, hi))                          # f0
        bounds.append((_ZETA_LO, _ZETA_HI))              # zeta
        for _ in range(n_traces):
            bounds.append((np.log(1e-6), np.log(1e6)))   # log-amplitude
    # Per-trace baseline upper bound capped at 30 % of that trace's peak.
    # Stops the baseline from absorbing the whole peak on weakly excited
    # bands (which used to let the Lorentzian width run away).
    for t in range(n_traces):
        peak_t = float(np.max(meas_norm[t]))
        base_hi = max(0.3 * peak_t, 1e-9)
        bounds.append((np.log(1e-9), np.log(base_hi)))   # log-baseline
    args = (freqs_fit, meas_norm, n_bands, n_traces, weights)

    x0 = _peak_pick_x0(freqs_fit, meas_norm, ranges, n_traces)
    # Clip seed inside bounds (the baseline floor can exceed 0.3·peak on a
    # noisy trace; minimize would reject an out-of-bounds x0).
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])
    x0 = np.clip(x0, lo_arr, hi_arr)
    local = minimize(_cost_lorentz_combined, x0, args=args,
                     method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 200, "ftol": 1e-9})
    if local.success and not _hits_zeta_bound(local.x, n_bands, n_traces):
        best_x, best_fun = local.x, local.fun
    else:
        de = _de_lorentz(bounds, args, x0=x0, seed=42)
        best_x, best_fun = de.x, de.fun
        if _hits_zeta_bound(best_x, n_bands, n_traces):
            retry = _de_lorentz(bounds, args, x0=None, seed=7)
            if retry.fun < best_fun:
                best_x, best_fun = retry.x, retry.fun

    cols = 2 + n_traces
    n_band_params = n_bands * cols
    params = best_x[:n_band_params].reshape(n_bands, cols).copy()
    params[:, 2:] = np.exp(params[:, 2:])
    baselines = np.exp(best_x[n_band_params:])
    order = np.argsort(params[:, 0])
    return params[order], baselines

def _lorentz_mode_shapes(params: np.ndarray) -> np.ndarray:
    """Infer body-coordinate mode shapes from fitted Lorentzian amplitudes.
    sqrt(amplitude) as participation, signs from nominal mode definitions."""
    shapes = np.zeros((4, 4), dtype=float)
    for i, (_, _, amp_front, amp_rear) in enumerate(params):
        front = np.sqrt(max(float(amp_front), 0.0))
        rear = np.sqrt(max(float(amp_rear), 0.0))
        if i == 0:       # Heave: in-phase
            shapes[:, i] = [front, 0.0, rear, 0.0]
        elif i == 1:     # Pitch: out-of-phase
            shapes[:, i] = [front, 0.0, -rear, 0.0]
        elif i == 2:     # Roll: in-phase
            shapes[:, i] = [0.0, front, 0.0, rear]
        else:            # Warp: out-of-phase
            shapes[:, i] = [0.0, front, 0.0, -rear]
        peak = np.max(np.abs(shapes[:, i]))
        if peak > 0:
            shapes[:, i] /= peak
    return shapes

# ======================================================
# MODEL PSD EVALUATOR (dispatches on fit method)
# ======================================================
def eval_fit_psds(result: dict, freqs_hz: np.ndarray,
                  include_baseline: bool = True) -> np.ndarray:
    """Return (4, nf) un-normalised model PSD per body DOF.

    For the lorentz method, an additive per-trace baseline is included by
    default (so the result matches the measured PSD floor). Pass
    `include_baseline=False` to recover the pure modal transfer-function
    shape.
    """
    method = result["method"]
    params = result["params"]

    if method == "lorentzian_combined":
        baselines = result.get("baselines")
        if baselines is None or not include_baseline:
            baselines = np.zeros(4)
        # Per body DOF: (band-rows, amplitude-column, baseline-index).
        # Heave/pitch rows fit rows 0–1; roll/warp rows fit rows 2–3.
        # Amplitude column 2 holds the "front" trace amp, col 3 the "rear".
        specs = [([0, 1], 2, 0), ([2, 3], 2, 1),
                 ([0, 1], 3, 2), ([2, 3], 3, 3)]
        return np.stack([
            _eval_lorentz_sum(freqs_hz, params[band_rows][:, [0, 1, amp_col]])
            + baselines[base_idx]
            for band_rows, amp_col, base_idx in specs
        ])

    if method == "body4dof":
        M, C, K = build_MCK(params)
        return compute_H_mag_sq(freqs_hz, M, C, K)

    raise ValueError(f"Unknown fit method: {method!r}")


# ======================================================
# MODAL ANALYSIS (body4dof)
# ======================================================
def extract_modes(M: np.ndarray, C: np.ndarray, K: np.ndarray):
    """Extract natural frequencies, damping ratios, and mode shapes via
    state-space eigenvalue decomposition."""
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
# PLOTTING
# ======================================================
_PLOT_FONT = DataPlotter.PLOT_FONT
_GRID_MAJOR = DataPlotter.GRID_STYLE["major"]
_GRID_MINOR = DataPlotter.GRID_STYLE["minor"]
_INK = "#1A1A1A"
_MEAS_COLOR = "#2000BF"
_FIT_COLOR  = "#D70000"
_MODE_COLOR = "#00AA55"
_RESID_COLOR = "#D70000"
_POS_BAR = "#2E86AB"
_NEG_BAR = "#E05263"
_SAVE_KW = dict(pad_inches=0.15, facecolor="white", bbox_inches="tight")


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")


def _add_suptitle(fig, event: str, run_name: str, plot_type: str,
                  method: str = None, extras: str = None) -> None:
    """Two-line figure header: bold title + subtitle."""
    fig.suptitle(plot_type.upper(),
                 fontsize=_PLOT_FONT["figure_title_size"],
                 fontweight="bold", color=_INK, y=0.995)
    sub_parts = [s for s in (event, run_name, f"method={method}" if method else None) if s]
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
        "font.family": font, "font.sans-serif": [_PLOT_FONT["family"]] + _PLOT_FONT["fallback"],
        "axes.titlesize": _PLOT_FONT["title_size"], "axes.titleweight": "bold",
        "axes.labelsize": _PLOT_FONT["label_size"], "axes.labelweight": "bold",
        "axes.edgecolor": _INK, "axes.labelcolor": _INK, "axes.titlecolor": _INK,
        "xtick.color": _INK, "ytick.color": _INK,
        "xtick.labelsize": _PLOT_FONT["tick_size"], "ytick.labelsize": _PLOT_FONT["tick_size"],
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "text.color": _INK, "legend.fontsize": _PLOT_FONT["legend_size"],
        "figure.titlesize": _PLOT_FONT["figure_title_size"], "figure.titleweight": "bold",
    })


def _style_axis(ax, grid_axis="both"):
    ax.grid(True, which="major", axis=grid_axis, **_GRID_MAJOR)
    ax.grid(True, which="minor", axis=grid_axis, **_GRID_MINOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _style_dof_row(ax, dof: int, ylabel_suffix: str = "(PSD)",
                   ylim_zero: bool = True) -> None:
    """Apply the standard 4-row DOF y-axis styling (label, coords, ylim, grid)."""
    suffix = f"\n{ylabel_suffix}" if ylabel_suffix else ""
    ax.set_ylabel(f"{DOF_LABELS[dof]}{suffix}", fontsize=9.5,
                  fontweight="bold", rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.085, 0.5)
    if ylim_zero:
        ax.set_ylim(bottom=0)
    _style_axis(ax, grid_axis="y")


def _vibrations_plots_dir(output_dir: Path | None) -> Path:
    """Resolve and create the vibrations plots directory."""
    base = output_dir if output_dir else Path(".")
    plots_dir = base / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _coherence_weights(coh_fit: np.ndarray | None, floor: float = 0.2):
    """Apply coherence floor to HP/RW weight rows; returns ``(w_hp, w_rw)``."""
    if coh_fit is None:
        return None, None
    return np.maximum(coh_fit[0], floor), np.maximum(coh_fit[1], floor)


def _plot_mode_shape_bars(ax, shape: np.ndarray, tick_labels: list,
                          title: str) -> None:
    """Plot a normalised mode-shape barh on ``ax`` and apply common styling."""
    ax.barh(range(4), shape,
            color=[_POS_BAR if s >= 0 else _NEG_BAR for s in shape],
            edgecolor=_INK, linewidth=0.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(tick_labels)
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color=_INK, linewidth=0.6)
    ax.set_xlim(-1.2, 1.2)
    _style_axis(ax, grid_axis="x")


def _add_legend(ax, loc="upper right"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
        loc=loc, fancybox=True, framealpha=0.92, edgecolor="#3C3C3C",
        borderpad=0.55, handlelength=1.8,
        prop={"family": _PLOT_FONT["family"], "weight": "bold", "size": _PLOT_FONT["legend_size"]},
    )
    legend.get_frame().set_linewidth(1.4)


def generate_plots(result: dict, freqs, psds, freqs_fit, psds_fit, T,
                   fmin, fmax,
                   output_dir: Path = None,
                   event: str = "", run_name: str = "",
                   output_dpi: int = 300):
    """Generate the full per-run vibration figure set."""
    _configure_style()

    plots_dir = _vibrations_plots_dir(output_dir)

    method = result["method"]
    fn = result["fn"]
    zeta = result["zeta"]
    mode_labels = result["mode_labels"]
    mode_shapes = result.get("mode_shapes")
    n_modes = len(fn)
    H_sq_fit = eval_fit_psds(result, freqs_fit)
    safe = _safe_name(run_name) if run_name else "fit"

    # PLOT 1: Measured vs Fitted PSDs (amplitude-scaled)
    H_sq_scaled = _scale_model_to_data(psds_fit, H_sq_fit)
    fig1, axes1 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for row, dof in enumerate(_PLOT_DOF_ORDER):
        ax = axes1[row]
        ax.plot(freqs_fit, psds_fit[dof], color=_MEAS_COLOR,
                linewidth=1.8, alpha=0.85, label="Measured" if row == 0 else None)
        ax.plot(freqs_fit, H_sq_scaled[dof], color=_FIT_COLOR,
                linewidth=1.8, alpha=0.85, label="Fitted" if row == 0 else None)
        for f_n in fn:
            if fmin <= f_n <= fmax:
                ax.axvline(f_n, color=_MODE_COLOR, linestyle="--",
                           linewidth=0.9, alpha=0.7)
        _style_dof_row(ax, dof)
        if row == 0:
            _add_legend(ax)
    axes1[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig1, event, run_name, "Modal Fit - Measured vs Fitted",
                  method=method, extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.955))
    fig1.savefig(plots_dir / f"vibrations_fit_{safe}.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig1)

    # PLOT 2: Full-band PSDs with mode frequencies
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for row, dof in enumerate(_PLOT_DOF_ORDER):
        ax = axes2[row]
        valid = freqs > 0.5
        ax.semilogy(freqs[valid], psds[dof, valid], color=_MEAS_COLOR,
                    linewidth=1.6, alpha=0.85)
        for i, f_n in enumerate(fn):
            mode_label = mode_labels[i]
            ax.axvline(f_n, color=_FIT_COLOR, linestyle="--",
                       linewidth=1.0, alpha=0.7,
                       label=f"{mode_label} {f_n:.2f} Hz" if row == 0 else None)
        _style_dof_row(ax, dof, ylim_zero=False)
        if row == 0:
            _add_legend(ax)
    axes2[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig2, event, run_name, "Body PSDs with Mode Frequencies",
                  method=method)
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.955))
    fig2.savefig(plots_dir / f"vibrations_psd_modes_{safe}.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig2)

    # PLOT 3: Mode shapes (body coordinates)
    if n_modes > 0 and mode_shapes is not None:
        fig3, axes3 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4.6), sharey=True)
        if n_modes == 1:
            axes3 = [axes3]

        dof_short = ["z_F", "th_F", "z_R", "th_R"]
        for i in range(n_modes):
            shape = np.real(mode_shapes[:, i])
            shape = shape / np.max(np.abs(shape))
            _plot_mode_shape_bars(
                axes3[i], shape, dof_short,
                f"{mode_labels[i]}\n{fn[i]:.2f} Hz  z={zeta[i]:.4f}",
            )
        _add_suptitle(fig3, event, run_name, "Mode Shapes - Body Coords",
                      method=method)
        plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.86))
        fig3.savefig(plots_dir / f"vibrations_mode_shapes_body_{safe}.png",
                     dpi=output_dpi, **_SAVE_KW)
        plt.close(fig3)

    # PLOT 4: Transfer function magnitude (normalised; baseline excluded)
    freqs_plot = np.linspace(0.5, fmax + 2.0, 500)
    H_sq_full = eval_fit_psds(result, freqs_plot, include_baseline=False)
    fig4, axes4 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for row, dof in enumerate(_PLOT_DOF_ORDER):
        ax = axes4[row]
        ax.plot(freqs_plot, _normalise(H_sq_full[dof]),
                color=_FIT_COLOR, linewidth=1.8, alpha=0.85)
        for f_n in fn:
            ax.axvline(f_n, color=_MODE_COLOR, linestyle="--",
                       linewidth=0.9, alpha=0.7)
        _style_dof_row(ax, dof, ylabel_suffix="(norm.)")
    axes4[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig4, event, run_name, "Fitted Transfer Function",
                  method=method)
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.955))
    fig4.savefig(plots_dir / f"vibrations_transfer_function_{safe}.png",
                 dpi=output_dpi, **_SAVE_KW)
    plt.close(fig4)

    # PLOT 5: Mode shapes (corner coordinates)
    if n_modes > 0 and mode_shapes is not None:
        T_inv = np.linalg.inv(T)
        fig5, axes5 = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4.6), sharey=True)
        if n_modes == 1:
            axes5 = [axes5]

        corner_labels = ["FL", "FR", "RL", "RR"]
        for i in range(n_modes):
            body_shape = np.real(mode_shapes[:, i])
            corner_shape = T_inv @ body_shape
            corner_shape = corner_shape / np.max(np.abs(corner_shape))
            _plot_mode_shape_bars(
                axes5[i], corner_shape, corner_labels,
                f"{mode_labels[i]}\n{fn[i]:.2f} Hz  z={zeta[i]:.4f}",
            )
        _add_suptitle(fig5, event, run_name, "Mode Shapes - Corner Coords",
                      method=method)
        plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.86))
        fig5.savefig(plots_dir / f"vibrations_mode_shapes_corner_{safe}.png",
                     dpi=output_dpi, **_SAVE_KW)
        plt.close(fig5)

    log.info("  Plots saved to: %s", plots_dir)

def _generate_diagnosis_plot(result: dict, freqs_fit, psds_fit,
                             fmin, fmax, run_name, output_dir,
                             event: str = "", output_dpi: int = 300):
    """Per-run diagnosis figure: measured vs fitted PSDs + residual SSE."""
    _configure_style()
    plots_dir = _vibrations_plots_dir(output_dir)

    H_sq_fit = eval_fit_psds(result, freqs_fit)
    H_sq_scaled = _scale_model_to_data(psds_fit, H_sq_fit)
    fn, mode_labels, method = result["fn"], result["mode_labels"], result["method"]
    zeta = result.get("zeta")
    is_lorentz = (method == "lorentzian_combined")

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5]})

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        ax = axes[row]
        meas = psds_fit[dof]
        model = H_sq_scaled[dof]
        ax.plot(freqs_fit, meas, color=_MEAS_COLOR, linewidth=1.6, alpha=0.85,
                label="Measured" if row == 0 else None)
        ax.plot(freqs_fit, model, color=_FIT_COLOR, linewidth=1.6, alpha=0.85,
                label="Fitted" if row == 0 else None)
        ax.fill_between(freqs_fit, meas, model, color=_FIT_COLOR, alpha=0.12)

        if is_lorentz:
            mode_rows = [0, 1] if dof in (0, 2) else [2, 3]
            mode_names = ["Heave", "Pitch"] if dof in (0, 2) else ["Roll", "Warp"]
            ann_lines = []
            for mr, mname in zip(mode_rows, mode_names):
                f0, z = result["params"][mr, 0], result["params"][mr, 1]
                if fmin <= f0 <= fmax:
                    ax.axvline(f0, color=_MODE_COLOR, linestyle="--",
                               linewidth=0.9, alpha=0.7)
                ann_lines.append(f"{mname}: {f0:.2f} Hz  z={z:.3f}")
            ax.text(0.985, 0.93, "\n".join(ann_lines), transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", family="monospace", color=_INK,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              alpha=0.92, edgecolor="#3C3C3C", linewidth=0.8))
        else:
            for i, f_n in enumerate(fn):
                if not (fmin <= f_n <= fmax):
                    continue
                z_str = f" z={zeta[i]:.3f}" if zeta is not None and not np.isnan(zeta[i]) else ""
                ax.axvline(f_n, color=_MODE_COLOR, linestyle="--", linewidth=0.9, alpha=0.7,
                           label=f"{mode_labels[i]} {f_n:.1f} Hz{z_str}" if row == 0 else None)

        _style_dof_row(ax, dof)
        if row == 0:
            _add_legend(ax, loc="upper left" if is_lorentz else "upper right")

    # Residual row (normalised for scale-independent SSE)
    ax_res = axes[4]
    residual = sum((_normalise(psds_fit[d]) - _normalise(H_sq_fit[d]))**2 for d in range(4))
    ax_res.fill_between(freqs_fit, 0, residual, color=_RESID_COLOR, alpha=0.3)
    ax_res.plot(freqs_fit, residual, color=_RESID_COLOR, linewidth=1.0, alpha=0.7)
    total_sse = float(np.trapezoid(residual, freqs_fit))
    ax_res.text(0.985, 0.92, f"integral SSE = {total_sse:.3f}", transform=ax_res.transAxes,
                fontsize=8.5, fontweight="bold", family="monospace", color=_INK,
                va="top", ha="right")
    ax_res.set_ylabel("Residual\n(SSE)", fontsize=9.5, fontweight="bold",
                      rotation=0, ha="right", va="center")
    ax_res.yaxis.set_label_coords(-0.085, 0.5)
    ax_res.set_ylim(bottom=0)
    ax_res.yaxis.set_major_locator(plt.MaxNLocator(3))
    _style_axis(ax_res, grid_axis="y")
    axes[-1].set_xlabel("Frequency [Hz]")

    _add_suptitle(fig, event, run_name, "Modal Fit - Diagnosis",
                  method=method, extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.955))
    safe_name = _safe_name(run_name)
    fig.savefig(plots_dir / f"vibrations_diag_{safe_name}.png", dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  Diagnosis plot saved: vibrations_diag_%s.png", safe_name)

# ======================================================
# MAIN PIPELINE
# ======================================================
def run_fit(filepath: Path, fs: float = RESAMPLE_RATE,
            fmin: float = 1.0, fmax: float = 12.0,
            nperseg: int | str = 1024, total_mass: float = None,
            wheelbase: float = None, pitch_inertia: float = None,
            roll_inertia: float = None, show_plots: bool = True,
            output_dir: Path = None, run_name: str = None,
            displacement_mode: bool = False,
            expected_freqs: dict = None,
            method: str = "lorentzian_combined",
            event: str = "",
            output_dpi: int = 300) -> dict:
    """Fit a modal model to measured body PSDs.

    nperseg: int or "auto". If "auto", selects the largest power-of-2
        NPERSEG that gives at least 50 Welch segments.

    Returns dict with keys: method, params, fn, zeta, mode_labels, mode_shapes.
    """
    label = run_name or filepath.stem
    log.info("Loading: %s", filepath.name)
    if displacement_mode:
        log.info("  Displacement mode: fitting damperpot displacement PSDs")
        primary = load_displacement_data(filepath, fs)
    else:
        primary = load_force_data(filepath, fs)
    n_samples = primary.shape[1]

    # Resolve NPERSEG
    if nperseg == "auto":
        nperseg = auto_nperseg(n_samples, fs)
        log.info("  Auto NPERSEG: %d (Δf=%.3f Hz, ~%d averages)",
                 nperseg, fs / nperseg,
                 int(2 * n_samples / nperseg - 1))
    log.info("  %d samples (%.1f s), fit %.1f-%.1f Hz, method=%s, nperseg=%d",
             n_samples, n_samples / fs, fmin, fmax, method, nperseg)

    T = T_BODY
    freqs, psds = compute_body_psds(primary, T, fs, nperseg=nperseg)
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[fit_mask]
    psds_fit = psds[:, fit_mask]
    # Light median filter (3-bin kernel) suppresses single-bin spikes
    # (gear-shift transients, FFT artefacts) without blunting peaks whose
    # FWHM is many bins wide. Applied per-DOF on the fit-window slice.
    psds_fit_smooth = np.stack([signal.medfilt(p, kernel_size=3) for p in psds_fit])
    # Global normalisation: divide every DOF by the same scalar (max across
    # all DOFs and bins). Preserves cross-DOF amplitude relationships, so
    # quiet DOFs (e.g. pitch on smooth circuits) cannot inflate the cost
    # surface to dominate the well-excited heave peak.
    global_peak = float(np.max(psds_fit_smooth))
    if global_peak <= 0.0:
        global_peak = 1.0
    meas_norm = psds_fit_smooth / global_peak

    coh_fit = compute_coherence_weights(primary, T, fs, nperseg=nperseg)[:, fit_mask]

    fr = _normalise_expected_freqs(expected_freqs)
    log.info("  Expected freq bands: %s",
             ", ".join(f"{m}={fr[m][0]:.1f}-{fr[m][1]:.1f}Hz"
                       for m in ("heave", "pitch", "roll", "warp")))

    # Damper-pot displacements are mechanically low-passed by the damper,
    # so they are only loaded for the body4dof cross-check.
    disp_norm = None
    if not displacement_mode and method == "body4dof":
        try:
            x_corner = load_displacement_data(filepath, fs)
            _, disp_psds = compute_body_psds(x_corner, T, fs, nperseg=nperseg)
            disp_norm = np.stack([_normalise(p) for p in disp_psds[:, fit_mask]])
            log.info("  Displacement PSDs loaded for roll/warp cross-check")
        except Exception:
            log.info("  No displacement data available")

    if method == "lorentzian_combined":
        result = _run_fit_lorentzian_combined(freqs_fit, meas_norm, fr, coh_fit)
        _log_modes(result)
    elif method == "body4dof":
        result = _run_fit_body4dof(filepath, fs, nperseg, T, fit_mask, freqs_fit,
                                   meas_norm, psds_fit, fr,
                                   total_mass, wheelbase, pitch_inertia,
                                   roll_inertia, expected_freqs,
                                   displacement_mode, disp_norm, coh_fit)
        _log_modes(result)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'lorentzian_combined' "
            "or 'body4dof'."
        )

    # Stash measurement arrays so cross-run plots can put models on physical units.
    result["psds_fit"] = psds_fit
    result["freqs_fit"] = freqs_fit

    _generate_diagnosis_plot(result, freqs_fit, psds_fit, fmin, fmax,
                             label, output_dir,
                             event=event, output_dpi=output_dpi)
    if show_plots:
        generate_plots(result, freqs, psds, freqs_fit, psds_fit, T,
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

def _run_fit_lorentzian_combined(freqs_fit, meas_norm, fr, coh_fit=None):
    bands_hp = [fr["heave"][:2], fr["pitch"][:2]]
    bands_rw = [fr["roll"][:2], fr["warp"][:2]]

    # Coherence weighting only — de-weight genuinely incoherent bins
    # (sensor noise tails, off-mode broadband). No expected-band Gaussian
    # bias: f₀ is already constrained to its expanded band via the search
    # bounds, and biasing the cost on top of that over-constrains the fit.
    weights_hp, weights_rw = _coherence_weights(coh_fit)

    log.info("  Fitting combined Lorentzians...")
    params_hp, base_hp = _fit_lorentz_combined(
        freqs_fit, meas_norm[[0, 2]], bands_hp, weights_hp)
    params_rw, base_rw = _fit_lorentz_combined(
        freqs_fit, meas_norm[[1, 3]], bands_rw, weights_rw)
    params = np.vstack((params_hp, params_rw))
    # baselines indexed by body DOF order [z_F, th_F, z_R, th_R]
    baselines = np.array([base_hp[0], base_rw[0], base_hp[1], base_rw[1]])

    return {
        "method": "lorentzian_combined",
        "params": params,
        "baselines": baselines,
        "fn": params[:, 0],
        "zeta": params[:, 1],
        "mode_labels": list(_MODE_ORDER),
        "mode_shapes": _lorentz_mode_shapes(params),
    }

def _run_fit_body4dof(filepath, fs, nperseg, T, fit_mask, freqs_fit,
                      meas_norm, psds_fit, fr,
                      total_mass, wheelbase, pitch_inertia, roll_inertia,
                      expected_freqs, displacement_mode, disp_norm=None,
                      coh_fit=None) -> dict:
    """Original 13-parameter body MCK fit. Returns a result dict."""
    bounds_log = np.log(BOUNDS_PHYSICAL)
    de_kwargs = dict(seed=42, popsize=40, polish=True, disp=False,
                     updating="deferred", workers=-1,
                     init="sobol", tol=1e-7, mutation=(0.5, 1.5))

    if expected_freqs:
        hp_x0, rw_x0 = _seed_from_expected_freqs(expected_freqs)
    else:
        hp_x0 = rw_x0 = None

    # Refine the seed by first running the cheap lorentzian_combined fit and
    # back-solving (k, c) from each mode's measured (f, ζ). This usually lands
    # the DE much closer to the true basin than the bounds-midpoint priors.
    try:
        lorentz_seed = _run_fit_lorentzian_combined(
            freqs_fit, meas_norm, fr, coh_fit=coh_fit,
        )
        fn = lorentz_seed["fn"]
        zt = lorentz_seed["zeta"]
        if np.all(np.isfinite(fn)) and np.all(np.isfinite(zt)):
            hp_x0, rw_x0 = _seed_from_modal_fit(
                fn[0], fn[1], fn[2], fn[3], zt[0], zt[1], zt[2], zt[3],
            )
            log.info("  Seeded body4dof DE from lorentzian_combined: "
                     "H=%.2fHz/%.3f P=%.2fHz/%.3f R=%.2fHz/%.3f W=%.2fHz/%.3f",
                     fn[0], zt[0], fn[1], zt[1], fn[2], zt[2], fn[3], zt[3])
    except Exception as exc:  # noqa: BLE001 - seed is optional
        log.warning("  Lorentz-based seed failed (%s); falling back to defaults", exc)

    # Coherence weighting only — same rationale as lorentzian_combined.
    hp_weights, rw_weights = _coherence_weights(coh_fit)

    hp_bounds = bounds_log[_HEAVE_PITCH_IDX]
    log.info("  Optimising heave/pitch subsystem (7 params)...")
    result_hp = differential_evolution(
        _cost_heave_pitch,
        bounds=list(zip(hp_bounds[:, 0], hp_bounds[:, 1])),
        args=(freqs_fit, meas_norm, total_mass, wheelbase, pitch_inertia,
              hp_weights),
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
              rw_weights),
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
# COMPARISON PLOT
# ======================================================
_DEFAULT_COLORS = [
    "#FF8000", "#2000BF", "#D70000", "#008CFF",
    "#00CC88", "#CC0066", "#FFD700", "#4C00BF",
]


def plot_comparison(results: list, fs: float = 100.0,
                    fmin: float = 1.0, fmax: float = 19.0,
                    nperseg: int | str = 1024, event: str = "",
                    output_dir: Path = None, output_dpi: int = 300):
    """Overlay normalised best-fit |H(jω)|² for multiple runs."""
    _configure_style()
    if not results:
        log.warning("No results to compare.")
        return

    plots_dir = _vibrations_plots_dir(output_dir)
    freqs_plot = np.linspace(fmin, fmax, 500)
    T = T_BODY

    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
    from collections import OrderedDict

    # Plot A: fitted PSD comparison (physical PSD units; model scaled per-DOF
    # to its own run's measurement, so absolute amplitudes are comparable
    # across runs sharing the same channel set).
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        meas_fit = res.get("psds_fit")
        freqs_fit = res.get("freqs_fit")
        if meas_fit is None or freqs_fit is None:
            H_sq_plot = eval_fit_psds(res, freqs_plot)
        else:
            H_sq_at_fit = eval_fit_psds(res, freqs_fit)
            scale = np.empty(meas_fit.shape[0])
            for d in range(meas_fit.shape[0]):
                denom = float(np.dot(H_sq_at_fit[d], H_sq_at_fit[d]))
                scale[d] = (float(np.dot(meas_fit[d], H_sq_at_fit[d])) / denom
                            if denom > 0 else 1.0)
            H_sq_plot = eval_fit_psds(res, freqs_plot) * scale[:, None]
        for row, dof in enumerate(_PLOT_DOF_ORDER):
            axes[row].plot(freqs_plot, H_sq_plot[dof], color=color,
                           linewidth=1.8, alpha=0.85, label=res["name"])

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        _style_dof_row(axes[row], dof, ylabel_suffix="")
    axes[-1].set_xlabel("Frequency [Hz]")

    # Mode information box
    mode_groups = OrderedDict()
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        for i, f_n in enumerate(res["fn"]):
            if fmin <= f_n <= fmax:
                mode_groups.setdefault(res["mode_labels"][i], []).append(
                    {"f": f_n, "z": res["zeta"][i], "color": color, "name": res["name"]})

    if mode_groups:
        legend_fs = _PLOT_FONT["legend_size"]
        n_entries = sum(1 + len(e) for e in mode_groups.values())
        info_fs = legend_fs if n_entries <= 12 else (legend_fs - 1 if n_entries <= 16 else legend_fs - 2)
        max_name_len = max(len(e["name"]) for entries in mode_groups.values() for e in entries)
        info_lines = []
        for mode_name, entries in mode_groups.items():
            info_lines.append((mode_name, _INK))
            for entry in entries:
                z_str = f"{entry['z']:.3f}" if not np.isnan(entry["z"]) else "n/a"
                info_lines.append((
                    f"  {entry['name'].ljust(max_name_len)}  f={entry['f']:.2f} Hz  z={z_str}",
                    entry["color"]))
        text_areas = [TextArea(text, textprops=dict(color=c, fontsize=info_fs,
                      fontweight="bold", family="monospace")) for text, c in info_lines]
        vpacker = VPacker(children=text_areas, pad=6, sep=2)
        # Anchor inside the right edge to avoid being clipped by
        # ``bbox_inches="tight"`` (figure-anchored artists are not always
        # picked up by tight-bbox calculation).
        ab = AnnotationBbox(vpacker, xy=(0.97, 0.945), xycoords="figure fraction",
                            box_alignment=(1.0, 1.0), frameon=True, pad=0,
                            bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white",
                                           alpha=0.92, edgecolor="#3C3C3C", linewidth=1.4))
        ab.set_zorder(10)
        fig.add_artist(ab)

    methods = sorted({r.get("method", "?") for r in results})
    method_str = methods[0] if len(methods) == 1 else "mixed (" + ",".join(methods) + ")"
    _add_suptitle(fig, event, f"{len(results)} runs", "Modal Fit - Comparison",
                  method=method_str, extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.95))
    fig.savefig(plots_dir / "vibrations_comparison_fit.png", dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)

    # Plot B: measured PSD comparison (raw physical PSD units)
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        F_corner = load_force_data(res["filepath"], fs)
        nps = auto_nperseg(F_corner.shape[1], fs) if nperseg == "auto" else nperseg
        freqs, psds = compute_body_psds(F_corner, T, fs, nperseg=nps)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        for row, dof in enumerate(_PLOT_DOF_ORDER):
            axes2[row].plot(freqs[freq_mask], psds[dof, freq_mask],
                            color=color, linewidth=1.6, alpha=0.85, label=res["name"])

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        _style_dof_row(axes2[row], dof, ylabel_suffix="")
        if row == 0:
            _add_legend(axes2[row])
    axes2[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig2, event, f"{len(results)} runs", "Measured Body PSDs - Comparison",
                  extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.955))
    fig2.savefig(plots_dir / "vibrations_comparison_psd.png", dpi=output_dpi, **_SAVE_KW)
    plt.close(fig2)

    # Summary table
    log.info("  %-20s %-12s %-12s %-12s %-12s", "Run", "Heave [Hz]", "Pitch [Hz]", "Roll [Hz]", "Warp [Hz]")
    for res in results:
        fn, modes = res["fn"], res["mode_labels"]
        def find_freq(label):
            for i, m in enumerate(modes):
                if m == label:
                    return f"{fn[i]:.2f}"
            return "-"
        log.info("  %-20s %-12s %-12s %-12s %-12s",
                 res["name"], find_freq("Heave"), find_freq("Pitch"),
                 find_freq("Roll"), find_freq("Warp"))
    log.info("  Plots saved to: %s", plots_dir)

# ======================================================
# CLI
# ======================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="4-DOF body-mode fitting.")
    parser.add_argument("data_file", help="Path to CSV data file with FPushrod channels.")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=19.0)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--total-mass", type=float, default=None)
    parser.add_argument("--wheelbase", type=float, default=None)
    parser.add_argument("--pitch-inertia", type=float, default=None)
    parser.add_argument("--roll-inertia", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--displacement-mode", action="store_true")
    parser.add_argument("--expected-freqs", type=float, nargs=4, default=None,
                        metavar=("HEAVE", "PITCH", "ROLL", "WARP"))
    parser.add_argument("--method", choices=("lorentzian_combined", "body4dof"),
                        default="lorentzian_combined")
    parser.add_argument("--event", type=str, default="")
    parser.add_argument("--output-dpi", type=int, default=300)
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
        filepath=data_path, fs=args.fs,
        fmin=args.fmin, fmax=args.fmax, nperseg=args.nperseg,
        total_mass=args.total_mass, wheelbase=args.wheelbase,
        pitch_inertia=args.pitch_inertia, roll_inertia=args.roll_inertia,
        show_plots=not args.no_plots, displacement_mode=args.displacement_mode,
        expected_freqs=expected_freqs, method=args.method,
        event=args.event, output_dpi=args.output_dpi,
    )


if __name__ == "__main__":
    main()
