"""Full 4-DOF body-modal MCK fit pipeline.

Fits a 13-parameter mass / inertia / damping / stiffness model to the body
PSDs by splitting the problem into a 7-param heave/pitch subsystem and a
6-param roll/warp subsystem (linked through the unsprung-mass coupling
parameter). The differential-evolution basin is seeded from the cheaper
:func:`engine.vibrations_lorentz.run_fit` solution when available.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from engine.logger import log
from .vibrations_lorentz import run_fit as run_fit_lorentz

PARAM_NAMES = ["mF", "mR", "IrF", "mu", "cFH", "cR", "cRH", "cW",
               "kFH", "kR", "kRH", "kW", "IrR"]
PARAM_UNITS = ["kg", "kg", "kg·m²", "kg", "Ns/m", "Nms/rad", "Ns/m", "Nms/rad",
               "N/m", "Nm/rad", "N/m", "Nm/rad", "kg·m²"]

BOUNDS_PHYSICAL = np.array([
    [200,    500],
    [300,    600],
    [10,      80],
    [50,     500],
    [500,   5000],
    [100,   2000],
    [500,   8000],
    [200,   5000],
    [50000,  500000],
    [5000,   80000],
    [80000,  800000],
    [50000,  2000000],
    [10,      80],
])

_HEAVE_PITCH_IDX = [0, 1, 3, 4, 6, 8, 10]
_ROLL_WARP_IDX = [2, 5, 7, 9, 11, 12]


def build_MCK(params: np.ndarray):
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


def _compute_H_ij_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * freqs_hz
    Z = (K[None] - (omega**2)[:, None, None] * M[None]
         + 1j * omega[:, None, None] * C[None])
    H = np.linalg.inv(Z)
    return np.transpose(np.abs(H)**2, (1, 2, 0))


def compute_H_mag_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    return _compute_H_ij_sq(freqs_hz, M, C, K).sum(axis=1)


def _normalise(arr: np.ndarray) -> np.ndarray:
    peak = np.max(arr)
    return arr / peak if peak > 0 else arr


def _shape_residual(meas_norm: np.ndarray, model: np.ndarray,
                    weights: np.ndarray | None = None) -> float:
    meas = _normalise(meas_norm)
    model_norm = _normalise(model)
    err = np.sqrt(meas) * (model_norm - meas)**2
    if weights is not None:
        err = err * weights
    return float(np.sum(err))


def _seed_from_modal_fit(fH: float, fP: float, fR: float, fW: float,
                         zH: float, zP: float, zR: float, zW: float) -> tuple:
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


def _seed_from_fr(fr: dict) -> tuple:
    fH, fP, fR, fW = fr["heave"][2], fr["pitch"][2], fr["roll"][2], fr["warp"][2]
    return _seed_from_modal_fit(fH, fP, fR, fW, 0.15, 0.15, 0.15, 0.15)


def _cost_heave_pitch(log_sub_params: np.ndarray, freqs_fit: np.ndarray,
                      meas_norm: np.ndarray,
                      total_mass: float = None, wheelbase: float = None,
                      pitch_inertia: float = None,
                      weights: np.ndarray = None) -> float:
    mF, mR, mu, cFH, cRH, kFH, kRH = np.exp(log_sub_params)
    M_hp = np.array([[mF - mu, mu], [mu, mR - mu]])
    if np.any(np.linalg.eigvalsh(M_hp) <= 0):
        return 1e15
    C_hp = np.diag([cFH, cRH])
    K_hp = np.diag([kFH, kRH])
    H_sq = _compute_H_ij_sq(freqs_fit, M_hp, C_hp, K_hp).sum(axis=1)
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
    IrF, cR, cW, kR, kW, IrR = np.exp(log_sub_params)
    M_rw = np.diag([IrF, IrR])
    C_rw = np.array([[cR + cW, -cW], [-cW, cR + cW]])
    K_rw = np.array([[kR + kW, -kW], [-kW, kR + kW]])
    H_ij_sq = _compute_H_ij_sq(freqs_fit, M_rw, C_rw, K_rw)
    H_sq = H_ij_sq.sum(axis=1)
    cost = (_shape_residual(meas_norm[1], H_sq[0], weights)
            + _shape_residual(meas_norm[3], H_sq[1], weights))
    if disp_norm is not None:
        pred_thf = H_ij_sq[0, 0] * meas_raw[1] + H_ij_sq[0, 1] * meas_raw[3]
        pred_thr = H_ij_sq[1, 0] * meas_raw[1] + H_ij_sq[1, 1] * meas_raw[3]
        cost += _shape_residual(disp_norm[1], pred_thf, weights)
        cost += _shape_residual(disp_norm[3], pred_thr, weights)
    nf = len(freqs_fit)
    if roll_inertia is not None:
        cost += 500.0 * nf * ((IrF + IrR - roll_inertia) / roll_inertia)**2
    return cost


def extract_modes(M: np.ndarray, C: np.ndarray, K: np.ndarray):
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
    heave_content = np.abs(shape[0]) + np.abs(shape[2])
    roll_content = np.abs(shape[1]) + np.abs(shape[3])
    if heave_content > roll_content:
        phase_diff = np.angle(shape[0] * np.conj(shape[2]))
        return "Heave" if abs(phase_diff) < np.pi / 2 else "Pitch"
    else:
        phase_diff = np.angle(shape[1] * np.conj(shape[3]))
        return "Roll" if abs(phase_diff) < np.pi / 2 else "Warp"


def _coherence_weights(coh_fit: np.ndarray | None, floor: float = 0.2):
    if coh_fit is None:
        return None, None
    return np.maximum(coh_fit[0], floor), np.maximum(coh_fit[1], floor)


def run_fit(freqs_fit: np.ndarray, meas_norm: np.ndarray,
            psds_fit: np.ndarray, fr: dict,
            total_mass: float | None = None,
            wheelbase: float | None = None,
            pitch_inertia: float | None = None,
            roll_inertia: float | None = None,
            disp_norm: np.ndarray | None = None,
            coh_fit: np.ndarray | None = None) -> dict:
    """13-parameter MCK fit. Seeded from `vibrations_lorentz.run_fit`."""
    bounds_log = np.log(BOUNDS_PHYSICAL)
    de_kwargs = dict(seed=42, popsize=40, polish=True, disp=False,
                     updating="deferred", workers=-1,
                     init="sobol", tol=1e-7, mutation=(0.5, 1.5))
    hp_x0, rw_x0 = _seed_from_fr(fr)
    try:
        lorentz_seed = run_fit_lorentz(freqs_fit, meas_norm, fr, coh_fit=coh_fit)
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
    hp_weights, rw_weights = _coherence_weights(coh_fit)
    hp_bounds = bounds_log[_HEAVE_PITCH_IDX]
    log.info("  Optimising heave/pitch subsystem (7 params)...")
    result_hp = differential_evolution(
        _cost_heave_pitch,
        bounds=list(zip(hp_bounds[:, 0], hp_bounds[:, 1])),
        args=(freqs_fit, meas_norm, total_mass, wheelbase, pitch_inertia,
              hp_weights),
        x0=hp_x0, **de_kwargs,
    )
    log.info("  Heave/pitch done (cost=%.4f)", result_hp.fun)
    rw_bounds = bounds_log[_ROLL_WARP_IDX]
    log.info("  Optimising roll/warp subsystem (6 params)...")
    result_rw = differential_evolution(
        _cost_roll_warp,
        bounds=list(zip(rw_bounds[:, 0], rw_bounds[:, 1])),
        args=(freqs_fit, meas_norm, psds_fit, roll_inertia, disp_norm,
              rw_weights),
        x0=rw_x0, **de_kwargs,
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
        "sigma_fn": np.full(len(fn), np.nan, dtype=float),
        "sigma_zeta": np.full(len(zeta), np.nan, dtype=float),
        "r_squared": (float("nan"), float("nan")),
        "mode_labels": mode_labels,
        "mode_shapes": mode_shapes,
    }


def eval_psds(result: dict, freqs_hz: np.ndarray) -> np.ndarray:
    """Evaluate the fitted MCK transfer-function-magnitude PSDs over freqs_hz."""
    M, C, K = build_MCK(result["params"])
    return compute_H_mag_sq(freqs_hz, M, C, K)
