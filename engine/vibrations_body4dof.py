"""Two-axle modal fit with absolute suspension rates.

The car is modelled as two axles moving in heave and roll. The front and rear
axle of one coordinate form a 2-DOF subsystem:

    heave: [z_F, z_R]    in phase -> Heave, out of phase -> Pitch
    roll:  [th_F, th_R]  in phase -> Roll,  out of phase -> Warp

Per subsystem, with axle motion q and suspension travel x (compression
positive), the axles obey

    M q'' = F,    F = K x + C x'

where K = diag(k_F, k_R) and C = diag(c_F, c_R) are the axle suspension rate
and damping, and M = [[m_F, m_FR], [m_FR, m_R]] holds the axle masses and their
front-rear coupling. The seven parameters are fitted in one weighted
least-squares problem to two measurements:

1. The pushrod-force / damper-travel response H = S_xF / S_xx, modelled as
   k + j w c per axle and weighted by its coherence-based error -> k, c.
2. The subsystem's two modal poles (f0, zeta) from the Lorentzian fit of the
   force PSDs, weighted by their sigmas -> M. The poles fix only |m_FR|; its
   sign comes from the measured front-to-rear force phase at the modes.

Natural frequencies, damping ratios and mode shapes are the eigen-solution of
the fitted model; each mode is labelled by the front/rear phase of its shape.
No parameter is fixed, so implausible masses or rates flag a poor fit.

Coordinates follow ``T_BODY`` (heave = corner mean, roll = left - right), so all
values are per-corner equivalents in pushrod force [N] and damper travel [mm]:
k [N/mm], c [N.s/mm], m [kg].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from engine.logger import log

MODE_ORDER = ("Heave", "Pitch", "Roll", "Warp")

# (name, body-DOF rows [front, rear], (in-phase label, out-of-phase label))
SUBSYSTEMS = (
    ("heave", (0, 2), ("Heave", "Pitch")),
    ("roll", (1, 3), ("Roll", "Warp")),
)

PARAM_UNITS = {"k": "N/mm", "c": "N.s/mm", "m": "kg"}

_KG = 1e3  # mass unit N.s^2/mm in kg
_RHO_MAX = 0.95  # bound on m_FR / sqrt(m_F m_R); keeps M positive definite
_LOG_RANGE = np.log(1e3)
_COH2_RANGE = (0.01, 0.99)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


def _unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit vector ``[log k_F, log k_R, log c_F, log c_R, log m_F, log m_R, rho]`` -> (k, c, M)."""
    k, c = np.exp(p[0:2]), np.exp(p[2:4])
    m_f, m_r = np.exp(p[4:6])
    m_fr = p[6] * np.sqrt(m_f * m_r)
    return k, c, np.array([[m_f, m_fr], [m_fr, m_r]])


def _matrices(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M, C, K) of fit vector ``p``."""
    k, c, M = _unpack(p)
    return M, np.diag(c), np.diag(k)


def modes(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Damped modes sorted by frequency: ``fn`` [Hz], ``zeta`` and shapes (columns) with front = 1.

    Overdamped modes are dropped, so fewer than ``len(M)`` modes may be returned.
    """
    n = M.shape[0]
    m_inv = np.linalg.inv(M)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-m_inv @ K, -m_inv @ C]])
    lam, vec = np.linalg.eig(A)
    keep = lam.imag > 0
    lam, vec = lam[keep], vec[:n, keep]
    order = np.argsort(np.abs(lam))
    lam, vec = lam[order], vec[:, order]
    wn = np.abs(lam)
    with np.errstate(divide="ignore", invalid="ignore"):  # an uncoupled mode can have no front motion
        return wn / (2.0 * np.pi), -lam.real / wn, vec / vec[0]


def _force_response(M: np.ndarray, C: np.ndarray, K: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Axle force per road displacement, (nf, 2, 2): F = M q'' with (K - w^2 M + j w C) q = (K + j w C) u."""
    D = K[None] + 1j * w[:, None, None] * C[None]
    Z = D - (w**2)[:, None, None] * M[None]
    return (w**2)[:, None, None] * (M @ np.linalg.solve(Z, D))


def _force_phase(p: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Model front-to-rear force phase [deg] at ``fn``, for independent equal road input per axle."""
    G = _force_response(*_matrices(p), 2.0 * np.pi * fn)
    return np.degrees(np.angle(np.sum(np.conj(G[:, 0, :]) * G[:, 1, :], axis=1)))


def _residuals(
    p: np.ndarray,
    w: np.ndarray,
    frf: np.ndarray,
    sigma: np.ndarray,
    f_meas: np.ndarray,
    z_meas: np.ndarray,
    sf: np.ndarray,
    sz: np.ndarray,
) -> np.ndarray:
    """FRF misfit of both axles (real and imaginary parts) and pole misfit, each in units of its sigma."""
    k, c, _ = _unpack(p)
    err = (k[:, None] + 1j * w * c[:, None] - frf) / sigma
    fn, zeta, _ = modes(*_matrices(p))
    if fn.size < 2:
        pole = np.full(4, 1e3)  # overdamped: no pole pair to compare
    else:
        pole = np.concatenate([(fn - f_meas) / sf, (zeta - z_meas) / sz])
    return np.concatenate([err.real.ravel(), err.imag.ravel(), pole])


def _summary(p: np.ndarray) -> np.ndarray:
    """``[fn (2), zeta (2), k (2), c (2), m_F, m_R, m_FR]`` of fit vector ``p``, modes by frequency, masses in kg."""
    k, c, M = _unpack(p)
    fn, zeta, _ = modes(*_matrices(p))
    return np.concatenate([fn, zeta, k, c, _KG * np.array([M[0, 0], M[1, 1], M[0, 1]])])


def _propagate(p: np.ndarray, cov: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """1-sigma of :func:`_summary` from the fit covariance, by central differences."""
    jac = np.column_stack([(_summary(p + d) - _summary(p - d)) / (2.0 * h) for d in h * np.eye(p.size)])
    return np.sqrt(np.abs(np.diag(jac @ cov @ jac.T)))


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------


def _fit_pair(
    freqs: np.ndarray,
    frf: np.ndarray,
    coh2: np.ndarray,
    n_avg: int,
    f_meas: np.ndarray,
    z_meas: np.ndarray,
    sf: np.ndarray,
    sz: np.ndarray,
    force_csd: np.ndarray,
    labels: tuple[str, str],
) -> dict | None:
    """Fit one front/rear subsystem. Returns modes in (in-phase, out-of-phase) order, or None."""
    w = 2.0 * np.pi * freqs
    coh2 = np.clip(coh2, *_COH2_RANGE)
    # H1 random error per real component (Bendat & Piersol).
    sigma = np.maximum(np.abs(frf) * np.sqrt((1.0 - coh2) / (2.0 * n_avg * coh2)), 1e-12)
    order = np.argsort(f_meas)
    f_meas, z_meas, sf, sz = f_meas[order], z_meas[order], sf[order], sz[order]

    # Seeds: k and c per axle by weighted linear least squares, masses from the poles.
    wt = sigma**-2
    k0 = np.maximum(np.sum(wt * frf.real, axis=1) / np.sum(wt, axis=1), 1e-3)
    c0 = np.maximum(np.sum(wt * w * frf.imag, axis=1) / np.sum(wt * w**2, axis=1), 1e-6)
    # Inflate each axle's FRF sigma by its scatter about constant k, c so a poorly
    # described FRF cannot outweigh the poles.
    seed_err = (k0[:, None] + 1j * w * c0[:, None] - frf) / sigma
    sigma = sigma * np.sqrt(np.maximum(np.mean(np.abs(seed_err) ** 2, axis=1, keepdims=True) / 2.0, 1.0))
    w_poles = 2.0 * np.pi * f_meas
    args = (w, frf, sigma, f_meas, z_meas, sf, sz)
    best = None
    for w_f, w_r in (w_poles, w_poles[::-1]):
        for rho in (-0.5, 0.0, 0.5):
            x0 = np.r_[np.log(k0), np.log(c0), np.log(k0[0] / w_f**2), np.log(k0[1] / w_r**2), rho]
            # Rates and masses may roam three orders of magnitude from their seeds.
            lo = np.r_[x0[:6] - _LOG_RANGE, -_RHO_MAX]
            hi = np.r_[x0[:6] + _LOG_RANGE, _RHO_MAX]
            fit = least_squares(_residuals, x0, bounds=(lo, hi), args=args, x_scale="jac")
            if best is None or fit.cost < best.cost:
                best = fit

    fn, _, _ = modes(*_matrices(best.x))
    if fn.size < 2:
        log.warning("    %s/%s: fitted model is overdamped; no modes extracted.", *labels)
        return None

    # Poles and rates fix only |m_FR|. Its sign decides which mode is in phase,
    # so take it from the measured front-to-rear force phase at the modes.
    phase_meas = np.degrees(np.angle(force_csd[np.abs(freqs[:, None] - fn[None, :]).argmin(axis=0)]))
    flip = np.r_[np.ones(6), -1.0]
    candidates = [best.x, best.x * flip]
    agreement = [np.sum(np.cos(np.radians(_force_phase(q, fn) - phase_meas))) for q in candidates]
    p = candidates[int(np.argmax(agreement))]
    k, c, M = _unpack(p)
    fn, zeta, shapes = modes(*_matrices(p))
    phase_model = _force_phase(p, fn)

    # Covariance inflated by the reduced chi^2 when the model under-fits.
    chi2 = 2.0 * best.cost / max(best.fun.size - p.size, 1)
    cov = np.linalg.pinv(best.jac.T @ best.jac) * max(chi2, 1.0)
    if p is not best.x:
        cov = cov * np.outer(flip, flip)
    try:
        sig = _propagate(p, cov)
    except ValueError:  # a perturbation turned a mode overdamped
        sig = np.full(11, np.nan)

    model = k[:, None] + 1j * w * c[:, None]
    ss_tot = float(np.sum(np.abs(frf - frf.mean(axis=1, keepdims=True)) ** 2))
    r_squared = 1.0 - float(np.sum(np.abs(frf - model) ** 2)) / ss_tot if ss_tot > 0.0 else float("nan")

    in_phase = int(np.argmax(np.cos(np.angle(shapes[1]))))
    pick = [in_phase, 1 - in_phase]
    for side, c_axle in zip(("front", "rear"), c):
        if c_axle < 1e-3:
            log.warning("    %s/%s %s: no damping in force/travel response (Im(H) <= 0); check damper pot.", *labels, side)
    for label, i in zip(labels, pick):
        if (abs(phase_model[i]) < 90.0) != (abs(phase_meas[i]) < 90.0):
            log.warning(
                "    %s: model front-rear phase %+.0f deg disagrees with measured %+.0f deg.",
                label,
                phase_model[i],
                phase_meas[i],
            )
    return {
        "fn": fn[pick],
        "zeta": zeta[pick],
        "sigma_fn": sig[0:2][pick],
        "sigma_zeta": sig[2:4][pick],
        "phase_model": phase_model[pick],
        "phase_meas": phase_meas[pick],
        "shapes": shapes[:, pick],
        "k": k,
        "c": c,
        "sigma_k": sig[4:6],
        "sigma_c": sig[6:8],
        "mass": _KG * np.array([M[0, 0], M[1, 1], M[0, 1]]),
        "sigma_mass": sig[8:11],
        "matrices": (M, np.diag(c), np.diag(k)),
        "r_squared": r_squared,
    }


def _pole_sigma(sigma: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Measured pole sigmas; 10 % of the value where the Lorentzian fit gave none."""
    sigma = np.asarray(sigma, dtype=float)
    return np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 0.1 * np.asarray(value, dtype=float))


def run_fit(
    freqs_fit: np.ndarray,
    frf: np.ndarray,
    coh2: np.ndarray,
    n_avg: int,
    poles: dict,
    force_csd: np.ndarray,
) -> dict:
    """Two-axle fit of the heave and roll subsystems.

    Parameters
    ----------
    freqs_fit : (nf,) array
        Frequency bins (Hz) of the fit window, all > 0.
    frf : (4, nf) complex array
        Pushrod force / damper travel response ``S_xF / S_xx`` per body DOF
        ``[z_F, theta_F, z_R, theta_R]``, both compression positive.
    coh2 : (4, nf) array
        Squared coherence between damper travel and pushrod force.
    n_avg : int
        Welch segments averaged into the spectra.
    poles : dict
        Measured poles in ``MODE_ORDER``: ``fn``, ``zeta``, ``sigma_fn``,
        ``sigma_zeta`` (a Lorentzian fit result).
    force_csd : (2, nf) complex array
        Front-to-rear force cross-spectra ``[S(z_F, z_R), S(theta_F, theta_R)]``.

    Returns
    -------
    dict with keys: method, fn, zeta, sigma_fn, sigma_zeta (4,) in
        ``MODE_ORDER``; mode_labels; mode_shapes (4, 4) complex, one column per
        mode, front = 1; phase_model, phase_meas (4,) front-rear force phase
        [deg]; axle_k, axle_c, sigma_axle_k, sigma_axle_c (4,) per body DOF;
        axle_params, axle_param_sigmas (named k/c/m values); matrices
        (per-subsystem M, C, K); poles_meas; frf; frf_coh2;
        r_squared (heave, roll) of the FRF fit.
    """
    n_avg = max(int(n_avg), 1)
    out = {key: np.full(4, np.nan) for key in ("fn", "zeta", "sigma_fn", "sigma_zeta", "phase_model", "phase_meas")}
    axle = {key: np.full(4, np.nan) for key in ("k", "c", "sigma_k", "sigma_c")}
    mode_shapes = np.zeros((4, 4), dtype=complex)
    axle_params: dict[str, float] = {}
    axle_param_sigmas: dict[str, float] = {}
    matrices: dict[str, tuple] = {}
    r_squared = []
    for s, (name, rows, labels) in enumerate(SUBSYSTEMS):
        log.info("  Fitting %s axles (%s / %s)...", name, *labels)
        idx = [2 * s, 2 * s + 1]
        f_meas = np.asarray(poles["fn"], dtype=float)[idx]
        z_meas = np.asarray(poles["zeta"], dtype=float)[idx]
        sub = _fit_pair(
            freqs_fit,
            frf[list(rows)],
            coh2[list(rows)],
            n_avg,
            f_meas,
            z_meas,
            _pole_sigma(np.asarray(poles["sigma_fn"])[idx], f_meas),
            _pole_sigma(np.asarray(poles["sigma_zeta"])[idx], z_meas),
            force_csd[s],
            labels,
        )
        if sub is None:
            r_squared.append(float("nan"))
            mode_shapes[:, idx] = np.nan
            continue
        r_squared.append(sub["r_squared"])
        for key in out:
            out[key][idx] = sub[key]
        mode_shapes[np.ix_(rows, idx)] = sub["shapes"]
        for key in axle:
            axle[key][list(rows)] = sub[key]
        for j, side in enumerate(("F", "R")):
            axle_params[f"k_{name}_{side}"], axle_param_sigmas[f"k_{name}_{side}"] = sub["k"][j], sub["sigma_k"][j]
            axle_params[f"c_{name}_{side}"], axle_param_sigmas[f"c_{name}_{side}"] = sub["c"][j], sub["sigma_c"][j]
        for j, tag in enumerate(("F", "R", "FR")):
            axle_params[f"m_{name}_{tag}"], axle_param_sigmas[f"m_{name}_{tag}"] = sub["mass"][j], sub["sigma_mass"][j]
        matrices[name] = sub["matrices"]

    return {
        "method": "body4dof",
        **out,
        "mode_labels": list(MODE_ORDER),
        "mode_shapes": mode_shapes,
        "axle_k": axle["k"],
        "axle_c": axle["c"],
        "sigma_axle_k": axle["sigma_k"],
        "sigma_axle_c": axle["sigma_c"],
        "axle_params": axle_params,
        "axle_param_sigmas": axle_param_sigmas,
        "matrices": matrices,
        "poles_meas": (np.asarray(poles["fn"], dtype=float), np.asarray(poles["zeta"], dtype=float)),
        "frf": frf,
        "frf_coh2": coh2,
        "r_squared": tuple(r_squared),
    }


def eval_psds(result: dict, freqs_hz: np.ndarray) -> np.ndarray:
    """Model pushrod-force PSD shapes for equal white-velocity road input at each axle.

    Returned shape: (4, len(freqs_hz)), rows = [z_F, theta_F, z_R, theta_R];
    the absolute level is arbitrary.
    """
    f = np.asarray(freqs_hz, dtype=float)
    w = 2.0 * np.pi * np.maximum(f, 1e-6)
    out = np.full((4, f.size), np.nan)
    for name, rows, _ in SUBSYSTEMS:
        if name not in result["matrices"]:
            continue
        M, C, K = result["matrices"][name]
        G = _force_response(M, C, K, w)
        out[list(rows)] = (np.abs(G) ** 2).sum(axis=2).T / w**2
    return out
