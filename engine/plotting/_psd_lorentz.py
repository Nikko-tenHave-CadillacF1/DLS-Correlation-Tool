"""Shared PSD Lorentzian-peak model + single-peak fit.

Used by :mod:`engine.plotting.generate_psd_hist` (per-plot ``lorentz_fit``
annotations on PSD figures). The heavier multi-peak / multi-subsystem
Lorentz stack used by the vibrations pipeline lives in
:mod:`engine.vibrations_lorentz`; this module intentionally stays scoped to
the light single-peak fit that runs during plot generation.
"""

from __future__ import annotations

import numpy as np

_ZETA_LO_PSD, _ZETA_HI_PSD = 1e-3, 1.0
_SATURATION_MARGIN = 0.05


def _lorentz_peak_model(f, f0, zeta, amp, baseline):
    """Single-DOF Lorentzian PSD peak plus a flat baseline."""
    denom = (f0**2 - f**2) ** 2 + (2.0 * zeta * f0 * f) ** 2
    return amp * f0**4 / np.maximum(denom, 1e-30) + baseline


def _fit_lorentz_peak(freq, power, f_lo, f_hi, min_points=8):
    """Fit a single Lorentzian + baseline inside ``[f_lo, f_hi]``.

    Returns ``(f0, zeta, amp, base, f_lo, f_hi, sigma_zeta, r_squared, saturated)``
    or ``None`` when the window has too few points / non-finite data.
    """
    from scipy.optimize import least_squares

    mask = (freq >= f_lo) & (freq <= f_hi)
    if int(mask.sum()) < min_points:
        return None
    f_fit = np.asarray(freq[mask], dtype=float)
    p_fit = np.asarray(power[mask], dtype=float)
    if not np.all(np.isfinite(p_fit)) or float(np.max(p_fit)) <= 0.0:
        return None
    p_peak = float(np.max(p_fit))
    p_base = float(np.percentile(p_fit, 10))
    f0_init = float(f_fit[int(np.argmax(p_fit))])
    amp_hi = max(p_peak * 1e3, 1.0)
    base_hi = max(p_peak, 1e-9)
    p0 = [f0_init, 0.05, max(p_peak - p_base, 1e-12), p_base]
    lo_bounds = [f_lo, _ZETA_LO_PSD, 0.0, 0.0]
    hi_bounds = [f_hi, _ZETA_HI_PSD, amp_hi, base_hi]
    log_p = np.log(np.maximum(p_fit, 1e-30))

    def residual(params):
        model = _lorentz_peak_model(f_fit, *params)
        return np.log(np.maximum(model, 1e-30)) - log_p

    try:
        res = least_squares(residual, p0, bounds=(lo_bounds, hi_bounds), max_nfev=2000)
    except Exception:
        return None
    f0_fit, zeta_fit, amp_fit, base_fit = (float(v) for v in res.x)
    log_p_pred = np.log(np.maximum(_lorentz_peak_model(f_fit, f0_fit, zeta_fit, amp_fit, base_fit), 1e-30))
    ss_res = float(np.sum((log_p - log_p_pred) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_obs = len(p_fit)
    dof = max(n_obs - 4, 1)
    mse = 2.0 * float(res.cost) / dof
    sigma_zeta = float("nan")
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * mse
        var_zeta = float(cov[1, 1])
        if var_zeta >= 0.0 and np.isfinite(var_zeta):
            sigma_zeta = float(np.sqrt(var_zeta))
    except np.linalg.LinAlgError:
        pass
    saturated = False
    for val, b_lo, b_hi in (
        (f0_fit, f_lo, f_hi),
        (zeta_fit, _ZETA_LO_PSD, _ZETA_HI_PSD),
    ):
        span = b_hi - b_lo
        if span > 0 and (val - b_lo < _SATURATION_MARGIN * span or b_hi - val < _SATURATION_MARGIN * span):
            saturated = True
            break
    return f0_fit, zeta_fit, amp_fit, base_fit, f_lo, f_hi, sigma_zeta, r_squared, saturated
