"""Combined-Lorentzian SDOF fit pipeline for 4-DOF body vibrations.

Fits two shared `(f₀, ζ)` modes per heave/pitch (front-rear z) and roll/warp
(front-rear θ) pair, with independent amplitudes per trace. Cost is log-space
residuals; uncertainties come from the Jacobian of a final `least_squares`
polish.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

from engine.logger import log

_MODE_ORDER = ["Heave", "Pitch", "Roll", "Warp"]
_LOG_EPS = 1e-30
_ZETA_LO, _ZETA_HI = 0.02, 0.70


def _eval_band_shapes(freqs_hz: np.ndarray, fz: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * freqs_hz
    omega0 = 2.0 * np.pi * fz[:, 0:1]
    zeta = fz[:, 1:2]
    denom = (omega0**2 - omega**2)**2 + (2.0 * zeta * omega0 * omega)**2
    return 1.0 / np.maximum(denom, 1e-30)


def _eval_lorentz_sum(freqs_hz: np.ndarray, dof_params: np.ndarray) -> np.ndarray:
    shapes = _eval_band_shapes(freqs_hz, dof_params[:, :2])
    return dof_params[:, 2] @ shapes


def _residual_lorentz_combined_log(packed: np.ndarray, freqs_fit: np.ndarray,
                                   meas_norm: np.ndarray, n_bands: int,
                                   n_traces: int,
                                   weights: np.ndarray | None) -> np.ndarray:
    cols = 2 + n_traces
    n_band_params = n_bands * cols
    params = packed[:n_band_params].reshape(n_bands, cols)
    baselines = np.exp(packed[n_band_params:])
    amps = np.exp(params[:, 2:])
    shapes = _eval_band_shapes(freqs_fit, params[:, :2])
    models = amps.T @ shapes + baselines[:, None]
    res = (np.log(np.maximum(models, _LOG_EPS))
           - np.log(np.maximum(meas_norm, _LOG_EPS)))
    if weights is not None:
        res = res * np.sqrt(weights)
    return res.ravel()


def _cost_lorentz_combined(packed: np.ndarray, freqs_fit: np.ndarray,
                           meas_norm: np.ndarray, n_bands: int,
                           n_traces: int, weights: np.ndarray) -> float:
    r = _residual_lorentz_combined_log(packed, freqs_fit, meas_norm,
                                       n_bands, n_traces, weights)
    return float(np.dot(r, r))


def _half_power_zeta(freqs: np.ndarray, psd: np.ndarray, peak_idx: int,
                     z_min: float = 0.05, z_max: float = 0.30) -> float:
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


def _peak_pick_x0(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                  ranges: list, n_traces: int) -> np.ndarray:
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


def _expanded_frequency_bounds(freqs_fit: np.ndarray, ranges: list,
                               expansion: float = 1.0,
                               min_margin: float = 0.5) -> list:
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


def _de_lorentz(bounds, args, x0, seed, popsize=12):
    return differential_evolution(
        _cost_lorentz_combined, bounds=bounds, args=args,
        x0=x0, seed=seed, polish=True, disp=False,
        updating="immediate", workers=1, popsize=popsize,
        init="sobol", tol=1e-3, mutation=(0.5, 1.5),
    )


def _hits_zeta_bound(x: np.ndarray, n_bands: int, n_traces: int,
                     margin: float = 0.02) -> bool:
    cols = 2 + n_traces
    n_band_params = n_bands * cols
    zetas = x[:n_band_params].reshape(n_bands, cols)[:, 1]
    span = _ZETA_HI - _ZETA_LO
    return bool(np.any(zetas <= _ZETA_LO + margin * span)
                or np.any(zetas >= _ZETA_HI - margin * span))


def _fit_lorentz_combined(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                          ranges: list, weights: np.ndarray):
    n_traces = meas_norm.shape[0]
    n_bands = len(ranges)
    freq_bounds = _expanded_frequency_bounds(freqs_fit, ranges)
    bounds = []
    for lo, hi in freq_bounds:
        bounds.append((lo, hi))
        bounds.append((_ZETA_LO, _ZETA_HI))
        for _ in range(n_traces):
            bounds.append((np.log(1e-6), np.log(1e6)))
    for t in range(n_traces):
        peak_t = float(np.max(meas_norm[t]))
        base_hi = max(0.3 * peak_t, 1e-9)
        bounds.append((np.log(1e-9), np.log(base_hi)))
    args = (freqs_fit, meas_norm, n_bands, n_traces, weights)
    x0 = _peak_pick_x0(freqs_fit, meas_norm, ranges, n_traces)
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
    margin = 1e-8 * np.maximum(np.abs(hi_arr - lo_arr), 1.0)
    x_polish = np.clip(best_x, lo_arr + margin, hi_arr - margin)
    sigmas_x = np.full_like(best_x, np.nan, dtype=float)
    r_squared = float("nan")
    try:
        ls = least_squares(
            _residual_lorentz_combined_log, x_polish,
            bounds=(lo_arr, hi_arr), args=args,
            method="trf", max_nfev=400,
        )
        best_x = ls.x
        ssr = float(np.dot(ls.fun, ls.fun))
        dof = max(ls.fun.size - x_polish.size, 1)
        mse = ssr / dof
        try:
            cov = np.linalg.inv(ls.jac.T @ ls.jac) * mse
            var = np.diag(cov)
            valid = np.isfinite(var) & (var > 0)
            sigmas_x = np.where(valid, np.sqrt(np.where(valid, var, 1.0)), np.nan)
        except np.linalg.LinAlgError:
            pass
        res_uw = _residual_lorentz_combined_log(
            ls.x, freqs_fit, meas_norm, n_bands, n_traces, None)
        log_meas = np.log(np.maximum(meas_norm, _LOG_EPS)).ravel()
        ss_res = float(np.dot(res_uw, res_uw))
        ss_tot = float(np.sum((log_meas - log_meas.mean()) ** 2))
        if ss_tot > 0:
            r_squared = 1.0 - ss_res / ss_tot
    except Exception as exc:  # noqa: BLE001 - polish is best-effort
        log.debug("Lorentz polish failed: %s", exc)
    params = best_x[:n_band_params].reshape(n_bands, cols).copy()
    params[:, 2:] = np.exp(params[:, 2:])
    baselines = np.exp(best_x[n_band_params:])
    sig_band = sigmas_x[:n_band_params].reshape(n_bands, cols)
    sigmas = sig_band[:, :2].copy()
    order = np.argsort(params[:, 0])
    return params[order], baselines, sigmas[order], r_squared


def _lorentz_mode_shapes(params: np.ndarray) -> np.ndarray:
    shapes = np.zeros((4, 4), dtype=float)
    for i, (_, _, amp_front, amp_rear) in enumerate(params):
        front = np.sqrt(max(float(amp_front), 0.0))
        rear = np.sqrt(max(float(amp_rear), 0.0))
        if i == 0:
            shapes[:, i] = [front, 0.0, rear, 0.0]
        elif i == 1:
            shapes[:, i] = [front, 0.0, -rear, 0.0]
        elif i == 2:
            shapes[:, i] = [0.0, front, 0.0, rear]
        else:
            shapes[:, i] = [0.0, front, 0.0, -rear]
        peak = np.max(np.abs(shapes[:, i]))
        if peak > 0:
            shapes[:, i] /= peak
    return shapes


def _coherence_weights(coh_fit: np.ndarray | None, floor: float = 0.2):
    if coh_fit is None:
        return None, None
    return np.maximum(coh_fit[0], floor), np.maximum(coh_fit[1], floor)


def run_fit(freqs_fit: np.ndarray, meas_norm: np.ndarray,
            fr: dict, coh_fit: np.ndarray | None = None) -> dict:
    """Combined Lorentzian fit across the 4 body DOFs.

    Returns a result dict with `method="lorentzian_combined"`, modal
    frequencies/dampings, uncertainties from the Jacobian polish, and
    `mode_shapes` reconstructed from the fitted amplitudes.
    """
    bands_hp = [fr["heave"][:2], fr["pitch"][:2]]
    bands_rw = [fr["roll"][:2], fr["warp"][:2]]
    weights_hp, weights_rw = _coherence_weights(coh_fit)
    log.info("  Fitting combined Lorentzians...")
    params_hp, base_hp, sig_hp, r2_hp = _fit_lorentz_combined(
        freqs_fit, meas_norm[[0, 2]], bands_hp, weights_hp)
    params_rw, base_rw, sig_rw, r2_rw = _fit_lorentz_combined(
        freqs_fit, meas_norm[[1, 3]], bands_rw, weights_rw)
    params = np.vstack((params_hp, params_rw))
    sigmas = np.vstack((sig_hp, sig_rw))
    baselines = np.array([base_hp[0], base_rw[0], base_hp[1], base_rw[1]])
    return {
        "method": "lorentzian_combined",
        "params": params,
        "baselines": baselines,
        "fn": params[:, 0],
        "zeta": params[:, 1],
        "sigma_fn": sigmas[:, 0],
        "sigma_zeta": sigmas[:, 1],
        "r_squared": (float(r2_hp), float(r2_rw)),
        "mode_labels": list(_MODE_ORDER),
        "mode_shapes": _lorentz_mode_shapes(params),
    }


def eval_psds(result: dict, freqs_hz: np.ndarray,
              include_baseline: bool = True) -> np.ndarray:
    """Evaluate the fitted Lorentzian PSDs across `freqs_hz` for all 4 DOFs."""
    params = result["params"]
    baselines = result.get("baselines")
    if baselines is None or not include_baseline:
        baselines = np.zeros(4)
    specs = [([0, 1], 2, 0), ([2, 3], 2, 1),
             ([0, 1], 3, 2), ([2, 3], 3, 3)]
    return np.stack([
        _eval_lorentz_sum(freqs_hz, params[band_rows][:, [0, 1, amp_col]])
        + baselines[base_idx]
        for band_rows, amp_col, base_idx in specs
    ])
