"""Combined-Lorentzian SDOF body-mode fit.

The four body-frame PSDs ``[z_F, theta_F, z_R, theta_R]`` are fitted as two
independent subsystems: Heave + Pitch on (z_F, z_R) and Roll + Warp on
(theta_F, theta_R). Each mode has one natural frequency ``f0`` and damping
ratio ``zeta`` shared by front and rear; only the amplitude differs. Each
trace also carries a flat noise floor.

Mode shape (SDOF receptance squared, w = 2 pi f, w0 = 2 pi f0):

    H(f)^2 = 1 / ((w0^2 - w^2)^2 + (2 zeta w0 w)^2)

Per subsystem:

1. Peak identification. Each expected band is the search range for one
   mode: its most central prominent local maximum of the mean (per-trace
   normalised) PSD is taken and refined to sub-bin accuracy by parabolic
   interpolation of the log-PSD. No fit window is needed.
2. Fit region. For each mode, the contiguous bins around its peak that stay
   above ``PEAK_REGION_FRAC`` of the peak height, stopping at a valley.
   Bins outside every region are ignored, so only the peaks and their
   immediate roll-off inform the modal parameters.
3. Least squares in log-PSD (uniform weight: the maximum-likelihood
   residual for a Welch average). Each mode is parameterised by its peak
   frequency ``f_p`` (held within one bin of the measured peak), ``zeta``
   and per-trace peak heights, so ``zeta`` only controls the width and
   roll-off. ``f0 = f_p / sqrt(1 - 2 zeta^2)`` is recovered afterwards.
4. Parameter sigmas from the Jacobian with the known log-PSD noise variance
   ``trigamma(K)`` of a K-segment Welch average.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import find_peaks
from scipy.special import polygamma

from engine.logger import log

# ----------------------------------------------------------------------------
# Public constants
# ----------------------------------------------------------------------------

MODE_ORDER = ("Heave", "Pitch", "Roll", "Warp")

# Damping-ratio bounds. A fit pinned at ZETA_MAX means the peak was not resolved.
ZETA_MIN, ZETA_MAX = 0.02, 0.50

# Fit-region edge, as a fraction of each mode's peak height.
PEAK_REGION_FRAC = 0.30

# Minimum prominence (fraction of the reference-trace maximum) for a peak to be identified.
PEAK_PROMINENCE_FRAC = 0.05

# Sigma sanity thresholds. When a Jacobian-derived parameter sigma exceeds
# one of these, the sigma is replaced with NaN before being returned by
# `run_fit`. The fitted *value* is preserved -- only the uncertainty is
# masked, so downstream plots render the point but contribute nothing to
# axes autoscale (no exploding errorbar / CI band from degenerate fits).
SIGMA_DEGENERATE_F0_BAND_FRAC = 0.50  # sigma_f0 > 0.5 * configured band width
SIGMA_DEGENERATE_F0_REL = 0.20  # sigma_f0 / f0 > 20 %
SIGMA_DEGENERATE_ZETA_ABS = 0.25  # sigma_zeta > 0.25 (half of [ZETA_MIN, ZETA_MAX])
SIGMA_DEGENERATE_ZETA_REL = 1.00  # sigma_zeta > zeta (uncertainty exceeds value)
SIGMA_DEGENERATE_LOG_AMP = 2.00  # sigma_log_amp > 2 (amp factor of e^2 ~ 7.4)

_LOG_EPS = 1e-30  # guards log(0)


# ----------------------------------------------------------------------------
# Model evaluation
# ----------------------------------------------------------------------------


def _lorentz_shape(freqs_hz: np.ndarray, f0_zeta: np.ndarray) -> np.ndarray:
    """SDOF receptance squared (unscaled), broadcast over bands.

    Parameters
    ----------
    freqs_hz : (nf,) array
    f0_zeta  : (n_bands, 2) array of (f0, zeta)

    Returns
    -------
    (n_bands, nf) array of Lorentzian shapes.
    """
    omega = 2.0 * np.pi * freqs_hz
    omega0 = 2.0 * np.pi * f0_zeta[:, 0:1]
    zeta = f0_zeta[:, 1:2]
    denom = (omega0**2 - omega**2) ** 2 + (2.0 * zeta * omega0 * omega) ** 2
    return 1.0 / np.maximum(denom, _LOG_EPS)


def _unit_peak_shape(freqs_hz: np.ndarray, f_peak: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Receptance-squared shapes scaled to 1 at their peak, located by peak frequency.

    The peak of ``H^2`` sits at ``f_peak = f0 * sqrt(1 - 2 zeta^2)``. Returns
    an ``(n_modes, nf)`` array.
    """
    f0 = f_peak / np.sqrt(1.0 - 2.0 * zeta**2)
    r = freqs_hz[None, :] / f0[:, None]
    z = zeta[:, None]
    return 4.0 * z**2 * (1.0 - z**2) / ((1.0 - r**2) ** 2 + (2.0 * z * r) ** 2)


# Packed fit vector per subsystem:
#   [f_peak, zeta, log_height_t0..tN] per mode, then [log_floor_t0..tN].

# Hann |W|^2 kernel: sampled every 1/_LEAK_OVERSAMPLE bin over +/- _LEAK_HALF_BINS bins.
_LEAK_OVERSAMPLE = 8
_LEAK_HALF_BINS = 4


def _hann_leakage(df: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequency offsets (Hz) and weights with which a Hann-windowed Welch bin averages the true PSD."""
    b = np.arange(-_LEAK_HALF_BINS * _LEAK_OVERSAMPLE, _LEAK_HALF_BINS * _LEAK_OVERSAMPLE + 1) / _LEAK_OVERSAMPLE
    at_one = np.isclose(np.abs(b), 1.0)
    amp = np.where(at_one, 0.5, np.sinc(b) / np.where(at_one, 1.0, 1.0 - b**2))
    weights = amp**2
    return b * df, weights / weights.sum()


def _model_log(
    x: np.ndarray,
    freqs_hz: np.ndarray,
    n_modes: int,
    n_traces: int,
    leakage: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Log of the per-trace model PSD, shape ``(n_traces, nf)``, optionally as seen through ``leakage``."""
    cols = 2 + n_traces
    modes = x[: n_modes * cols].reshape(n_modes, cols)
    heights = np.exp(modes[:, 2:]).T  # (n_traces, n_modes)
    if leakage is None:
        peaks = heights @ _unit_peak_shape(freqs_hz, modes[:, 0], modes[:, 1])
    else:
        offsets, weights = leakage
        grid = (freqs_hz[:, None] + offsets[None, :]).ravel()
        dense = heights @ _unit_peak_shape(grid, modes[:, 0], modes[:, 1])
        peaks = dense.reshape(n_traces, freqs_hz.size, offsets.size) @ weights
    return np.log(peaks + np.exp(x[n_modes * cols :])[:, None])


def _residuals(
    x: np.ndarray,
    freqs_hz: np.ndarray,
    log_meas: np.ndarray,
    mask: np.ndarray,
    n_modes: int,
    n_traces: int,
    leakage: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Log-PSD residuals on the fit-region bins of every trace."""
    return (_model_log(x, freqs_hz, n_modes, n_traces, leakage) - log_meas)[:, mask].ravel()


def _to_reported(x: np.ndarray, n_modes: int, n_traces: int) -> np.ndarray:
    """Map ``(f_peak, zeta, log_height_t)`` per mode to ``(f0, zeta, log_amp_t)`` of :func:`_lorentz_shape`."""
    cols = 2 + n_traces
    modes = x[: n_modes * cols].reshape(n_modes, cols)
    f_peak, zeta = modes[:, 0], modes[:, 1]
    f0 = f_peak / np.sqrt(1.0 - 2.0 * zeta**2)
    # Peak value of _lorentz_shape is 1 / (4 zeta^2 (1 - zeta^2) w0^4).
    log_amp = modes[:, 2:] + np.log(4.0 * zeta**2 * (1.0 - zeta**2) * (2.0 * np.pi * f0) ** 4)[:, None]
    return np.column_stack([f0, zeta, log_amp])


# ----------------------------------------------------------------------------
# Peak identification
# ----------------------------------------------------------------------------


def _half_power_zeta(
    freqs: np.ndarray, psd: np.ndarray, peak_idx: int, z_min: float = 0.05, z_max: float = 0.30
) -> float:
    """Classical half-power (-3dB) damping estimate around `peak_idx`."""
    peak = float(psd[peak_idx])
    if peak <= 0.0:
        return 0.10
    half = 0.5 * peak
    lo = peak_idx
    hi = peak_idx
    while lo > 0 and psd[lo] > half:
        lo -= 1
    while hi < len(psd) - 1 and psd[hi] > half:
        hi += 1
    f0 = float(freqs[peak_idx])
    bw = float(freqs[hi] - freqs[lo])
    if f0 <= 0.0 or bw <= 0.0:
        return 0.10
    return float(np.clip(bw / (2.0 * f0), z_min, z_max))


def _band_mask(freqs: np.ndarray, bands: list[tuple[float, float]]) -> np.ndarray:
    """Bins inside any of ``bands``."""
    mask = np.zeros(freqs.size, dtype=bool)
    for lo, hi in bands:
        mask |= (freqs >= lo) & (freqs <= hi)
    return mask


def _pick_seed_peaks(
    freqs_fit: np.ndarray, ref: np.ndarray, bands: list[tuple[float, float]]
) -> list[tuple[float, int] | None]:
    """Assign to each band the prominent local maximum inside it closest to its centre.

    Local maxima are found over the whole spectrum (so a peak sitting on a
    band edge still qualifies), prominence is judged against the largest
    value inside the bands, and a Hungarian assignment keeps one peak per
    band. Returns ``(f_peak, peak_idx)`` per band, or ``None`` when a band
    holds no candidate.
    """
    if not bands:
        return []
    in_bands = _band_mask(freqs_fit, bands)
    p_max = float(ref[in_bands].max()) if in_bands.any() else 0.0
    if p_max <= 0.0:
        return [None] * len(bands)
    peaks, _ = find_peaks(ref, prominence=PEAK_PROMINENCE_FRAC * p_max)
    if peaks.size == 0:
        return [None] * len(bands)
    cand_f = freqs_fit[peaks]
    lo = np.array([b[0] for b in bands])
    hi = np.array([b[1] for b in bands])
    allowed = (cand_f[:, None] >= lo[None, :]) & (cand_f[:, None] <= hi[None, :])
    cost = np.where(allowed, np.abs(cand_f[:, None] - 0.5 * (lo + hi)[None, :]), 1e6)
    row_ind, col_ind = linear_sum_assignment(cost)
    assignments: list[tuple[float, int] | None] = [None] * len(bands)
    for k, b in zip(row_ind, col_ind):
        if allowed[k, b]:
            assignments[b] = (float(cand_f[k]), int(peaks[k]))
    return assignments


def _peak_reference(meas: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Mean of the traces, each normalised to its maximum inside ``search``."""
    peaks = np.max(meas[:, search], axis=1, keepdims=True)
    return np.mean(meas / np.where(peaks > 0.0, peaks, 1.0), axis=0)


def _refine_peak(freqs: np.ndarray, ref: np.ndarray, idx: int) -> float:
    """Sub-bin peak frequency from a parabola through the log-PSD at ``idx - 1 .. idx + 1``."""
    if idx <= 0 or idx >= len(ref) - 1:
        return float(freqs[idx])
    y_m, y_0, y_p = np.log(np.maximum(ref[idx - 1 : idx + 2], _LOG_EPS))
    curvature = y_m - 2.0 * y_0 + y_p
    if curvature >= 0.0:
        return float(freqs[idx])
    return float(freqs[idx] + 0.5 * (y_m - y_p) / curvature * (freqs[idx + 1] - freqs[idx]))


def _peak_region(ref: np.ndarray, idx: int) -> np.ndarray:
    """Contiguous bins around ``idx`` above ``PEAK_REGION_FRAC`` of its height, stopping at valleys."""
    threshold = PEAK_REGION_FRAC * ref[idx]
    lo = idx
    while lo > 0 and threshold <= ref[lo - 1] <= ref[lo]:
        lo -= 1
    hi = idx
    while hi < len(ref) - 1 and threshold <= ref[hi + 1] <= ref[hi]:
        hi += 1
    region = np.zeros(len(ref), dtype=bool)
    region[lo : hi + 1] = True
    return region


# ----------------------------------------------------------------------------
# Single-subsystem fit (heave+pitch OR roll+warp)
# ----------------------------------------------------------------------------


def _fit_subsystem(
    freqs_fit: np.ndarray,
    meas: np.ndarray,
    bands: list[tuple[float, float]],
    labels: list[str],
    n_avg: int | None = None,
    quiet: bool = False,
    welch_hann: bool = False,
):
    """Fit 1+ modes shared by 1+ traces on the peak regions of their PSDs.

    Each band is the search range for one mode's peak. Returns a dict with
    ``params`` rows ``(f0, zeta, amp_t...)`` in the units of
    :func:`_lorentz_shape`; ``sigmas`` rows ``(sigma_f0, sigma_zeta,
    sigma_log_height_t...)``; ``heights`` (peak PSD per mode and trace);
    ``floors``; ``peak_found``; ``r_squared``; ``fit_mask``; and the packed
    vector ``x`` + ``leakage`` for :func:`_model_log`. With ``n_avg`` (Welch
    segments averaged) the sigmas use the log-PSD noise variance
    ``trigamma(n_avg)``; otherwise the residual MSE.
    """
    n_traces, n_modes = meas.shape[0], len(bands)
    cols = 2 + n_traces
    df = float(freqs_fit[1] - freqs_fit[0])
    leakage = _hann_leakage(df) if welch_hann else None
    ref = _peak_reference(meas, _band_mask(freqs_fit, bands))

    # ---- 1. Locate each mode's peak and its fit region.
    peaks = []  # (idx, f_peak, f_peak_lo, f_peak_hi)
    peak_found = []
    fit_mask = np.zeros(freqs_fit.size, dtype=bool)
    for (b_lo, b_hi), label, pick in zip(bands, labels, _pick_seed_peaks(freqs_fit, ref, bands)):
        peak_found.append(pick is not None)
        if pick is not None:
            idx = pick[1]
            f_peak = _refine_peak(freqs_fit, ref, idx)
            f_lo, f_hi = f_peak - df, f_peak + df
            region = _peak_region(ref, idx)
        else:
            region = (freqs_fit >= b_lo) & (freqs_fit <= b_hi)
            if not region.any():
                region = np.ones(freqs_fit.size, dtype=bool)
            idx = int(np.flatnonzero(region)[np.argmax(ref[region])])
            f_peak = float(freqs_fit[idx])
            f_lo = min(float(freqs_fit[region][0]), f_peak - 0.5 * df)
            f_hi = max(float(freqs_fit[region][-1]), f_peak + 0.5 * df)
            if not quiet:
                log.warning("    %s: no distinct peak in %.1f-%.1f Hz; fit is weakly constrained.", label, b_lo, b_hi)
        peaks.append((idx, f_peak, f_lo, f_hi))
        fit_mask |= region

    # ---- 2. Seeds and bounds. Each floor is capped at its trace's lowest fitted bin.
    log_meas = np.log(np.maximum(meas, _LOG_EPS))
    floor_hi = np.log(np.maximum(meas[:, fit_mask].min(axis=1), 1e-12))
    x0, lo, hi = [], [], []
    for idx, f_peak, f_lo, f_hi in peaks:
        x0 += [f_peak, _half_power_zeta(freqs_fit, ref, idx)]
        lo += [f_lo, ZETA_MIN]
        hi += [f_hi, ZETA_MAX]
        for t in range(n_traces):
            x0.append(log_meas[t, idx])
            lo.append(log_meas[t].max() - 25.0)
            hi.append(log_meas[t].max() + 2.0)
    x0 += list(floor_hi - np.log(2.0))
    lo += list(floor_hi - 25.0)
    hi += list(floor_hi)
    lo, hi = np.array(lo), np.array(hi)
    x0 = np.clip(np.array(x0), lo + 1e-9, hi - 1e-9)

    # ---- 3. Single log-space least-squares fit on the peak regions.
    fit = least_squares(
        _residuals,
        x0,
        bounds=(lo, hi),
        args=(freqs_fit, log_meas, fit_mask, n_modes, n_traces, leakage),
        x_scale="jac",
    )
    x = fit.x

    # ---- 4. Sigmas of (f0, zeta, log peak height) from the fit covariance.
    sigmas = np.full((n_modes, cols), np.nan)
    if n_avg is not None and int(n_avg) >= 1:
        noise_var = float(polygamma(1, int(n_avg)))
    else:
        noise_var = float(fit.fun @ fit.fun) / max(fit.fun.size - x.size, 1)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * noise_var
        for m in range(n_modes):
            r = m * cols
            f_peak, z = x[r], x[r + 1]
            c2 = 1.0 / (1.0 - 2.0 * z**2)
            # f0 = f_peak / sqrt(1 - 2 zeta^2): gradient w.r.t. (f_peak, zeta).
            grad = np.array([np.sqrt(c2), 2.0 * z * f_peak * np.sqrt(c2) * c2])
            var = np.concatenate([[grad @ cov[r : r + 2, r : r + 2] @ grad], np.diag(cov)[r + 1 : r + cols]])
            sigmas[m] = np.where(var > 0.0, np.sqrt(np.abs(var)), np.nan)
    except np.linalg.LinAlgError:
        pass

    lm = log_meas[:, fit_mask]
    ss_tot = float(np.sum((lm - lm.mean()) ** 2))
    r_squared = 1.0 - float(fit.fun @ fit.fun) / ss_tot if ss_tot > 0.0 else float("nan")

    reported = _to_reported(x, n_modes, n_traces)
    if not quiet:
        for label, z in zip(labels, reported[:, 1]):
            if z <= ZETA_MIN + 1e-3 or z >= ZETA_MAX - 1e-3:
                log.warning("    %s damping pinned at bound (zeta=%.3f); peak not resolved.", label, z)
    params = reported.copy()
    params[:, 2:] = np.exp(reported[:, 2:])
    return {
        "params": params,
        "sigmas": sigmas,
        "heights": np.exp(x[: n_modes * cols].reshape(n_modes, cols)[:, 2:]),
        "floors": np.exp(x[n_modes * cols :]),
        "peak_found": np.array(peak_found),
        "r_squared": r_squared,
        "fit_mask": fit_mask,
        "x": x,
        "leakage": leakage,
    }


def fit_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: tuple[float, float],
    n_avg: int | None = None,
    welch_hann: bool = True,
) -> dict | None:
    """Peak-first single-mode fit of one PSD whose peak lies inside ``band``.

    Same procedure as :func:`run_fit` for a single trace. Returns ``None``
    when the band holds fewer than 3 bins or no positive data. Keys:
    ``f0``, ``zeta``, ``sigma_f0``, ``sigma_zeta``, ``peak_height``,
    ``floor``, ``peak_found``, ``r_squared``, ``fit_mask`` and ``curve``
    (``(f, psd)`` of the fitted model over the fit region, as seen by Welch
    when ``welch_hann``).
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    keep = (freqs > 0.0) & np.isfinite(psd)
    freqs, psd = freqs[keep], psd[keep]
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if int(in_band.sum()) < 3 or float(psd[in_band].max()) <= 0.0:
        return None
    scale = float(psd[in_band].max())
    sub = _fit_subsystem(
        freqs, (psd / scale)[None, :], [band], ["Peak"], n_avg=n_avg, quiet=True, welch_hann=welch_hann
    )
    region = freqs[sub["fit_mask"]]
    f_curve = np.linspace(region[0], region[-1], 200)
    y_curve = np.exp(_model_log(sub["x"], f_curve, 1, 1, sub["leakage"]))[0] * scale
    return {
        "f0": float(sub["params"][0, 0]),
        "zeta": float(sub["params"][0, 1]),
        "sigma_f0": float(sub["sigmas"][0, 0]),
        "sigma_zeta": float(sub["sigmas"][0, 1]),
        "peak_height": float(sub["heights"][0, 0]) * scale,
        "floor": float(sub["floors"][0]) * scale,
        "peak_found": bool(sub["peak_found"][0]),
        "r_squared": float(sub["r_squared"]),
        "fit_mask": sub["fit_mask"],
        "curve": (f_curve, y_curve),
    }


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def _sanitise_sigmas(
    params: np.ndarray,
    sigmas: np.ndarray,
    bands: list[tuple[float, float]],
    mode_labels: list[str],
    quiet: bool = False,
) -> np.ndarray:
    """Replace untrustworthy Jacobian-derived sigmas with NaN.

    Compares each parameter sigma against the `SIGMA_DEGENERATE_*` module
    constants. When a threshold is exceeded the sigma is set to NaN -- the
    fitted *value* is left untouched. Downstream plot consumers treat NaN
    sigmas as zero-length errorbars / suppressed CI bands, which keeps
    pathologically wide covariance estimates from blowing up axes autoscale.

    A WARNING is logged for every flagged parameter so degenerate fits are
    visible without having to inspect plots.
    """
    sanitised = sigmas.copy()
    n_traces = sigmas.shape[1] - 2
    trace_names = ["front", "rear"] + [f"t{t}" for t in range(2, n_traces)]
    for i, ((lo, hi), label) in enumerate(zip(bands, mode_labels)):
        band_width = max(float(hi) - float(lo), 1e-6)
        f0 = float(params[i, 0])
        zeta = float(params[i, 1])
        s_f0 = float(sigmas[i, 0])
        s_zeta = float(sigmas[i, 1])
        if np.isfinite(s_f0):
            why = []
            if s_f0 > SIGMA_DEGENERATE_F0_BAND_FRAC * band_width:
                why.append(f"sigma_f0={s_f0:.3f} > {SIGMA_DEGENERATE_F0_BAND_FRAC:.2f}*band_width({band_width:.2f})")
            if f0 > 0.0 and s_f0 / f0 > SIGMA_DEGENERATE_F0_REL:
                why.append(f"sigma_f0/f0={s_f0 / f0:.2f} > {SIGMA_DEGENERATE_F0_REL:.2f}")
            if why:
                sanitised[i, 0] = np.nan
                if not quiet:
                    log.warning("    %s degenerate f0 sigma (%s)", label, "; ".join(why))
        if np.isfinite(s_zeta):
            why = []
            if s_zeta > SIGMA_DEGENERATE_ZETA_ABS:
                why.append(f"sigma_zeta={s_zeta:.3f} > {SIGMA_DEGENERATE_ZETA_ABS:.2f}")
            if zeta > 0.0 and s_zeta / zeta > SIGMA_DEGENERATE_ZETA_REL:
                why.append(f"sigma_zeta/zeta={s_zeta / zeta:.2f} > {SIGMA_DEGENERATE_ZETA_REL:.2f}")
            if why:
                sanitised[i, 1] = np.nan
                if not quiet:
                    log.warning("    %s degenerate zeta sigma (%s)", label, "; ".join(why))
        for t in range(n_traces):
            s_log_amp = float(sigmas[i, 2 + t])
            if np.isfinite(s_log_amp) and s_log_amp > SIGMA_DEGENERATE_LOG_AMP:
                sanitised[i, 2 + t] = np.nan
                if not quiet:
                    log.warning(
                        "    %s degenerate amp sigma (%s, sigma_log_amp=%.2f > %.1f)",
                        label,
                        trace_names[t],
                        s_log_amp,
                        SIGMA_DEGENERATE_LOG_AMP,
                    )
    return sanitised


def run_fit(
    freqs_fit: np.ndarray,
    body_psds_norm: np.ndarray,
    expected_bands: dict,
    n_avg: int | None = None,
    quiet: bool = False,
    welch_hann: bool = False,
) -> dict:
    """Combined-Lorentzian fit across the 4 body DOFs.

    Parameters
    ----------
    freqs_fit : (nf,) array
        Uniformly spaced frequency bins (Hz); may span the whole spectrum.
    body_psds_norm : (4, nf) array
        Unsmoothed Welch PSDs, ordered [z_F, theta_F, z_R, theta_R].
    expected_bands : dict
        Per-mode (lo, hi[, mid]) tuples keyed by lowercase mode name: the
        range in which each mode's peak is searched. The fitted roll-off may
        extend outside it.
    n_avg : int or None
        Number of Welch segments averaged into ``body_psds_norm``. When
        supplied, sigmas use the known log-PSD noise variance
        ``trigamma(K)``; otherwise the residual MSE.
    quiet : bool
        Suppress per-fit logging (bootstrap draws).
    welch_hann : bool
        The PSDs are Hann-windowed Welch estimates: fit the model as seen
        through the window's spectral kernel, so the reported ``f0``/``zeta``
        describe the true response rather than the leakage-broadened one.

    Returns
    -------
    dict with keys: method, params (4x4), baselines (4,), fn (4,),
        zeta (4,), amp_front (4,), amp_rear (4,) (fitted peak PSD of each
        mode on the front/rear trace, input units), sigma_fn (4,),
        sigma_zeta (4,), sigma_amp_front (4,), sigma_amp_rear (4,),
        peak_found (4,), r_squared (R^2_hp, R^2_rw), mode_labels,
        fit_mask (4, nf) bins used per body DOF.
    """
    bands_hp = [expected_bands["heave"][:2], expected_bands["pitch"][:2]]
    bands_rw = [expected_bands["roll"][:2], expected_bands["warp"][:2]]
    if not quiet:
        log.info("  Fitting combined Lorentzians...")

    fit_kw = {"n_avg": n_avg, "quiet": quiet, "welch_hann": welch_hann}
    hp = _fit_subsystem(freqs_fit, body_psds_norm[[0, 2]], bands_hp, ["Heave", "Pitch"], **fit_kw)
    rw = _fit_subsystem(freqs_fit, body_psds_norm[[1, 3]], bands_rw, ["Roll", "Warp"], **fit_kw)
    sig_hp = _sanitise_sigmas(hp["params"], hp["sigmas"], bands_hp, ["Heave", "Pitch"], quiet=quiet)
    sig_rw = _sanitise_sigmas(rw["params"], rw["sigmas"], bands_rw, ["Roll", "Warp"], quiet=quiet)

    params = np.vstack((hp["params"], rw["params"]))
    sigmas = np.vstack((sig_hp, sig_rw))
    heights = np.vstack((hp["heights"], rw["heights"]))
    baselines = np.array([hp["floors"][0], rw["floors"][0], hp["floors"][1], rw["floors"][1]])

    # Delta-method: sigma_amp ~= amp * sigma_log_amp.
    amp_front = heights[:, 0]
    amp_rear = heights[:, 1]
    sigma_amp_front = np.where(np.isfinite(sigmas[:, 2]), amp_front * sigmas[:, 2], np.nan)
    sigma_amp_rear = np.where(np.isfinite(sigmas[:, 3]), amp_rear * sigmas[:, 3], np.nan)

    return {
        "method": "lorentzian_combined",
        "params": params,
        "baselines": baselines,
        "leakage_df": float(freqs_fit[1] - freqs_fit[0]) if welch_hann else None,
        "fn": params[:, 0],
        "zeta": params[:, 1],
        "amp_front": amp_front,
        "amp_rear": amp_rear,
        "sigma_fn": sigmas[:, 0],
        "sigma_zeta": sigmas[:, 1],
        "sigma_amp_front": sigma_amp_front,
        "sigma_amp_rear": sigma_amp_rear,
        "peak_found": np.concatenate([hp["peak_found"], rw["peak_found"]]),
        "r_squared": (float(hp["r_squared"]), float(rw["r_squared"])),
        "mode_labels": list(MODE_ORDER),
        "fit_mask": np.stack([hp["fit_mask"], rw["fit_mask"], hp["fit_mask"], rw["fit_mask"]]),
    }


def eval_psds(result: dict, freqs_hz: np.ndarray, include_baseline: bool = True) -> np.ndarray:
    """Evaluate the fitted PSDs at `freqs_hz` for all 4 body DOFs (normalised fit units).

    Returned shape: (4, len(freqs_hz)), rows = [z_F, theta_F, z_R, theta_R].
    z rows sum the Heave + Pitch modes, theta rows sum Roll + Warp, each
    with its front or rear amplitude plus the trace's flat floor. Results
    fitted with ``welch_hann`` are returned as seen through the Hann window,
    i.e. directly comparable to the measured Welch PSD.
    """
    params = result["params"]  # (4, 4): modes x [f0, zeta, amp_front, amp_rear]
    baselines = result.get("baselines")
    if baselines is None or not include_baseline:
        baselines = np.zeros(4)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    leakage_df = result.get("leakage_df")
    if leakage_df:
        offsets, weights = _hann_leakage(float(leakage_df))
        grid = (freqs_hz[:, None] + offsets[None, :]).ravel()
    else:
        grid = freqs_hz
    # (mode_rows, amp_col, baseline_idx) for each output DOF row.
    specs = [
        ([0, 1], 2, 0),  # z_F     <- heave/pitch front amplitudes
        ([2, 3], 2, 1),  # theta_F <- roll/warp  front amplitudes
        ([0, 1], 3, 2),  # z_R     <- heave/pitch rear amplitudes
        ([2, 3], 3, 3),  # theta_R <- roll/warp  rear amplitudes
    ]
    out = []
    for mode_rows, amp_col, base_idx in specs:
        sub = params[mode_rows]
        peaks = sub[:, amp_col] @ _lorentz_shape(grid, sub[:, :2])
        if leakage_df:
            peaks = peaks.reshape(freqs_hz.size, -1) @ weights
        out.append(peaks + float(baselines[base_idx]))
    return np.stack(out)
