"""Combined-Lorentzian SDOF body-mode fit.

Audit summary
-------------
Given the four body-frame PSDs

    body_psds[0]  = heave-component traces (z_F)
    body_psds[1]  = roll-component traces  (theta_F)
    body_psds[2]  = heave-component traces (z_R)
    body_psds[3]  = roll-component traces  (theta_R)

we fit TWO Lorentzian peaks to the front+rear z-PSDs (Heave + Pitch) and
TWO Lorentzian peaks to the front+rear theta-PSDs (Roll + Warp).

Per subsystem (heave/pitch or roll/warp) the fit parameters are:

    band 1 : (f0_a, zeta_a, amp_front_a, amp_rear_a)
    band 2 : (f0_b, zeta_b, amp_front_b, amp_rear_b)
    plus a per-trace baseline offset (broadband noise floor).

`f0` and `zeta` are SHARED between front and rear (one mode picks up at
both axles), only the amplitudes differ. This is the physical assumption
that justifies grouping front+rear into one optimisation.

Model PSD for a single Lorentzian peak (SDOF receptance squared):

    H(omega; f0, zeta)^2 = 1 / ((omega0^2 - omega^2)^2 + (2 zeta omega0 omega)^2)

with omega = 2 pi f and omega0 = 2 pi f0.

Pipeline
~~~~~~~~
1. Build initial guess from peak-pick + half-power damping.
2. Run L-BFGS-B locally in log-amplitude space.
3. If local minimum hits the damping bounds, fall back to
   differential evolution (Sobol-seeded, popsize=12).
4. Final `least_squares` polish in log space gives a Jacobian
   from which (f0, zeta) standard errors are derived.

All cost evaluations are LOG-SPACE residuals so a broad noise floor
does not swamp a sharp resonance peak. A per-bin weight equal to the
PSD magnitude (and optional coherence) re-emphasises the peaks.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import (differential_evolution, least_squares,
                            linear_sum_assignment, minimize)
from scipy.signal import find_peaks
from scipy.special import polygamma

from engine.logger import log

# ----------------------------------------------------------------------------
# Public constants
# ----------------------------------------------------------------------------

MODE_ORDER = ("Heave", "Pitch", "Roll", "Warp")

# Bounds on damping ratio (dimensionless). Real chassis modes sit
# comfortably between 5 % and 30 %; the hard limits give the optimiser
# room without letting it degenerate to a flat curve (zeta -> 1) or a
# zero-width spike (zeta -> 0). The upper limit deliberately stays above
# the physical envelope so a fit that ends up near ZETA_MAX is clearly
# visible as "failed to find a peak" in the diagnosis plots.
ZETA_MIN, ZETA_MAX = 0.02, 0.50

# Minimum frequency separation (Hz) enforced between adjacent bands within a
# subsystem (Heave/Pitch, Roll/Warp). Prevents the optimiser from collapsing
# two Lorentzians onto the same physical peak when several peaks compete
# inside the joint search range.
MIN_BAND_GAP_HZ = 1.0

# Per-band cost-mass normalisation. The fit weights inside each configured
# band are rescaled so every band contributes the same total mass to the
# cost (independent of how tall the band's peak is). Without this a strong
# peak (e.g. Roll) drowns out the cost contribution of a weaker peak in the
# same subsystem (e.g. Warp) and the weak peak's f0/zeta come out poorly
# resolved. The baseline (out-of-band) bins get BASELINE_WEIGHT_RATIO times
# a single band's mass -- enough to constrain the per-trace baseline param
# without letting the broadband floor compete with the peak fits.
BASELINE_WEIGHT_RATIO = 0.25

# Soft-prior strength on the f0 parameter of each mode. The expected-band
# hints from the workflow are treated as *preferences*, not walls: the
# optimiser sees the full fit window as bounds, and a quadratic penalty is
# added to the cost that grows only when f0 strays OUTSIDE its expected
# band. Cost contribution is ``lambda * (excess / halfwidth)^2`` where
# ``excess = max(0, |f0 - centre| - halfwidth)``. Calibrated so the prior
# stays subdominant to the mass-normalised per-band data cost (~0.05):
# at lambda=0.1 a 1x-halfwidth violation adds ~0.1 squared-residual which
# swamps the data and can drag f0 several Hz away from a legitimate peak
# outside the band. At lambda=0.01 the same violation adds ~0.01 -- still
# enough to break identifiability ties but small enough that a clear
# out-of-band peak wins. Set to 0.0 to disable.
BAND_PRIOR_LAMBDA = 0.01

# Sigma sanity thresholds. When a Jacobian-derived parameter sigma exceeds
# one of these, the sigma is replaced with NaN before being returned by
# `run_fit`. The fitted *value* is preserved -- only the uncertainty is
# masked, so downstream plots render the point but contribute nothing to
# axes autoscale (no exploding errorbar / CI band from degenerate fits).
SIGMA_DEGENERATE_F0_BAND_FRAC = 0.50   # sigma_f0 > 0.5 * configured band width
SIGMA_DEGENERATE_F0_REL       = 0.20   # sigma_f0 / f0 > 20 %
SIGMA_DEGENERATE_ZETA_ABS     = 0.25   # sigma_zeta > 0.25 (half of [ZETA_MIN, ZETA_MAX])
SIGMA_DEGENERATE_ZETA_REL     = 1.00   # sigma_zeta > zeta (uncertainty exceeds value)
SIGMA_DEGENERATE_LOG_AMP      = 2.00   # sigma_log_amp > 2 (amp factor of e^2 ~ 7.4)

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
    denom = (omega0**2 - omega**2)**2 + (2.0 * zeta * omega0 * omega)**2
    return 1.0 / np.maximum(denom, _LOG_EPS)


def _model_psds(packed: np.ndarray, freqs_hz: np.ndarray,
                n_bands: int, n_traces: int,
                f_ref: float | None = None) -> np.ndarray:
    """Reconstruct per-trace model PSDs from the packed parameter vector.

    Two packing layouts are supported, distinguished by ``f_ref``:

    ``f_ref is None`` (legacy flat baseline):
        [ band0_f0, band0_zeta, band0_logamp_t0, ..., band0_logamp_tN,
          band1_f0, band1_zeta, ...,
          baseline_logamp_t0, ..., baseline_logamp_tN ]

    ``f_ref`` supplied (log-linear sloped baseline):
        [ ... same band block ...,
          baseline_logamp_t0, ..., baseline_logamp_tN,
          slope_t0,           ..., slope_tN            ]

    The sloped variant produces ``B_j(f) = exp(log_base_j + slope_j*log(f/f_ref))``
    -- a power-law floor that absorbs the broadband roll-off which
    otherwise inflates fitted zeta.
    """
    cols = 2 + n_traces
    band_params = packed[:n_bands * cols].reshape(n_bands, cols)
    tail = packed[n_bands * cols:]
    log_baselines = tail[:n_traces]
    amps = np.exp(band_params[:, 2:])                    # (n_bands, n_traces)
    shapes = _lorentz_shape(freqs_hz, band_params[:, :2])  # (n_bands, nf)
    if f_ref is not None and tail.size >= 2 * n_traces:
        slopes = tail[n_traces:2 * n_traces]
        log_f_ratio = np.log(np.maximum(freqs_hz, _LOG_EPS) / f_ref)
        base_bins = np.exp(log_baselines[:, None]
                           + slopes[:, None] * log_f_ratio[None, :])  # (n_traces, nf)
        return amps.T @ shapes + base_bins
    baselines = np.exp(log_baselines)                    # (n_traces,)
    return amps.T @ shapes + baselines[:, None]


def _residuals_log(packed: np.ndarray, freqs_fit: np.ndarray,
                   meas_norm: np.ndarray, n_bands: int, n_traces: int,
                   weights: np.ndarray | None,
                   band_centres: np.ndarray | None = None,
                   band_halfwidths: np.ndarray | None = None,
                   prior_lambda: float = 0.0,
                   f_ref: float | None = None) -> np.ndarray:
    """Per-bin log-space residual vector (used by least_squares + scalar cost).

    When ``prior_lambda > 0`` and ``band_centres`` / ``band_halfwidths`` are
    supplied, appends ``n_bands`` extra residual entries encoding the soft
    band-centre prior (zero inside each expected band, quadratically
    growing outside). When ``f_ref`` is supplied the packed vector is
    interpreted with a sloped baseline (see :func:`_model_psds`).
    """
    models = _model_psds(packed, freqs_fit, n_bands, n_traces, f_ref=f_ref)
    res = (np.log(np.maximum(models, _LOG_EPS))
           - np.log(np.maximum(meas_norm, _LOG_EPS)))
    if weights is not None:
        res = res * np.sqrt(weights)
    out = res.ravel()
    if prior_lambda > 0.0 and band_centres is not None:
        cols = 2 + n_traces
        f0s = packed[:n_bands * cols].reshape(n_bands, cols)[:, 0]
        excess = np.maximum(np.abs(f0s - band_centres) - band_halfwidths, 0.0)
        prior_res = np.sqrt(prior_lambda) * (excess / band_halfwidths)
        out = np.concatenate([out, prior_res])
    return out


def _scalar_cost(packed: np.ndarray, freqs_fit: np.ndarray,
                 meas_norm: np.ndarray, n_bands: int, n_traces: int,
                 weights: np.ndarray,
                 band_centres: np.ndarray | None = None,
                 band_halfwidths: np.ndarray | None = None,
                 prior_lambda: float = 0.0,
                 f_ref: float | None = None) -> float:
    r = _residuals_log(packed, freqs_fit, meas_norm, n_bands, n_traces,
                       weights, band_centres, band_halfwidths, prior_lambda,
                       f_ref=f_ref)
    return float(np.dot(r, r))


# ----------------------------------------------------------------------------
# Initial-guess construction
# ----------------------------------------------------------------------------

def _half_power_zeta(freqs: np.ndarray, psd: np.ndarray, peak_idx: int,
                     z_min: float = 0.05, z_max: float = 0.30) -> float:
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


def _pick_seed_peaks(freqs_fit: np.ndarray, smooth_psd: np.ndarray,
                     bands: list[tuple[float, float]]
                     ) -> list[tuple[float, int] | None]:
    """Assign one PSD peak to each expected band by minimum-distance cost.

    The simple ``np.argmax`` seed used previously flips between competing
    peaks when two prominent lobes share a search window (e.g. a 4.4 Hz
    wheel-hop ridge next to the 5.5 Hz heave peak). Instead we:

    1. Find all prominent peaks across the UNION of the configured bands.
    2. Solve a Hungarian-style assignment minimising the total distance
       from each peak frequency to each band centre.

    Returns a list of ``(f_peak, peak_idx_in_full_array)`` per band, or
    ``None`` for bands that had no candidate peak (caller falls back to
    ``argmax`` for those).
    """
    if not bands:
        return []
    union_lo = min(float(b[0]) for b in bands)
    union_hi = max(float(b[1]) for b in bands)
    mask = (freqs_fit >= union_lo) & (freqs_fit <= union_hi)
    if not np.any(mask):
        return [None] * len(bands)
    region_f = freqs_fit[mask]
    region_p = smooth_psd[mask]
    region_idx = np.flatnonzero(mask)
    # Prominence threshold scales with the region's peak so noise wiggles
    # don't qualify. 2 % of peak is a conservative floor.
    p_max = float(region_p.max()) if region_p.size else 0.0
    if p_max <= 0.0:
        return [None] * len(bands)
    peaks_local, _ = find_peaks(region_p, prominence=0.02 * p_max)
    if peaks_local.size == 0:
        return [None] * len(bands)
    cand_f = region_f[peaks_local]
    cand_full_idx = region_idx[peaks_local]
    centres = np.array([0.5 * (lo + hi) for lo, hi in bands])
    # Cost: |peak_f - band_centre|. Heavily penalise peaks that fall
    # outside the band's expanded envelope (1.5x band width either side)
    # so the assignment still prefers peaks near the expected centre.
    widths = np.array([max(hi - lo, 1e-6) for lo, hi in bands])
    cost = np.abs(cand_f[:, None] - centres[None, :])
    out_of_envelope = cost > 1.5 * widths[None, :]
    cost = cost + np.where(out_of_envelope, 10.0 * widths[None, :], 0.0)
    if cand_f.size >= len(bands):
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # Fewer peaks than bands: greedy per-band, leave the rest None.
        row_ind = []
        col_ind = []
        unused = list(range(cand_f.size))
        for b in range(len(bands)):
            if not unused:
                break
            best = min(unused, key=lambda k: cost[k, b])
            row_ind.append(best)
            col_ind.append(b)
            unused.remove(best)
    assignments: list[tuple[float, int] | None] = [None] * len(bands)
    for k, b in zip(row_ind, col_ind):
        assignments[b] = (float(cand_f[k]), int(cand_full_idx[k]))
    return assignments


def _initial_guess(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                   bands: list[tuple[float, float]],
                   n_traces: int) -> np.ndarray:
    """Peak-pick + half-power damping seed for each expected band."""
    cols = 2 + n_traces
    x0 = np.zeros(len(bands) * cols + n_traces)
    mean_trace = np.mean(meas_norm, axis=0)
    if len(mean_trace) >= 5:
        mean_smooth = np.convolve(mean_trace, np.ones(5) / 5.0, mode="same")
    else:
        mean_smooth = mean_trace
    assignments = _pick_seed_peaks(freqs_fit, mean_smooth, bands)
    for i, (lo, hi) in enumerate(bands):
        seed = assignments[i] if i < len(assignments) else None
        if seed is not None:
            f_peak, idx_full = seed
            zeta0 = _half_power_zeta(freqs_fit, mean_smooth, idx_full)
            meas_peak = float(max(mean_smooth[idx_full], 1e-6))
        else:
            # Fallback: argmax inside the band envelope.
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
        # Solve for amplitude that places the Lorentzian peak at the measured peak.
        omega0 = 2.0 * np.pi * f_peak
        shape_peak = 1.0 / (2.0 * zeta0 * omega0**2)**2
        log_amp = float(np.log(meas_peak / shape_peak))
        base = i * cols
        x0[base] = f_peak
        x0[base + 1] = zeta0
        x0[base + 2:base + cols] = log_amp
    # Per-trace baseline seed: 5th-percentile of the measured trace.
    for t in range(n_traces):
        floor_seed = float(max(np.percentile(meas_norm[t], 5.0), 1e-9))
        x0[len(bands) * cols + t] = float(np.log(floor_seed))
    return x0


# ----------------------------------------------------------------------------
# Single-subsystem fit (heave+pitch OR roll+warp)
# ----------------------------------------------------------------------------

def _expanded_bounds(freqs_fit: np.ndarray, bands: list[tuple[float, float]],
                     expansion: float = 0.3,
                     min_margin: float = 0.5) -> list[tuple[float, float]]:
    """Return the full fit window as f0 bounds for every band.

    The ``expected_freqs`` entries supplied by the workflow are treated as
    *soft* priors, not hard walls: the optimiser is free to place any
    mode's f0 anywhere in the analysis window, and the ``BAND_PRIOR_LAMBDA``
    penalty in the cost function pulls it back toward its expected centre
    only when the data doesn't argue otherwise. This lets bootstrap CIs
    extend below/above the nominal band edges (fixing the one-sided CI
    artefact caused by walls sitting on the point estimate) and lets the
    fit report the true f0 when the actual peak is a bit outside the
    hint. Mode ordering within each subsystem is still enforced by the
    ``MIN_BAND_GAP_HZ`` seed-based partition inside :func:`_fit_subsystem`.

    ``expansion`` and ``min_margin`` are kept in the signature for
    backward compatibility but no longer used.
    """
    del expansion, min_margin  # bounds are now the full fit window
    fit_lo, fit_hi = float(np.min(freqs_fit)), float(np.max(freqs_fit))
    return [(fit_lo, fit_hi) for _ in bands]


def _zeta_pinned_to_bound(x: np.ndarray, n_bands: int, n_traces: int,
                          margin: float = 0.02) -> bool:
    """True if any fitted damping ratio sits at the edge of `[ZETA_MIN, ZETA_MAX]`."""
    cols = 2 + n_traces
    zetas = x[:n_bands * cols].reshape(n_bands, cols)[:, 1]
    span = ZETA_MAX - ZETA_MIN
    return bool(np.any(zetas <= ZETA_MIN + margin * span)
                or np.any(zetas >= ZETA_MAX - margin * span))


def _fit_subsystem(freqs_fit: np.ndarray, meas_norm: np.ndarray,
                   bands: list[tuple[float, float]],
                   coherence: np.ndarray | None,
                   n_avg: int | None = None,
                   x0_seed: np.ndarray | None = None):
    """Fit (f0, zeta, amplitudes) for one subsystem (2 bands, 2 traces).

    Returns (params, baselines, sigma_params, r_squared).

    When ``n_avg`` (the number of Welch segments averaged into ``meas_norm``)
    is supplied, parameter sigmas are derived in absolute-sigma form using
    the known log-amplitude noise variance ``trigamma(K)`` of a K-averaged
    chi-squared PSD bin. Otherwise the legacy MSE-scaled covariance is used.

    ``x0_seed`` short-circuits both the peak-picking initial guess and the
    differential-evolution global step: pass a ``(n_bands, 2 + n_traces)``
    array of ``(f0, zeta, amp_1, ..., amp_n)`` in *linear* amplitude units
    (the seed is converted to log-amp internally) and only the local
    L-BFGS-B + least_squares polish is run. Used by the segment-bootstrap
    driver so each bootstrap iteration stays near the point-estimate basin
    and avoids mode-swap chaos.
    """
    n_traces, n_bands = meas_norm.shape[0], len(bands)
    cols = 2 + n_traces
    # Reference frequency for the sloped-baseline power law (log-linear
    # in log(f/f_ref)). Geometric mean of the fit window centres the
    # slope parameter so log(f/f_ref) is roughly antisymmetric about 0
    # across the window.
    f_ref = float(np.sqrt(freqs_fit[0] * freqs_fit[-1])) if freqs_fit.size else 1.0
    # Bound the per-trace slope so the baseline can capture broadband
    # roll-off. The physical range (log-log slope of tyre/road excitation
    # PSD) sits in +/-1 to +/-2; widen to +/-6 so the LS polish can walk
    # freely on sessions with strong wheel-hop bleed above 11 Hz that pull
    # the effective floor up steeply. If the slope pins at +/-6 the fit is
    # flagged for review (see _sanitise_sigmas warning).
    _SLOPE_MIN, _SLOPE_MAX = -6.0, 6.0

    # ---- bounds: (f0, zeta, log_amp x n_traces) per band, then
    #      (log_baseline x n_traces) + (slope x n_traces)
    bounds = []
    for lo, hi in _expanded_bounds(freqs_fit, bands):
        bounds.append((lo, hi))
        bounds.append((ZETA_MIN, ZETA_MAX))
        bounds.extend([(np.log(1e-6), np.log(1e6))] * n_traces)
    for t in range(n_traces):
        peak_t = float(np.max(meas_norm[t]))
        # With the sloped baseline B(f)=B_j*(f/f_ref)^alpha the anchor B_j
        # only sets the value at f_ref; the effective PSD floor across the
        # band varies with alpha. Cap B_j at 0.5*peak_t (not the tighter
        # 0.3*peak_t used with the legacy flat baseline) to give the LS
        # polish room to trade baseline height against slope on sessions
        # with strong broadband roll-off. Empirical: at 0.3*peak_t the LS
        # traps into slope=-4 pin + Warp zeta=0.5 pin on RED_P1 / BLUE_P1;
        # at 0.5*peak_t it recovers the physical fit (slope~-0.5, zeta~0.16).
        base_hi = max(0.5 * peak_t, 1e-9)
        bounds.append((np.log(1e-9), np.log(base_hi)))
    for t in range(n_traces):
        bounds.append((_SLOPE_MIN, _SLOPE_MAX))

    # ---- Cost weights.
    #
    # Design (revised 2026-07 pt.2):
    #   Log-space residuals already have the correct noise model for a
    #   Welch PSD (approximately Gaussian log-noise with variance
    #   trigamma(K)), so the maximum-likelihood weight per bin is uniform.
    #   Prior versions applied a ``sqrt(meas/trace_mean)`` peak-tip
    #   emphasis on top of that -- it made f0 more robust against
    #   non-Lorentzian shoulders, but it downweighted the shoulder bins
    #   that carry the damping-ratio information. That produced inflated
    #   zeta uncertainty. Uniform-in-log weights restore proper zeta
    #   identifiability; f0 robustness comes from the soft band prior
    #   (BAND_PRIOR_LAMBDA) instead.
    #
    #   Coherence is still applied (low-coherence bins are noise-driven
    #   and shouldn't participate in the fit), and each configured band
    #   is mass-normalised so a tall peak (Roll) doesn't drown out a
    #   weaker peak (Warp) in the same subsystem.
    if coherence is None:
        weights = np.ones_like(meas_norm)
    else:
        coh = np.asarray(coherence, dtype=float)
        coh = np.where(np.isfinite(coh) & (coh > 0), coh, 0.0)
        weights = np.broadcast_to(coh[None, :], meas_norm.shape).copy()

    band_masks = [(freqs_fit >= lo) & (freqs_fit <= hi) for lo, hi in bands]
    in_any_band = np.any(np.stack(band_masks), axis=0) if band_masks else \
        np.zeros_like(freqs_fit, dtype=bool)
    background_mask = ~in_any_band
    for mask in band_masks:
        if not mask.any():
            continue
        total = float(weights[:, mask].sum())
        if total > 0.0:
            weights[:, mask] *= 1.0 / total
    if background_mask.any():
        total_bg = float(weights[:, background_mask].sum())
        if total_bg > 0.0:
            weights[:, background_mask] *= BASELINE_WEIGHT_RATIO / total_bg

    # Soft f0 prior: centre + halfwidth per configured band. Applied as a
    # quadratic penalty in _residuals_log with strength BAND_PRIOR_LAMBDA.
    band_centres = np.array([0.5 * (lo + hi) for lo, hi in bands], dtype=float)
    band_halfwidths = np.maximum(
        np.array([0.5 * (hi - lo) for lo, hi in bands], dtype=float), 1e-6
    )
    args = (freqs_fit, meas_norm, n_bands, n_traces, weights,
            band_centres, band_halfwidths, BAND_PRIOR_LAMBDA, f_ref)
    n_data_rows = int(weights.size)

    # Slope seeds start at zero. Data-driven OLS on the "background" bins
    # sounds physical but the out-of-band bins are contaminated by the
    # Lorentzian tails (which are strong within +/- a couple of bandwidths
    # of each mode) so OLS gives a badly biased slope. Trust the LS polish
    # to walk slopes off zero.
    slope_seeds = np.zeros(n_traces)

    if x0_seed is None:
        x0_bands = _initial_guess(freqs_fit, meas_norm, bands, n_traces)
        x0 = np.concatenate([x0_bands, slope_seeds])
    else:
        # Pack (f0, zeta, log_amp x n_traces) per band + baseline seeds
        # from a supplied point-estimate result. Baselines fall back to
        # the same 5th-percentile heuristic as _initial_guess; slopes
        # start at 0 (LS polish will restore the point-estimate slope).
        seed = np.asarray(x0_seed, dtype=float)
        if seed.shape != (n_bands, cols):
            raise ValueError(
                f"x0_seed shape {seed.shape} != expected ({n_bands}, {cols})"
            )
        x0 = np.zeros(n_bands * cols + 2 * n_traces)
        for i in range(n_bands):
            base = i * cols
            x0[base] = seed[i, 0]
            x0[base + 1] = seed[i, 1]
            x0[base + 2:base + cols] = np.log(
                np.maximum(seed[i, 2:], 1e-12)
            )
        for t in range(n_traces):
            floor_seed = float(max(np.percentile(meas_norm[t], 5.0), 1e-9))
            x0[n_bands * cols + t] = float(np.log(floor_seed))
        # Bootstrap path: seed slopes from the same background-OLS estimate
        # so each draw starts near the point-estimate slope basin instead of
        # at zero (which caused L-BFGS-B to over-shoot toward the slope bound
        # on some sessions).
        x0[n_bands * cols + n_traces:n_bands * cols + 2 * n_traces] = slope_seeds

    # Step 2 -- enforce a minimum frequency gap between adjacent bands by
    # partitioning the f0 search range at the midpoint between the seeded
    # peaks. This is an *ordering identifiability* constraint (mode k must
    # sit below mode k+1) rather than a location prior -- output params
    # are always sorted by f0 anyway, so without it the two Lorentzians
    # in a subsystem could collapse onto the same peak. The soft
    # BAND_PRIOR_LAMBDA term carries the location bias.
    if n_bands >= 2:
        seed_freqs = [float(x0[i * cols]) for i in range(n_bands)]
        order = sorted(range(n_bands), key=lambda i: seed_freqs[i])
        for k in range(len(order) - 1):
            i_lo, i_hi = order[k], order[k + 1]
            f_lo, f_hi = seed_freqs[i_lo], seed_freqs[i_hi]
            if f_hi - f_lo < MIN_BAND_GAP_HZ:
                # Seeds are already too close to partition cleanly; leave
                # the original bounds in place so the optimiser still has
                # somewhere to go.
                continue
            midpoint = 0.5 * (f_lo + f_hi)
            lo_b = list(bounds[i_lo * cols])
            hi_b = list(bounds[i_hi * cols])
            lo_b[1] = min(lo_b[1], midpoint)  # cap lower-band f0 upper bound
            hi_b[0] = max(hi_b[0], midpoint)  # lift upper-band f0 lower bound
            bounds[i_lo * cols] = tuple(lo_b)
            bounds[i_hi * cols] = tuple(hi_b)

    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])
    x0 = np.clip(x0, lo_arr, hi_arr)

    # ---- Sloped-baseline architecture: LBFGS+DE search the 10-D flat model
    # (band params + flat baselines), then release the slopes for the LS
    # polish. This preserves the DE fallback that guards against zeta
    # pinning without paying its cost in the extended 12-D space where
    # slopes and Lorentzian tails are near-degenerate. Verified fix for
    # RED_P1 where the joint 12-D LBFGS pinned slope=-4 and zeta=0.50.
    n_flat = n_bands * cols + n_traces
    lo_flat = lo_arr[:n_flat]; hi_flat = hi_arr[:n_flat]
    bounds_flat = [(float(lo_flat[i]), float(hi_flat[i])) for i in range(n_flat)]
    x0_flat = x0[:n_flat]
    args_flat = (freqs_fit, meas_norm, n_bands, n_traces, weights,
                 band_centres, band_halfwidths, BAND_PRIOR_LAMBDA, None)

    # ---- Effort budgets. From a supplied point-estimate seed (bootstrap
    # path) the fit is a small perturbation, so cap L-BFGS-B and LS
    # iterations aggressively. Otherwise use the full budget.
    if x0_seed is not None:
        lbfgs_opts = {"maxiter": 60, "ftol": 1e-8}
        ls_max_nfev = 120
    else:
        lbfgs_opts = {"maxiter": 200, "ftol": 1e-9}
        ls_max_nfev = 400

    # ---- Step 1: cheap L-BFGS-B on the flat model.
    local = minimize(_scalar_cost, x0_flat, args=args_flat, method="L-BFGS-B",
                     bounds=bounds_flat, options=lbfgs_opts)
    if x0_seed is not None:
        # Bootstrap path: stay near the supplied seed. Skip DE regardless
        # of local success; the trust-region polish below refines and the
        # bootstrap driver captures whatever variance the local step sees.
        best_x_flat = local.x if local.success else x0_flat
    elif local.success and not _zeta_pinned_to_bound(local.x, n_bands, n_traces):
        best_x_flat = local.x
    else:
        # ---- Step 2: differential evolution fallback (global, flat model).
        de = differential_evolution(
            _scalar_cost, bounds=bounds_flat, args=args_flat, x0=x0_flat, seed=42,
            polish=True, disp=False, updating="immediate", workers=1,
            popsize=8, init="sobol", tol=1e-2, mutation=(0.5, 1.5),
            maxiter=120,
        )
        best_x_flat = de.x
        if _zeta_pinned_to_bound(best_x_flat, n_bands, n_traces):
            retry = differential_evolution(
                _scalar_cost, bounds=bounds_flat, args=args_flat, seed=7,
                polish=True, disp=False, updating="immediate", workers=1,
                popsize=8, init="sobol", tol=1e-2, mutation=(0.5, 1.5),
                maxiter=120,
            )
            if retry.fun < de.fun:
                best_x_flat = retry.x

    # Expand to the sloped-model parameter space (append slope=0 seeds).
    best_x = np.concatenate([best_x_flat, slope_seeds])

    # ---- Step 3: least_squares polish in log space on the FULL sloped
    # model. Releases the per-trace slopes for the first time, so the
    # baseline can absorb broadband roll-off that was previously being
    # forced into the Lorentzian widths.
    margin = 1e-8 * np.maximum(np.abs(hi_arr - lo_arr), 1.0)
    x_polish = np.clip(best_x, lo_arr + margin, hi_arr - margin)
    sigmas_x = np.full_like(best_x, np.nan, dtype=float)
    r_squared = float("nan")

    # Also run a direct LS pass from the RAW initial_guess (peak-pick + flat
    # baseline + slope=0). On sessions with strong broadband roll-off the
    # LBFGS-flat + LS-sloped path lands in a local minimum where the
    # Lorentzian tails have absorbed the roll-off (zeta pinned high) and
    # LS cannot escape by moving slope alone. Direct LS from the raw seed
    # explores the joint (zeta, slope) subspace freely and often finds a
    # basin with much lower cost. Keep the lower-cost candidate.
    x_raw_seed = np.concatenate([
        _initial_guess(freqs_fit, meas_norm, bands, n_traces),
        slope_seeds,
    ])
    x_raw_seed = np.clip(x_raw_seed, lo_arr + margin, hi_arr - margin)
    try:
        ls_raw = least_squares(_residuals_log, x_raw_seed,
                               bounds=(lo_arr, hi_arr),
                               args=args, method="trf",
                               max_nfev=ls_max_nfev)
    except Exception:  # noqa: BLE001
        ls_raw = None

    try:
        ls = least_squares(_residuals_log, x_polish, bounds=(lo_arr, hi_arr),
                           args=args, method="trf", max_nfev=ls_max_nfev)
        # Compare cost against the raw-seed alternative; keep the winner.
        if ls_raw is not None:
            cost_polish = float(np.dot(ls.fun, ls.fun))
            cost_raw = float(np.dot(ls_raw.fun, ls_raw.fun))
            if cost_raw < cost_polish:
                ls = ls_raw
        best_x = ls.x
        # Covariance: prefer the absolute-sigma form using the known Welch
        # log-PSD noise variance (trigamma(K)). The per-band mass weights
        # are a fit-quality heuristic, not a noise model, so we recover the
        # unweighted Jacobian by dividing ls.jac by sqrt(weights). The
        # tail of ls.jac corresponds to the soft-prior residuals -- those
        # are regularisation, not data, so they're excluded from the
        # noise-based covariance.
        try:
            data_jac = ls.jac[:n_data_rows, :]
            data_fun = ls.fun[:n_data_rows]
            if n_avg is not None and int(n_avg) >= 1 and ls.jac is not None:
                sigma_log_sq = float(polygamma(1, int(n_avg)))
                sqrt_w_flat = np.sqrt(np.asarray(weights, dtype=float).ravel())
                nonzero = sqrt_w_flat > 0
                safe_w = np.where(nonzero, sqrt_w_flat, 1.0)
                jac_uw = data_jac / safe_w[:, None]
                if np.any(~nonzero):
                    jac_uw[~nonzero, :] = 0.0
                cov = np.linalg.inv(jac_uw.T @ jac_uw) * sigma_log_sq
            else:
                ssr = float(np.dot(data_fun, data_fun))
                dof = max(data_fun.size - x_polish.size, 1)
                cov = np.linalg.inv(data_jac.T @ data_jac) * (ssr / dof)
            var = np.diag(cov)
            ok = np.isfinite(var) & (var > 0)
            sigmas_x = np.where(ok, np.sqrt(np.where(ok, var, 1.0)), np.nan)
        except np.linalg.LinAlgError:
            pass
        # R^2 in log space (unweighted, prior excluded)
        res_uw = _residuals_log(ls.x, freqs_fit, meas_norm,
                                n_bands, n_traces, None,
                                None, None, 0.0, f_ref)
        log_meas = np.log(np.maximum(meas_norm, _LOG_EPS)).ravel()
        ss_res = float(np.dot(res_uw, res_uw))
        ss_tot = float(np.sum((log_meas - log_meas.mean()) ** 2))
        if ss_tot > 0:
            r_squared = 1.0 - ss_res / ss_tot
    except Exception as exc:  # noqa: BLE001 - polish is best-effort
        log.debug("Lorentz polish failed: %s", exc)

    # ---- Unpack into linear-amplitude params + sort bands by f0.
    params = best_x[:n_bands * cols].reshape(n_bands, cols).copy()
    params[:, 2:] = np.exp(params[:, 2:])
    tail = best_x[n_bands * cols:]
    baselines = np.exp(tail[:n_traces])
    slopes = tail[n_traces:2 * n_traces].copy() if tail.size >= 2 * n_traces \
        else np.zeros(n_traces)
    sigmas = sigmas_x[:n_bands * cols].reshape(n_bands, cols).copy()
    order = np.argsort(params[:, 0])
    return params[order], baselines, slopes, sigmas[order], r_squared


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def _sanitise_sigmas(params: np.ndarray, sigmas: np.ndarray,
                     bands: list[tuple[float, float]],
                     mode_labels: list[str],
                     quiet: bool = False) -> np.ndarray:
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
                why.append(f"sigma_f0={s_f0:.3f} > "
                           f"{SIGMA_DEGENERATE_F0_BAND_FRAC:.2f}*band_width"
                           f"({band_width:.2f})")
            if f0 > 0.0 and s_f0 / f0 > SIGMA_DEGENERATE_F0_REL:
                why.append(f"sigma_f0/f0={s_f0/f0:.2f} > "
                           f"{SIGMA_DEGENERATE_F0_REL:.2f}")
            if why:
                sanitised[i, 0] = np.nan
                if not quiet:
                    log.warning("    %s degenerate f0 sigma (%s)",
                                label, "; ".join(why))
        if np.isfinite(s_zeta):
            why = []
            if s_zeta > SIGMA_DEGENERATE_ZETA_ABS:
                why.append(f"sigma_zeta={s_zeta:.3f} > "
                           f"{SIGMA_DEGENERATE_ZETA_ABS:.2f}")
            if zeta > 0.0 and s_zeta / zeta > SIGMA_DEGENERATE_ZETA_REL:
                why.append(f"sigma_zeta/zeta={s_zeta/zeta:.2f} > "
                           f"{SIGMA_DEGENERATE_ZETA_REL:.2f}")
            if why:
                sanitised[i, 1] = np.nan
                if not quiet:
                    log.warning("    %s degenerate zeta sigma (%s)",
                                label, "; ".join(why))
        for t in range(n_traces):
            s_log_amp = float(sigmas[i, 2 + t])
            if np.isfinite(s_log_amp) and s_log_amp > SIGMA_DEGENERATE_LOG_AMP:
                sanitised[i, 2 + t] = np.nan
                if not quiet:
                    log.warning("    %s degenerate amp sigma (%s, "
                                "sigma_log_amp=%.2f > %.1f)",
                                label, trace_names[t], s_log_amp,
                                SIGMA_DEGENERATE_LOG_AMP)
    return sanitised


def _shape_modes_from_amps(params: np.ndarray) -> np.ndarray:
    """Reconstruct (4, 4) body-DOF mode-shape matrix from fitted amplitudes.

    Mode ordering: 0=Heave, 1=Pitch, 2=Roll, 3=Warp.
    Body-DOF rows: 0=z_F, 1=theta_F, 2=z_R, 3=theta_R.

    Heave/Pitch live in rows (0, 2); Roll/Warp live in rows (1, 3).
    Pitch flips the rear sign; Warp flips the rear sign; each shape is
    normalised so the largest |entry| is 1.
    """
    shapes = np.zeros((4, 4), dtype=float)
    for i, row in enumerate(params):
        amp_front = float(row[2])
        amp_rear = float(row[3])
        front = np.sqrt(max(amp_front, 0.0))
        rear = np.sqrt(max(amp_rear, 0.0))
        if i == 0:    # Heave
            shapes[:, i] = [front, 0.0, rear, 0.0]
        elif i == 1:  # Pitch
            shapes[:, i] = [front, 0.0, -rear, 0.0]
        elif i == 2:  # Roll
            shapes[:, i] = [0.0, front, 0.0, rear]
        else:          # Warp
            shapes[:, i] = [0.0, front, 0.0, -rear]
        peak = np.max(np.abs(shapes[:, i]))
        if peak > 0:
            shapes[:, i] /= peak
    return shapes


def run_fit(freqs_fit: np.ndarray, body_psds_norm: np.ndarray,
            expected_bands: dict,
            coh_hp: np.ndarray | None = None,
            coh_rw: np.ndarray | None = None,
            n_avg: int | None = None,
            x0_seed: np.ndarray | None = None) -> dict:
    """Combined-Lorentzian fit across the 4 body DOFs.

    Parameters
    ----------
    freqs_fit : (nf,) array
        Frequency bins (Hz) inside the fit window.
    body_psds_norm : (4, nf) array
        Body-DOF PSDs normalised to a common peak, ordered
        [z_F, theta_F, z_R, theta_R].
    expected_bands : dict
        Per-mode (lo, hi, mid) tuples keyed by lowercase mode name.
    coh_hp, coh_rw : (nf,) arrays or None
        Optional coherence between z_F & z_R (heave/pitch) and between
        theta_F & theta_R (roll/warp). Used as per-bin weights.
    n_avg : int or None
        Number of Welch segments averaged into ``body_psds_norm``. When
        supplied, parameter sigmas are scaled in absolute terms using the
        known log-PSD noise variance ``trigamma(K)``. None falls back to
        the legacy MSE-scaled covariance.
    x0_seed : (4, 4) array or None
        Optional point-estimate seed used by the segment-bootstrap
        driver. Rows are ordered ``[Heave, Pitch, Roll, Warp]`` with
        columns ``(f0, zeta, amp_front, amp_rear)`` in linear units. When
        supplied, peak-picking and differential-evolution are skipped and
        only the local L-BFGS-B + least_squares polish runs.

    Returns
    -------
    dict with keys: method, params (4x4), baselines (4,), fn (4,),
        zeta (4,), amp_front (4,), amp_rear (4,), sigma_fn (4,),
        sigma_zeta (4,), sigma_amp_front (4,), sigma_amp_rear (4,),
        r_squared (R^2_hp, R^2_rw), mode_labels, mode_shapes (4x4).
    """
    bands_hp = [expected_bands["heave"][:2], expected_bands["pitch"][:2]]
    bands_rw = [expected_bands["roll"][:2],  expected_bands["warp"][:2]]
    quiet = x0_seed is not None
    if not quiet:
        log.info("  Fitting combined Lorentzians...")

    seed_hp = None
    seed_rw = None
    if x0_seed is not None:
        seed_arr = np.asarray(x0_seed, dtype=float)
        if seed_arr.shape != (4, 4):
            raise ValueError(
                f"x0_seed must be (4, 4) [Heave/Pitch/Roll/Warp x "
                f"(f0, zeta, amp_front, amp_rear)]; got {seed_arr.shape}"
            )
        seed_hp = seed_arr[[0, 1]]
        seed_rw = seed_arr[[2, 3]]

    # Subsystem 1: heave/pitch share z_F / z_R amplitudes.
    params_hp, base_hp, slopes_hp, sig_hp, r2_hp = _fit_subsystem(
        freqs_fit, body_psds_norm[[0, 2]], bands_hp, coh_hp,
        n_avg=n_avg, x0_seed=seed_hp)
    sig_hp = _sanitise_sigmas(params_hp, sig_hp, bands_hp,
                              ["Heave", "Pitch"], quiet=quiet)
    # Subsystem 2: roll/warp share theta_F / theta_R amplitudes.
    params_rw, base_rw, slopes_rw, sig_rw, r2_rw = _fit_subsystem(
        freqs_fit, body_psds_norm[[1, 3]], bands_rw, coh_rw,
        n_avg=n_avg, x0_seed=seed_rw)
    sig_rw = _sanitise_sigmas(params_rw, sig_rw, bands_rw,
                              ["Roll", "Warp"], quiet=quiet)

    params = np.vstack((params_hp, params_rw))
    sigmas = np.vstack((sig_hp, sig_rw))
    baselines = np.array([base_hp[0], base_rw[0], base_hp[1], base_rw[1]])
    # baseline_slopes ordered like baselines: [z_F, theta_F, z_R, theta_R]
    baseline_slopes = np.array([slopes_hp[0], slopes_rw[0],
                                slopes_hp[1], slopes_rw[1]])
    # Reference frequency used by the sloped-baseline model. Downstream
    # evaluators (eval_psds, diagnosis plot) need this to reconstruct
    # the per-bin baseline.
    baseline_f_ref = float(np.sqrt(freqs_fit[0] * freqs_fit[-1])) \
        if freqs_fit.size else 1.0

    # Delta-method: sigma_amp ~= amp * sigma_log_amp.
    amp_front = params[:, 2]
    amp_rear = params[:, 3]
    sigma_amp_front = np.where(np.isfinite(sigmas[:, 2]),
                               amp_front * sigmas[:, 2], np.nan)
    sigma_amp_rear = np.where(np.isfinite(sigmas[:, 3]),
                              amp_rear * sigmas[:, 3], np.nan)

    return {
        "method": "lorentzian_combined",
        "params": params,
        "baselines": baselines,
        "baseline_slopes": baseline_slopes,
        "baseline_f_ref": baseline_f_ref,
        "fn": params[:, 0],
        "zeta": params[:, 1],
        "amp_front": amp_front,
        "amp_rear": amp_rear,
        "sigma_fn": sigmas[:, 0],
        "sigma_zeta": sigmas[:, 1],
        "sigma_amp_front": sigma_amp_front,
        "sigma_amp_rear": sigma_amp_rear,
        "r_squared": (float(r2_hp), float(r2_rw)),
        "mode_labels": list(MODE_ORDER),
        "mode_shapes": _shape_modes_from_amps(params),
    }


def eval_psds(result: dict, freqs_hz: np.ndarray,
              include_baseline: bool = True) -> np.ndarray:
    """Evaluate the fitted Lorentzian PSDs at `freqs_hz` for all 4 body DOFs.

    Returned shape: (4, len(freqs_hz)), rows = [z_F, theta_F, z_R, theta_R].

    Row 0 (z_F) and row 2 (z_R) sum the heave + pitch bands, picking
    front vs rear amplitude. Row 1 (theta_F) and row 3 (theta_R) sum the
    roll + warp bands the same way. When the result carries per-trace
    ``baseline_slopes`` and ``baseline_f_ref`` (V2 sloped-baseline model)
    the baseline term becomes ``B_j * (f/f_ref)^slope_j`` per bin.
    """
    params = result["params"]                       # (4, 4): bands x [f0,zeta,af,ar]
    baselines = result.get("baselines")
    slopes = result.get("baseline_slopes")
    f_ref = result.get("baseline_f_ref")
    if baselines is None or not include_baseline:
        baselines = np.zeros(4)
        slopes = np.zeros(4)
        f_ref = None
    if slopes is None or f_ref is None:
        slopes = np.zeros(len(baselines))
        f_ref = None
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    # (band_rows, amp_col, baseline_idx) for each output DOF row.
    specs = [
        ([0, 1], 2, 0),   # z_F     <- heave/pitch front amplitudes
        ([2, 3], 2, 1),   # theta_F <- roll/warp  front amplitudes
        ([0, 1], 3, 2),   # z_R     <- heave/pitch rear amplitudes
        ([2, 3], 3, 3),   # theta_R <- roll/warp  rear amplitudes
    ]
    out = []
    for band_rows, amp_col, base_idx in specs:
        sub = params[band_rows][:, [0, 1, amp_col]]   # (n_bands, 3)
        shapes = _lorentz_shape(freqs_hz, sub[:, :2]) # (n_bands, nf)
        if f_ref is not None:
            log_f_ratio = np.log(np.maximum(freqs_hz, _LOG_EPS) / float(f_ref))
            base_bins = float(baselines[base_idx]) * np.exp(
                float(slopes[base_idx]) * log_f_ratio
            )
        else:
            base_bins = float(baselines[base_idx])
        out.append(sub[:, 2] @ shapes + base_bins)
    return np.stack(out)
