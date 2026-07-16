"""Segment-bootstrap confidence intervals for the Lorentz modal fit.

Option B of the CI methodology: resample the *Welch segments* underlying
each body-PSD estimate, refit the combined Lorentzian model seeded from
the point estimate, and report percentile CIs on
``(fn, zeta, amp_front, amp_rear)`` for every mode.

Block size 1 (default): the measured lag-1 autocorrelation of the
spectrogram over segment index is ~0.12-0.30 in the Warp band across
the 26R09SIL BLUE data, well below the 0.5 threshold at which a block
bootstrap is required to remove correlation bias. Larger blocks add
sampling variance without reducing bias -- verified in
``tools/diagnose_zeta_ci.py`` EXP D (block=1 gave the narrowest CI).
Set ``block_size >= 2`` when the underlying data is known to have
long-range autocorrelation (e.g. resample-rate << peak Lorentz width).

Design notes
------------
* We do **not** resample coherence — those weights are frozen from the
  point estimate, so every bootstrap fit sees the same weighting kernel.
  Bootstrapping coherence would conflate PSD-noise variance with the
  independent frequency-dependent weighting, which is not what we want
  the CI to represent.
* The peak-picking initial guess and differential-evolution global step
  are skipped inside every bootstrap iteration (``x0_seed`` short-circuit
  in :func:`engine.vibrations_lorentz.run_fit`); this keeps each draw
  ~50 ms and, more importantly, avoids mode-swap chaos where a bootstrap
  iteration lands in a different local basin than the point estimate.
* The absolute-sigma covariance path in ``run_fit`` still runs (with
  ``n_avg`` reflecting the number of *resampled* segments) but its
  parametric sigmas are unused — the bootstrap replaces them with
  percentile CIs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from . import vibrations_lorentz
from .logger import log
from .vibrations_io import compute_body_psds

_MA3 = np.ones(3) / 3.0


def _smooth_and_normalise(psds_fit: np.ndarray) -> np.ndarray:
    """Same medfilt5 + ma3 + peak-normalise pipeline as the point fit."""
    smooth = np.stack([np.convolve(signal.medfilt(p, kernel_size=5), _MA3, mode="same") for p in psds_fit])
    peak = float(np.max(smooth))
    if peak <= 0.0:
        peak = 1.0
    return smooth / peak


def _params_to_seed(params: np.ndarray) -> np.ndarray:
    """Convert a (4, 4) ``result['params']`` matrix into ``x0_seed`` form.

    ``params`` rows are [Heave, Pitch, Roll, Warp] with columns
    ``(f0, zeta, amp_front, amp_rear)`` in *linear* amplitude - which is
    exactly the layout :func:`vibrations_lorentz.run_fit` expects.
    """
    arr = np.asarray(params, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"expected params of shape (4, 4); got {arr.shape}")
    # Clip to interior of bounds so L-BFGS-B doesn't start on a wall.
    arr = arr.copy()
    arr[:, 0] = np.clip(arr[:, 0], 0.5, 30.0)
    arr[:, 1] = np.clip(arr[:, 1], 0.021, 0.499)
    arr[:, 2:] = np.clip(arr[:, 2:], 1e-9, None)
    return arr


def _block_indices(n_seg: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw block-bootstrap indices covering ``n_seg`` segments.

    Blocks of ``block_size`` consecutive segments are drawn with
    replacement, starting positions uniform on ``[0, n_seg - block_size]``.
    The concatenated sequence is truncated to length ``n_seg``.
    """
    if block_size < 1:
        block_size = 1
    if n_seg <= block_size:
        return rng.integers(0, n_seg, size=n_seg)
    n_blocks = int(np.ceil(n_seg / block_size))
    starts = rng.integers(0, n_seg - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n_seg]


def bootstrap_modal_fit(
    corners: np.ndarray,
    fs: float,
    nperseg: int,
    fmin: float,
    fmax: float,
    expected_freqs: dict,
    coh_hp: np.ndarray,
    coh_rw: np.ndarray,
    point_result: dict,
    *,
    n_boot: int = 300,
    rng: int | np.random.Generator | None = 0,
    block_size: int = 1,
    ci_lo_pct: float = 2.5,
    ci_hi_pct: float = 97.5,
) -> dict[str, Any]:
    """Run a segment block-bootstrap and return percentile CIs.

    Parameters
    ----------
    corners : (4, N) array
        Sign-corrected corner pushrod/displacement channels in
        ``[FL, FR, RL, RR]`` order.
    fs, nperseg, fmin, fmax, expected_freqs : same as
        :func:`vibrations_io.run_fit_from_arrays`. These must match the
        point-estimate call exactly.
    coh_hp, coh_rw : (nf_fit,) arrays
        Frozen coherence weights from the point estimate (already cropped
        to the fit window and floored at 0.2).
    point_result : dict
        Output of the point-estimate ``vibrations_lorentz.run_fit`` call;
        used both to seed each bootstrap fit and as a fallback if an
        individual bootstrap iteration fails.
    n_boot : int
        Number of bootstrap iterations. 200-300 is typical.
    rng : int, ``np.random.Generator`` or None
        Random seed / generator.
    block_size : int
        Block length for the block-bootstrap. Default 1 (i.i.d. segment
        resampling) is optimal when the inter-segment autocorrelation is
        weak (measured ~0.15-0.30 for typical race data). Raise to 2+ only
        when the Welch segment autocorrelation is known to be strong.

    Returns
    -------
    dict with keys ``fn_lo``, ``fn_hi``, ``fn_samples``,
    ``zeta_lo``, ``zeta_hi``, ``zeta_samples``,
    ``amp_front_lo``, ``amp_front_hi``, ``amp_front_samples``,
    ``amp_rear_lo``, ``amp_rear_hi``, ``amp_rear_samples``,
    plus ``n_boot_effective`` (draws that produced usable params).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # ---- One spectrogram pass; every bootstrap draw resamples this axis.
    freqs, sxx = compute_body_psds(corners, fs, nperseg=nperseg, return_segments=True)
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[fit_mask]
    sxx_fit = sxx[:, fit_mask, :]  # (4, nf_fit, n_seg)
    n_seg = sxx_fit.shape[-1]
    if n_seg < 2:
        log.warning(
            "bootstrap_modal_fit: only %d Welch segment(s) available; cannot construct CIs. Returning NaN.",
            n_seg,
        )
        nan4 = np.full(4, np.nan)
        return {
            "fn_lo": nan4,
            "fn_hi": nan4,
            "fn_samples": np.empty((0, 4)),
            "zeta_lo": nan4,
            "zeta_hi": nan4,
            "zeta_samples": np.empty((0, 4)),
            "amp_front_lo": nan4,
            "amp_front_hi": nan4,
            "amp_front_samples": np.empty((0, 4)),
            "amp_rear_lo": nan4,
            "amp_rear_hi": nan4,
            "amp_rear_samples": np.empty((0, 4)),
            "n_boot_effective": 0,
        }

    x0_seed = _params_to_seed(point_result["params"])
    fn_ref = np.asarray(point_result["params"], dtype=float)[:, 0]
    log_fn_ref = np.log(np.clip(fn_ref, 1e-6, None))
    self_idx = np.arange(len(fn_ref))

    fn_samples = np.full((n_boot, 4), np.nan)
    zeta_samples = np.full((n_boot, 4), np.nan)
    ampf_samples = np.full((n_boot, 4), np.nan)
    ampr_samples = np.full((n_boot, 4), np.nan)

    n_ok = 0
    n_swapped = 0
    for i in range(n_boot):
        idx = _block_indices(n_seg, block_size, rng)
        psd_avg = sxx_fit[:, :, idx].mean(axis=-1)  # (4, nf_fit)
        meas_norm = _smooth_and_normalise(psd_avg)
        # n_avg for the resampled draw: unique segments (approximate) - the
        # block-bootstrap gives correlated averages, so the effective count
        # is smaller than idx.size. A conservative estimate is
        # len(np.unique(idx)); the sigma path is not used downstream anyway.
        n_avg_i = max(int(np.unique(idx).size), 1)
        try:
            res_i = vibrations_lorentz.run_fit(
                freqs_fit,
                meas_norm,
                expected_freqs,
                coh_hp=coh_hp,
                coh_rw=coh_rw,
                n_avg=n_avg_i,
                x0_seed=x0_seed,
            )
        except Exception as exc:  # noqa: BLE001 - individual draws may fail
            log.debug("bootstrap iter %d failed: %s", i, exc)
            continue
        params_i = np.asarray(res_i["params"], dtype=float)  # (4, 4)
        # Mode-swap rejection: each draw's mode-j f0 must be nearest (in log
        # frequency) to its own reference, not to any other mode's reference.
        # A swap manifests as e.g. a "Heave" sample landing near Pitch's f0
        # because in that resample Heave was too weak to identify.
        fn_i = params_i[:, 0]
        if np.any(fn_i <= 0) or not np.all(np.isfinite(fn_i)):
            n_swapped += 1
            continue
        log_fn_i = np.log(fn_i)
        dist = np.abs(log_fn_i[:, None] - log_fn_ref[None, :])
        if not np.array_equal(np.argmin(dist, axis=1), self_idx):
            n_swapped += 1
            continue
        fn_samples[i] = fn_i
        zeta_samples[i] = params_i[:, 1]
        ampf_samples[i] = params_i[:, 2]
        ampr_samples[i] = params_i[:, 3]
        n_ok += 1

    if n_ok == 0:
        log.warning("bootstrap_modal_fit: 0 / %d draws succeeded", n_boot)
        nan4 = np.full(4, np.nan)
        return {
            "fn_lo": nan4,
            "fn_hi": nan4,
            "fn_samples": fn_samples,
            "zeta_lo": nan4,
            "zeta_hi": nan4,
            "zeta_samples": zeta_samples,
            "amp_front_lo": nan4,
            "amp_front_hi": nan4,
            "amp_front_samples": ampf_samples,
            "amp_rear_lo": nan4,
            "amp_rear_hi": nan4,
            "amp_rear_samples": ampr_samples,
            "n_boot_effective": 0,
        }

    def _pct(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.nanpercentile(arr, ci_lo_pct, axis=0)
        hi = np.nanpercentile(arr, ci_hi_pct, axis=0)
        return lo, hi

    fn_lo, fn_hi = _pct(fn_samples)
    z_lo, z_hi = _pct(zeta_samples)
    af_lo, af_hi = _pct(ampf_samples)
    ar_lo, ar_hi = _pct(ampr_samples)
    if n_swapped > 0:
        log.info(
            "  Bootstrap CIs: %d / %d draws OK (block=%d, K_seg=%d, mode-swaps rejected=%d)",
            n_ok,
            n_boot,
            block_size,
            n_seg,
            n_swapped,
        )
    else:
        log.info(
            "  Bootstrap CIs: %d / %d draws OK (block=%d, K_seg=%d)",
            n_ok,
            n_boot,
            block_size,
            n_seg,
        )
    return {
        "fn_lo": fn_lo,
        "fn_hi": fn_hi,
        "fn_samples": fn_samples,
        "zeta_lo": z_lo,
        "zeta_hi": z_hi,
        "zeta_samples": zeta_samples,
        "amp_front_lo": af_lo,
        "amp_front_hi": af_hi,
        "amp_front_samples": ampf_samples,
        "amp_rear_lo": ar_lo,
        "amp_rear_hi": ar_hi,
        "amp_rear_samples": ampr_samples,
        "n_boot_effective": n_ok,
    }
