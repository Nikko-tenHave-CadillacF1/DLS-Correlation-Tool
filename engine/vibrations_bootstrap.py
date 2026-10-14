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
* Every draw runs the same procedure as the point estimate (peak
  identification, peak-region selection, log-space least squares), so
  the CI includes the uncertainty in locating each peak.
* Draws whose modes land nearer another mode's point estimate (mode swaps)
  are rejected.
* The parametric sigmas returned by each draw's ``run_fit`` are unused;
  the bootstrap replaces them with percentile CIs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import vibrations_lorentz
from .logger import log
from .vibrations_io import _analysis_mask, compute_body_psds


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
    fmin: float | None,
    fmax: float | None,
    expected_freqs: dict,
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
    point_result : dict
        Output of the point-estimate ``vibrations_lorentz.run_fit`` call;
        its f0 values are the references for mode-swap rejection.
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
    fit_mask = _analysis_mask(freqs, fmin, fmax)
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

    fn_ref = np.asarray(point_result["params"], dtype=float)[:, 0]
    log_fn_ref = np.log(np.clip(fn_ref, 1e-6, None))
    self_idx = np.arange(len(fn_ref))
    # Modes can only swap within a subsystem: (Heave, Pitch) and (Roll, Warp).
    same_subsystem = (self_idx[:, None] // 2) == (self_idx[None, :] // 2)

    fn_samples = np.full((n_boot, 4), np.nan)
    zeta_samples = np.full((n_boot, 4), np.nan)
    ampf_samples = np.full((n_boot, 4), np.nan)
    ampr_samples = np.full((n_boot, 4), np.nan)

    n_ok = 0
    n_swapped = 0
    for i in range(n_boot):
        idx = _block_indices(n_seg, block_size, rng)
        psd_avg = sxx_fit[:, :, idx].mean(axis=-1)  # (4, nf_fit)
        norm = max(float(np.max(psd_avg)), 1e-300)
        n_avg_i = max(int(np.unique(idx).size), 1)
        try:
            res_i = vibrations_lorentz.run_fit(
                freqs_fit, psd_avg / norm, expected_freqs, n_avg=n_avg_i, quiet=True, welch_hann=True
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
        dist = np.where(same_subsystem, np.abs(log_fn_i[:, None] - log_fn_ref[None, :]), np.inf)
        if not np.array_equal(np.argmin(dist, axis=1), self_idx):
            n_swapped += 1
            continue
        fn_samples[i] = fn_i
        zeta_samples[i] = params_i[:, 1]
        ampf_samples[i] = res_i["amp_front"] * norm
        ampr_samples[i] = res_i["amp_rear"] * norm
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
