"""Tests for the standalone `engine.vibrations_lorentz` fit kernel.

These target pure-math helpers so they run fast and depend only on numpy
and scipy — no channel_config, no file I/O, no matplotlib.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.vibrations_lorentz import (
    MIN_BAND_GAP_HZ,
    MODE_ORDER,
    ZETA_MAX,
    ZETA_MIN,
    _lorentz_shape,
    run_fit,
)


class TestConstants:
    def test_mode_order(self):
        assert MODE_ORDER == ("Heave", "Pitch", "Roll", "Warp")

    def test_zeta_bounds(self):
        assert 0.0 < ZETA_MIN < ZETA_MAX < 1.0

    def test_min_band_gap_hz(self):
        assert MIN_BAND_GAP_HZ > 0


class TestLorentzShape:
    def test_returns_nbands_by_nf(self):
        freqs = np.linspace(1.0, 10.0, 128)
        params = np.array([[5.0, 0.05], [7.0, 0.08]])
        shape = _lorentz_shape(freqs, params)
        # Kernel returns (n_bands, nf) — one row per Lorentzian band.
        assert shape.shape == (2, freqs.size)

    def test_peak_at_natural_frequency(self):
        # A weakly-damped Lorentzian peaks at approximately f0.
        f0 = 6.0
        freqs = np.linspace(1.0, 12.0, 4096)
        shape = _lorentz_shape(freqs, np.array([[f0, 0.02]]))
        peak_idx = int(np.argmax(shape[0, :]))
        # Peak location within a couple of bins of f0.
        assert freqs[peak_idx] == pytest.approx(f0, abs=0.05)

    def test_monotonic_in_zeta_at_peak(self):
        # Sharper (lower zeta) resonance produces a taller peak.
        freqs = np.linspace(1.0, 12.0, 4096)
        low_z = _lorentz_shape(freqs, np.array([[6.0, 0.02]]))[0, :].max()
        high_z = _lorentz_shape(freqs, np.array([[6.0, 0.20]]))[0, :].max()
        assert low_z > high_z


def _synthetic_body_psds(
    freqs: np.ndarray,
    peaks: dict[str, float],
    zetas: dict[str, float] | None = None,
    noise_floor: float = 1e-4,
) -> np.ndarray:
    """Build a (4, nf) synthetic body-PSD stack with clean Lorentzian peaks.

    Rows: [z_F, theta_F, z_R, theta_R].
    """
    zetas = zetas or {}
    z_default = 0.06

    def one(f0):
        z = zetas.get(str(f0), z_default)
        shape = _lorentz_shape(freqs, np.array([[f0, z]]))[0, :]
        return shape / shape.max()

    z_F = one(peaks["heave"]) * 1.0 + one(peaks["pitch"]) * 0.5 + noise_floor
    z_R = one(peaks["heave"]) * 0.5 + one(peaks["pitch"]) * 1.0 + noise_floor
    t_F = one(peaks["roll"]) * 1.0 + one(peaks["warp"]) * 0.4 + noise_floor
    t_R = one(peaks["roll"]) * 0.5 + one(peaks["warp"]) * 1.0 + noise_floor
    stack = np.vstack([z_F, t_F, z_R, t_R])
    # Peak-normalise the whole stack (mirrors what the IO layer does).
    return stack / stack.max()


class TestRunFit:
    """End-to-end sanity: recover known peak locations from clean data."""

    def test_recovers_synthetic_peaks(self):
        freqs = np.linspace(1.0, 15.0, 2048)
        true_peaks = {"heave": 5.0, "pitch": 8.5, "roll": 6.0, "warp": 11.0}
        psds = _synthetic_body_psds(freqs, true_peaks)
        expected_bands = {
            "heave": (3.0, 6.5),
            "pitch": (7.0, 10.5),
            "roll":  (4.0, 7.5),
            "warp":  (9.0, 13.0),
        }
        result = run_fit(freqs, psds, expected_bands)
        assert result["fn"].shape == (4,)
        assert result["zeta"].shape == (4,)
        # Each mode should land within 0.5 Hz of the truth on clean data.
        for i, mode in enumerate(("Heave", "Pitch", "Roll", "Warp")):
            f_fit = result["fn"][i]
            f_true = true_peaks[mode.lower()]
            assert np.isfinite(f_fit), f"{mode} produced non-finite f0"
            assert abs(f_fit - f_true) < 0.5, (
                f"{mode}: fitted {f_fit:.3f} Hz vs true {f_true:.3f} Hz"
            )
        # Damping stays within advertised bounds.
        for z in result["zeta"]:
            assert ZETA_MIN <= z <= ZETA_MAX

    def test_result_dict_has_expected_keys(self):
        freqs = np.linspace(1.0, 15.0, 1024)
        psds = _synthetic_body_psds(
            freqs, {"heave": 5.0, "pitch": 8.5, "roll": 6.0, "warp": 11.0}
        )
        bands = {
            "heave": (3.0, 6.5), "pitch": (7.0, 10.5),
            "roll":  (4.0, 7.5), "warp":  (9.0, 13.0),
        }
        r = run_fit(freqs, psds, bands)
        for key in ("params", "baselines", "fn", "zeta",
                    "amp_front", "amp_rear",
                    "sigma_fn", "sigma_zeta",
                    "r_squared", "mode_labels"):
            assert key in r, f"missing key {key!r} in run_fit result"
        assert list(r["mode_labels"]) == list(MODE_ORDER)
