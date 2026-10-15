"""Synthetic ground-truth tests for the two-axle `engine.vibrations_body4dof` fit."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as signal

from engine.vibrations_body4dof import MODE_ORDER, eval_psds, modes, run_fit

FS = 100.0
N = int(600 * FS)
NPERSEG = 512

# Per subsystem: k [N/mm], c [N.s/mm], M [N.s^2/mm] (m_FR > 0 couples the axles).
TRUE = {
    "heave": (np.array([200.0, 250.0]), np.array([3.0, 4.0]), np.array([[0.15, 0.03], [0.03, 0.20]])),
    "roll": (np.array([150.0, 120.0]), np.array([2.0, 1.5]), np.array([[0.05, -0.01], [-0.01, 0.06]])),
}


def _simulate(k, c, M, rng):
    """Road-excited axle pair by FFT synthesis: returns (force, travel), each (2, N)."""
    f = np.fft.rfftfreq(N, 1.0 / FS)
    w = 2.0 * np.pi * f
    # Independent white-velocity road input at each axle.
    v = np.fft.rfft(rng.standard_normal((2, N)), axis=1)
    u = np.zeros_like(v)
    u[:, 1:] = v[:, 1:] / (1j * w[1:])
    D = np.diag(k)[None] + 1j * w[:, None, None] * np.diag(c)[None]
    Z = D - (w**2)[:, None, None] * M[None]
    Z[0] = np.eye(2)
    q = np.einsum("fij,jf->if", np.linalg.solve(Z, D), u)
    x = u - q  # suspension travel, compression positive
    F = np.einsum("fij,jf->if", D, x)
    force, travel = np.fft.irfft(F, n=N, axis=1), np.fft.irfft(x, n=N, axis=1)
    force += 0.05 * force.std(axis=1, keepdims=True) * rng.standard_normal(force.shape)
    return force, travel


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(0)
    force, travel = np.zeros((4, N)), np.zeros((4, N))
    poles = {key: np.zeros(4) for key in ("fn", "zeta", "sigma_fn", "sigma_zeta")}
    for s, (name, rows) in enumerate((("heave", [0, 2]), ("roll", [1, 3]))):
        k, c, M = TRUE[name]
        force[rows], travel[rows] = _simulate(k, c, M, rng)
        fn, zeta, _ = modes(M, np.diag(c), np.diag(k))
        poles["fn"][2 * s : 2 * s + 2], poles["zeta"][2 * s : 2 * s + 2] = fn, zeta
    poles["sigma_fn"][:] = 0.02
    poles["sigma_zeta"][:] = 0.01

    kw = dict(fs=FS, window="hann", nperseg=NPERSEG, axis=-1)
    freqs, s_xx = signal.welch(travel, **kw)
    _, s_ff = signal.welch(force, **kw)
    _, s_xf = signal.csd(travel, force, **kw)
    _, force_csd = signal.csd(force[[0, 1]], force[[2, 3]], **kw)
    band = (freqs >= 2.0) & (freqs <= 13.0)
    n_avg = int(2 * N / NPERSEG - 1)
    result = run_fit(
        freqs[band],
        (s_xf / s_xx)[:, band],
        (np.abs(s_xf) ** 2 / (s_xx * s_ff))[:, band],
        n_avg,
        poles,
        force_csd[:, band],
    )
    return result, freqs[band]


def test_recovers_axle_rates_and_masses(fit):
    result, _ = fit
    for name in ("heave", "roll"):
        k, c, M = TRUE[name]
        for j, side in enumerate("FR"):
            assert result["axle_params"][f"k_{name}_{side}"] == pytest.approx(k[j], rel=0.03)
            assert result["axle_params"][f"c_{name}_{side}"] == pytest.approx(c[j], rel=0.05)
        assert result["axle_params"][f"m_{name}_F"] == pytest.approx(1e3 * M[0, 0], rel=0.05)
        assert result["axle_params"][f"m_{name}_R"] == pytest.approx(1e3 * M[1, 1], rel=0.05)
        assert result["axle_params"][f"m_{name}_FR"] == pytest.approx(1e3 * M[0, 1], rel=0.15)


def test_labels_follow_front_rear_phase(fit):
    result, _ = fit
    assert result["mode_labels"] == list(MODE_ORDER)
    for i in range(4):
        rear = result["mode_shapes"][2 if i < 2 else 3, i]
        in_phase = i in (0, 2)
        assert (np.cos(np.angle(rear)) > 0) == in_phase
        assert (abs(result["phase_model"][i]) < 90) == (abs(result["phase_meas"][i]) < 90)


def test_modes_match_truth_with_sigmas(fit):
    result, _ = fit
    assert np.all(np.isfinite(result["sigma_fn"])) and np.all(np.isfinite(result["sigma_zeta"]))
    for s, name in enumerate(("heave", "roll")):
        k, c, M = TRUE[name]
        fn, zeta, _ = modes(M, np.diag(c), np.diag(k))
        np.testing.assert_allclose(np.sort(result["fn"][2 * s : 2 * s + 2]), fn, rtol=0.02)


def test_eval_psds_shape(fit):
    result, freqs = fit
    psds = eval_psds(result, freqs)
    assert psds.shape == (4, freqs.size)
    assert np.all(np.isfinite(psds)) and np.all(psds > 0)
