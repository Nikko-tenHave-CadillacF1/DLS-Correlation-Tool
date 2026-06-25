"""Full 4-DOF body-modal MCK fit.

Audit summary
-------------
We treat the sprung body as four lumped degrees of freedom:

    q = [ z_F      heave displacement of the front axle (m)
          theta_F  roll  rotation at the front axle (rad)
          z_R      heave displacement of the rear axle (m)
          theta_R  roll  rotation at the rear axle (rad) ]

driven by a force vector F. The equation of motion is

    M q'' + C q' + K q = F

with three (4x4) matrices `M`, `C`, `K` parameterised by 13 physical
quantities (front / rear mass + roll inertia, four damping coefficients,
four stiffness coefficients, and a single coupling mass `mu` that ties the
front and rear heave channels together through the unsprung mass).

Per Welch's theorem, for white-ish force inputs:

    |H_ij(omega)|^2 ~= S_q_i_from_F_j(omega) / S_F_j(omega)

so |H_ij|^2 is what the measured PSD ratios look like (up to amplitude).
We therefore fit the SHAPE of the measured PSDs, not their absolute
levels - both are independently normalised to peak 1 before residuals
are computed.

Decoupling
~~~~~~~~~~
The full 13-parameter MCK problem has many local minima. We split it
into two independent subproblems that share no parameters:

    Heave/Pitch (7 params): mF, mR, mu, cFH, cRH, kFH, kRH
    Roll/Warp   (6 params): IrF, cR, cW, kR, kW, IrR

Each subsystem becomes a 2-DOF MCK and is solved by `scipy.optimize.
differential_evolution` in log-parameter space. Initial seeds come from
the expected band centres, or - when the caller supplies one - from a
prior Lorentzian fit (`seed_modes` argument). After convergence, the
13 physical parameters are recombined into the full (4x4) M/C/K and the
mode shapes are extracted via state-space eigendecomposition.

Modes are then classified into Heave / Pitch / Roll / Warp based on
which body DOFs dominate the shape and on the phase relationship between
front and rear.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from engine.logger import log

# ----------------------------------------------------------------------------
# Public constants
# ----------------------------------------------------------------------------

PARAM_NAMES = ["mF", "mR", "IrF", "mu", "cFH", "cR", "cRH", "cW",
               "kFH", "kR", "kRH", "kW", "IrR"]
PARAM_UNITS = ["kg", "kg", "kg.m^2", "kg", "Ns/m", "Nms/rad", "Ns/m", "Nms/rad",
               "N/m", "Nm/rad", "N/m", "Nm/rad", "kg.m^2"]

# Lower/upper bounds on each physical parameter (linear scale).
# Optimisation happens in log-space inside these bounds.
BOUNDS_PHYSICAL = np.array([
    [200,    500],       # mF
    [300,    600],       # mR
    [10,      80],       # IrF
    [50,     500],       # mu
    [500,   5000],       # cFH
    [100,   2000],       # cR
    [500,   8000],       # cRH
    [200,   5000],       # cW
    [50000,  500000],    # kFH
    [5000,   80000],     # kR
    [80000,  800000],    # kRH
    [50000,  2000000],   # kW
    [10,      80],       # IrR
])

_HEAVE_PITCH_IDX = [0, 1, 3, 4, 6, 8, 10]   # mF, mR, mu, cFH, cRH, kFH, kRH
_ROLL_WARP_IDX   = [2, 5, 7, 9, 11, 12]      # IrF, cR, cW, kR, kW, IrR

MODE_ORDER = ("Heave", "Pitch", "Roll", "Warp")


# ----------------------------------------------------------------------------
# Model construction & evaluation
# ----------------------------------------------------------------------------

def build_MCK(params: np.ndarray):
    """Assemble the full (4x4) mass, damping, stiffness matrices.

    Sign conventions follow `q = [z_F, theta_F, z_R, theta_R]`. The
    coupling mass `mu` cross-links heave between front and rear; the
    warp stiffness/damping (`kW`, `cW`) cross-links theta_F and theta_R.
    """
    mF, mR, IrF, mu = params[0:4]
    cFH, cR, cRH, cW = params[4:8]
    kFH, kR, kRH, kW = params[8:12]
    IrR = params[12]
    M = np.array([
        [mF - mu, 0,   mu,      0  ],
        [0,       IrF, 0,        0  ],
        [mu,      0,   mR - mu, 0  ],
        [0,       0,   0,        IrR],
    ])
    C = np.array([
        [cFH, 0,        0,    0      ],
        [0,   cR + cW,  0,   -cW     ],
        [0,   0,        cRH,  0      ],
        [0,  -cW,       0,    cR + cW],
    ])
    K = np.array([
        [kFH, 0,        0,    0      ],
        [0,   kR + kW,  0,   -kW     ],
        [0,   0,        kRH,  0      ],
        [0,  -kW,       0,    kR + kW],
    ])
    return M, C, K


def _frf_squared(freqs_hz: np.ndarray, M: np.ndarray,
                 C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """|H_ij(omega)|^2 stacked over frequencies. Shape (n, n, nf)."""
    omega = 2.0 * np.pi * freqs_hz
    Z = (K[None] - (omega**2)[:, None, None] * M[None]
         + 1j * omega[:, None, None] * C[None])
    H = np.linalg.inv(Z)
    return np.transpose(np.abs(H)**2, (1, 2, 0))


def compute_H_mag_sq(freqs_hz: np.ndarray, M: np.ndarray,
                     C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Sum-over-driving-DOF |H_ij|^2: returns per-output PSD shape (n, nf)."""
    return _frf_squared(freqs_hz, M, C, K).sum(axis=1)


# ----------------------------------------------------------------------------
# Cost functions (one per subsystem)
# ----------------------------------------------------------------------------

def _normalise(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(arr))
    return arr / peak if peak > 0 else arr


def _shape_residual(meas_norm: np.ndarray, model: np.ndarray,
                    weights: np.ndarray | None = None) -> float:
    """Amplitude-weighted normalised-shape SSE.

    Both signals are normalised to peak 1 first, then the squared
    pointwise error is weighted by sqrt(meas) so that high-PSD bins
    (near the peaks) dominate the residual.
    """
    meas = _normalise(meas_norm)
    err = np.sqrt(meas) * (_normalise(model) - meas) ** 2
    if weights is not None:
        err = err * weights
    return float(np.sum(err))


def _cost_heave_pitch(log_params: np.ndarray, freqs_fit: np.ndarray,
                      meas_norm: np.ndarray,
                      total_mass: float | None, wheelbase: float | None,
                      pitch_inertia: float | None,
                      weights: np.ndarray | None) -> float:
    mF, mR, mu, cFH, cRH, kFH, kRH = np.exp(log_params)
    # 2-DOF heave subsystem
    M_hp = np.array([[mF - mu, mu], [mu, mR - mu]])
    if np.any(np.linalg.eigvalsh(M_hp) <= 0):
        return 1e15        # mass matrix not positive-definite
    C_hp = np.diag([cFH, cRH])
    K_hp = np.diag([kFH, kRH])
    H_sq = _frf_squared(freqs_fit, M_hp, C_hp, K_hp).sum(axis=1)
    cost = (_shape_residual(meas_norm[0], H_sq[0], weights)
            + _shape_residual(meas_norm[2], H_sq[1], weights))
    nf = len(freqs_fit)
    if total_mass is not None:
        cost += 500.0 * nf * ((mF + mR - total_mass) / total_mass) ** 2
    if pitch_inertia is not None and wheelbase is not None:
        cost += 500.0 * nf * ((mu * wheelbase - pitch_inertia) / pitch_inertia) ** 2
    return cost


def _cost_roll_warp(log_params: np.ndarray, freqs_fit: np.ndarray,
                    meas_norm: np.ndarray, meas_raw: np.ndarray,
                    roll_inertia: float | None,
                    disp_norm: np.ndarray | None,
                    weights: np.ndarray | None) -> float:
    IrF, cR, cW, kR, kW, IrR = np.exp(log_params)
    # 2-DOF roll subsystem (front + rear theta linked by warp k/c)
    M_rw = np.diag([IrF, IrR])
    C_rw = np.array([[cR + cW, -cW], [-cW, cR + cW]])
    K_rw = np.array([[kR + kW, -kW], [-kW, kR + kW]])
    H_ij_sq = _frf_squared(freqs_fit, M_rw, C_rw, K_rw)
    H_sq = H_ij_sq.sum(axis=1)
    cost = (_shape_residual(meas_norm[1], H_sq[0], weights)
            + _shape_residual(meas_norm[3], H_sq[1], weights))
    if disp_norm is not None:
        # Cross-check: predicted displacement PSD = |H_ij|^2 * measured force PSD.
        pred_thf = H_ij_sq[0, 0] * meas_raw[1] + H_ij_sq[0, 1] * meas_raw[3]
        pred_thr = H_ij_sq[1, 0] * meas_raw[1] + H_ij_sq[1, 1] * meas_raw[3]
        cost += _shape_residual(disp_norm[1], pred_thf, weights)
        cost += _shape_residual(disp_norm[3], pred_thr, weights)
    nf = len(freqs_fit)
    if roll_inertia is not None:
        cost += 500.0 * nf * ((IrF + IrR - roll_inertia) / roll_inertia) ** 2
    return cost


# ----------------------------------------------------------------------------
# Initial-seed construction
# ----------------------------------------------------------------------------

def _seed_from_modes(fH: float, fP: float, fR: float, fW: float,
                     zH: float, zP: float, zR: float, zW: float):
    """Map (f, zeta) per mode onto a feasible 13-parameter initial guess.

    Treats each mode as an SDOF with the textbook relation
    `omega_n = sqrt(k/m)` and `c = 2 zeta sqrt(k m)`.
    """
    mF, mR, IrF, IrR = 320.0, 450.0, 30.0, 30.0
    mu = BOUNDS_PHYSICAL[3, 0] * 2.0
    omH, omP, omR, omW = (2.0 * np.pi * f for f in (fH, fP, fR, fW))
    kFH = omH ** 2 * (mF - mu)
    kRH = omP ** 2 * (mR - mu)
    kR  = omR ** 2 * IrF
    kW  = (omW ** 2 * IrF - kR) / 2.0
    cFH = 2.0 * zH * np.sqrt(max(kFH * (mF - mu), 1.0))
    cRH = 2.0 * zP * np.sqrt(max(kRH * (mR - mu), 1.0))
    cR  = 2.0 * zR * np.sqrt(max(kR  * IrF, 1.0))
    cW  = 2.0 * zW * np.sqrt(max(kW  * IrF, 1.0))
    seed = np.array([mF, mR, IrF, mu, cFH, cR, cRH, cW,
                     kFH, kR, kRH, kW, IrR])
    seed = np.clip(seed,
                   BOUNDS_PHYSICAL[:, 0] * 1.001,
                   BOUNDS_PHYSICAL[:, 1] * 0.999)
    log_seed = np.log(seed)
    return log_seed[_HEAVE_PITCH_IDX], log_seed[_ROLL_WARP_IDX]


def _resolve_seed(expected_bands: dict,
                  seed_modes: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Pick the optimiser seed: explicit `seed_modes` wins; otherwise use
    the centres of the expected bands with a default 15 % damping ratio."""
    if seed_modes:
        try:
            fH, zH = seed_modes["heave"]
            fP, zP = seed_modes["pitch"]
            fR, zR = seed_modes["roll"]
            fW, zW = seed_modes["warp"]
            if all(np.isfinite([fH, fP, fR, fW, zH, zP, zR, zW])):
                log.info("  Body4DOF seeded from caller modes: "
                         "H=%.2fHz/%.3f  P=%.2fHz/%.3f  "
                         "R=%.2fHz/%.3f  W=%.2fHz/%.3f",
                         fH, zH, fP, zP, fR, zR, fW, zW)
                return _seed_from_modes(fH, fP, fR, fW, zH, zP, zR, zW)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("  Invalid seed_modes (%s); using band-centre defaults",
                        exc)
    fH, fP = expected_bands["heave"][2], expected_bands["pitch"][2]
    fR, fW = expected_bands["roll"][2],  expected_bands["warp"][2]
    return _seed_from_modes(fH, fP, fR, fW, 0.15, 0.15, 0.15, 0.15)


# ----------------------------------------------------------------------------
# Mode extraction
# ----------------------------------------------------------------------------

def extract_modes(M: np.ndarray, C: np.ndarray, K: np.ndarray):
    """State-space eigendecomposition -> (fn, zeta, mode_shapes)."""
    n = M.shape[0]
    M_inv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -M_inv @ K
    A[n:, n:] = -M_inv @ C
    eigvals, eigvecs = np.linalg.eig(A)
    # Keep only the +imag half (complex-conjugate pair).
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
    """Heuristic classifier from a complex mode-shape vector (length 4)."""
    heave_content = np.abs(shape[0]) + np.abs(shape[2])
    roll_content = np.abs(shape[1]) + np.abs(shape[3])
    if heave_content > roll_content:
        # Front vs rear heave in phase => Heave; out-of-phase => Pitch.
        phase_diff = np.angle(shape[0] * np.conj(shape[2]))
        return "Heave" if abs(phase_diff) < np.pi / 2 else "Pitch"
    # Front vs rear roll in phase => Roll; out-of-phase => Warp.
    phase_diff = np.angle(shape[1] * np.conj(shape[3]))
    return "Roll" if abs(phase_diff) < np.pi / 2 else "Warp"


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

_DE_KWARGS = dict(seed=42, popsize=40, polish=True, disp=False,
                  updating="deferred", workers=-1,
                  init="sobol", tol=1e-7, mutation=(0.5, 1.5))


def run_fit(freqs_fit: np.ndarray, body_psds_norm: np.ndarray,
            body_psds_raw: np.ndarray, expected_bands: dict,
            seed_modes: dict | None = None,
            total_mass: float | None = None,
            wheelbase: float | None = None,
            pitch_inertia: float | None = None,
            roll_inertia: float | None = None,
            disp_psds_norm: np.ndarray | None = None,
            coh_hp: np.ndarray | None = None,
            coh_rw: np.ndarray | None = None) -> dict:
    """13-parameter MCK fit, decoupled into heave/pitch and roll/warp.

    Parameters
    ----------
    freqs_fit : (nf,) array
        Frequency bins (Hz) inside the fit window.
    body_psds_norm : (4, nf) array
        Body-DOF PSDs normalised to a common peak, ordered
        [z_F, theta_F, z_R, theta_R].
    body_psds_raw : (4, nf) array
        Same shape but un-normalised (needed by the roll/warp
        displacement cross-check).
    expected_bands : dict
        Per-mode (lo, hi, mid) tuples keyed by lowercase mode name.
    seed_modes : dict or None
        Optional {"heave": (f0, zeta), "pitch": (f0, zeta),
                  "roll": (f0, zeta), "warp": (f0, zeta)} used to
        seed the differential-evolution basin. When omitted, the
        band-centre defaults are used.
    total_mass, wheelbase, pitch_inertia, roll_inertia : float, optional
        Physical constraints added as soft penalties to the cost.
    disp_psds_norm : (4, nf) array, optional
        Body-DOF displacement PSDs normalised to peak; enables the
        roll/warp displacement cross-check.
    coh_hp, coh_rw : (nf,) arrays, optional
        Coherence weights, applied as multiplicative bin weights.

    Returns
    -------
    dict with keys: method, params (13,), fn (4,), zeta (4,),
        sigma_fn (4, NaN), sigma_zeta (4, NaN),
        r_squared (NaN, NaN), mode_labels, mode_shapes (4x4).
    """
    bounds_log = np.log(BOUNDS_PHYSICAL)
    hp_x0, rw_x0 = _resolve_seed(expected_bands, seed_modes)

    log.info("  Optimising heave/pitch subsystem (7 params)...")
    hp_bounds = bounds_log[_HEAVE_PITCH_IDX]
    result_hp = differential_evolution(
        _cost_heave_pitch,
        bounds=list(zip(hp_bounds[:, 0], hp_bounds[:, 1])),
        args=(freqs_fit, body_psds_norm, total_mass, wheelbase,
              pitch_inertia, coh_hp),
        x0=hp_x0, **_DE_KWARGS,
    )
    log.info("  Heave/pitch done (cost=%.4f)", result_hp.fun)

    log.info("  Optimising roll/warp subsystem (6 params)...")
    rw_bounds = bounds_log[_ROLL_WARP_IDX]
    result_rw = differential_evolution(
        _cost_roll_warp,
        bounds=list(zip(rw_bounds[:, 0], rw_bounds[:, 1])),
        args=(freqs_fit, body_psds_norm, body_psds_raw, roll_inertia,
              disp_psds_norm, coh_rw),
        x0=rw_x0, **_DE_KWARGS,
    )
    log.info("  Roll/warp done (cost=%.4f)", result_rw.fun)

    # Recombine the 13 physical params and extract the modal basis.
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
    """Evaluate the fitted MCK transfer-function PSDs at `freqs_hz`.

    Returned shape: (4, len(freqs_hz)), rows = [z_F, theta_F, z_R, theta_R].
    """
    M, C, K = build_MCK(result["params"])
    return compute_H_mag_sq(freqs_hz, M, C, K)
