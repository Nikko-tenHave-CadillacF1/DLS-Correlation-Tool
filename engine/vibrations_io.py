"""Vibrations I/O, plotting and fit orchestration.

This module is the glue between raw measurement files (CSV or parquet)
and the two pure-math fit pipelines:

    engine.vibrations_lorentz   - per-mode shared-(f0, zeta) Lorentzian fit
    engine.vibrations_body4dof  - 13-parameter MCK fit

Responsibilities
----------------
1. **Loading** - CSV / parquet input, channel resolution with DLS
   underscore-alias support, sign-convention application.
2. **Pre-processing** - high-pass filter, body-frame transform via
   `T_BODY`, Welch PSD + coherence estimation, peak-normalisation,
   expected-band sanitisation.
3. **Dispatch** - calls `vibrations_lorentz.run_fit` directly, OR
   `vibrations_body4dof.run_fit` (which we can optionally seed with a
   prior Lorentz fit for a better DE basin).
4. **Plotting** - three figures per workflow:
       vibrations_diag_<run>.png         per-run measured vs fitted
       vibrations_mode_shapes_body_<run> per-run mode-shape bars
       vibrations_comparison_fit.png     multi-run overlay
5. **CLI** - `python -m engine.vibrations_io <file> [options]`.

The two fit modules import NOTHING from each other and NOTHING from
this file, which keeps each fit method independently auditable.
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel_config import RESAMPLE_RATE
from engine import vibrations_body4dof, vibrations_lorentz
from engine.datafunctions import (
    _apply_butterworth_filter_to_data,
    auto_nperseg,
    resample_to_uniform_rate,
    sanitize_numeric_series,
)
from engine.dataplotter import DataPlotter
from engine.logger import log
from engine.plot_runtime import _expand_folder_runs

# ============================================================================
# Constants
# ============================================================================

FORCE_CHANNELS = ["FPushrodFL", "FPushrodFR", "FPushrodRL", "FPushrodRR"]
DISPLACEMENT_CHANNELS = ["xDamperPotFL", "xDamperPotFR",
                         "xDamperPotRL", "xDamperPotRR"]

# Per-source pushrod sign convention. Raw FPushrod polarity differs by
# data source - the body-frame transform requires all four corners to
# have a consistent "compression positive" sign.
_PUSHROD_CORNER_SIGNS = {
    "DLS": np.array([-1, -1, +1, +1]),
    "CAR": np.array([-1, -1, +1, +1]),
    "DIL": np.array([-1, -1, -1, -1]),
}
_DEFAULT_CORNER_SIGNS = _PUSHROD_CORNER_SIGNS["CAR"]

# Corner-to-body transform: [FL, FR, RL, RR] -> [z_F, theta_F, z_R, theta_R].
T_BODY = np.array([
    [0.5,  0.5,  0,    0  ],
    [1,   -1,    0,    0  ],
    [0,    0,    0.5,  0.5],
    [0,    0,    1,   -1  ],
])

DOF_LABELS = ["Heave Front (z_F)", "Roll Front (th_F)",
              "Heave Rear (z_R)",  "Roll Rear (th_R)"]
_PLOT_DOF_ORDER = [0, 2, 1, 3]   # Heave_F, Heave_R, Roll_F, Roll_R
MODE_ORDER = ("Heave", "Pitch", "Roll", "Warp")

# Default expected modal bands (lo, hi) in Hz. Overridden per-run via
# the `expected_freqs` argument.
_DEFAULT_EXPECTED_FREQS = {
    "heave": (2.0,  6.0),
    "pitch": (6.0, 11.0),
    "roll":  (3.0,  6.0),
    "warp":  (5.0, 11.0),
}


# ============================================================================
# File I/O
# ============================================================================

def _detect_source_type(filepath: Path) -> str:
    upper = filepath.name.upper()
    if filepath.suffix.lower() == ".parquet" or "_DLS" in upper:
        return "DLS"
    if "GMDIL" in upper or "_DIL" in upper:
        return "DIL"
    if "MAC" in upper:
        return "CAR"
    return "CAR"


def _resolve_parquet_column(raw_cols: list, logical: str) -> str | None:
    """Case-insensitive, underscore-tolerant lookup for parquet columns."""
    raw_set = set(raw_cols)
    candidates = [
        logical, logical.lower(), logical.upper(),
        f"_{logical}", f"_{logical.lower()}",
        logical[0].upper() + logical[1:] if logical else logical,
    ]
    for c in candidates:
        if c in raw_set:
            return c
    target = logical.lower()
    for raw in raw_cols:
        if raw.lower() == target:
            return raw
        if raw.startswith("_") and raw[1:].lower() == target:
            return raw
    return None


def _parquet_schema_columns(filepath: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return [str(c).strip() for c in pq.read_schema(filepath).names]
    except ImportError:
        try:
            import fastparquet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Reading .parquet requires 'pyarrow' or 'fastparquet'. "
                "Install one via: pip install pyarrow"
            ) from exc
        return [str(c).strip()
                for c in fastparquet.ParquetFile(str(filepath)).columns]


def _read_parquet_channels(filepath: Path, channels: list,
                           fs: float) -> pd.DataFrame:
    raw_cols = _parquet_schema_columns(filepath)
    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for ch in channels:
        raw = _resolve_parquet_column(raw_cols, ch)
        if raw is None:
            missing.append(ch)
        else:
            rename_map[raw] = ch
    if missing:
        raise ValueError(
            f"Parquet '{filepath.name}' is missing required channels: {missing}"
        )
    time_raw = _resolve_parquet_column(raw_cols, "tLap")
    cols_to_load = list(rename_map)
    if time_raw is not None and time_raw not in cols_to_load:
        cols_to_load.append(time_raw)
    df = pd.read_parquet(filepath, columns=cols_to_load)
    if time_raw is not None:
        rename_map[time_raw] = "tLap"
    df = df.rename(columns=rename_map)
    if "tLap" in df.columns:
        df = resample_to_uniform_rate(df, target_rate=fs, time_col="tLap",
                                      run_name=filepath.stem)
    return df


def _load_raw_corners(filepath: Path, channels: list,
                      fs: float) -> np.ndarray:
    """Load + sanitise + high-pass filter four corner channels.

    Returns a (4, N) array in the order requested via `channels`. Sign
    conventions are NOT applied here - the caller is responsible.
    """
    if filepath.suffix.lower() == ".parquet":
        df = _read_parquet_channels(filepath, channels, fs)
    else:
        df = pd.read_csv(filepath, sep=",", skiprows=[0, 2],
                         header=0, low_memory=False)
    for ch in channels:
        df[ch] = sanitize_numeric_series(df[ch])
    df[channels] = df[channels].interpolate(method="linear", limit=100, axis=0)
    df = df.dropna(subset=channels)
    for ch in channels:
        filtered, ok = _apply_butterworth_filter_to_data(
            df[ch].values, cutoff=2, order=2, sample_rate=fs, btype="high"
        )
        if ok:
            df[ch] = filtered
        else:
            log.warning("High-pass filter failed for channel '%s'.", ch)
    return df[channels].astype(float).values.T


def _preflight_check(filepath: Path, channels: list) -> None:
    """Validate file existence + channel presence before loading."""
    if not filepath.exists():
        raise FileNotFoundError(f"Vibrations input not found: {filepath}")
    if filepath.suffix.lower() == ".parquet":
        raw_cols = _parquet_schema_columns(filepath)
        missing = [c for c in channels
                   if _resolve_parquet_column(raw_cols, c) is None]
    else:
        header = pd.read_csv(filepath, sep=",", nrows=0, skiprows=[0, 2],
                             header=0, low_memory=False).columns.tolist()
        present = set(header)
        missing = [c for c in channels if c not in present]
    if missing:
        raise ValueError(
            f"'{filepath.name}' is missing required channels: {missing}"
        )


# ============================================================================
# Body-frame transform + PSDs
# ============================================================================

def _to_body_frame(corner: np.ndarray) -> np.ndarray:
    """[FL, FR, RL, RR] -> [z_F, theta_F, z_R, theta_R]."""
    return T_BODY @ corner


def compute_body_psds(corner: np.ndarray, fs: float,
                      nperseg: int = 1024, *,
                      return_segments: bool = False
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSDs of the body-frame channels.

    When ``return_segments=True`` the per-segment spectrogram is returned
    instead of the segment-averaged PSD: shape ``(4, n_freqs, n_segments)``.
    Averaging that array over the last axis reproduces ``signal.welch``
    with the same ``nperseg`` and default 50 % overlap, so the bootstrap
    path can resample segments before averaging without changing the
    point-estimate result.
    """
    body = _to_body_frame(corner)
    if not return_segments:
        return signal.welch(body, fs, window="hann", nperseg=nperseg, axis=1)
    f, _, sxx = signal.spectrogram(
        body, fs=fs, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant",
        scaling="density", mode="psd", axis=1,
    )
    return f, sxx


def compute_coherences(corner: np.ndarray, fs: float, nperseg: int = 512,
                       smooth_bins: int = 5
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency coherence between z_F/z_R and theta_F/theta_R.

    Returns `(coh_hp, coh_rw)` as 1D arrays of magnitude (sqrt of squared
    coherence), each clipped to [0, 1] and optionally box-smoothed.
    """
    body = _to_body_frame(corner)
    _, coh_hp = signal.coherence(body[0], body[2], fs=fs, nperseg=nperseg)
    _, coh_rw = signal.coherence(body[1], body[3], fs=fs, nperseg=nperseg)
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins) / smooth_bins
        coh_hp = np.convolve(coh_hp, kernel, mode="same")
        coh_rw = np.convolve(coh_rw, kernel, mode="same")
    return (np.sqrt(np.clip(coh_hp, 0.0, 1.0)),
            np.sqrt(np.clip(coh_rw, 0.0, 1.0)))


# ============================================================================
# Expected-band sanitisation
# ============================================================================

def _normalise_expected_freqs(expected_freqs: dict | None) -> dict:
    """Promote user-supplied bands into `{mode: (lo, hi, mid)}` tuples.

    Accepts either `(lo, hi)` tuples, scalar centres (`+/- 15 %`), or
    falls back to `_DEFAULT_EXPECTED_FREQS`. The returned bands are used
    as *initial-guess hints* for peak-picking and seed construction --
    they are not hard priors on the optimiser, which is allowed to widen
    its search out to the full `[fmin, fmax]` of the fit window.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for mode in MODE_ORDER:
        key = mode.lower()
        v = (expected_freqs or {}).get(key)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
        elif v is not None:
            f = float(v)
            lo, hi = f * 0.85, f * 1.15
        else:
            lo, hi = _DEFAULT_EXPECTED_FREQS[key]
        out[key] = (lo, hi, 0.5 * (lo + hi))
    return out


# ============================================================================
# Fit dispatch
# ============================================================================

def _seed_modes_from_lorentz(result: dict) -> dict | None:
    """Build a `seed_modes` dict for body4dof from a Lorentz fit result."""
    fn, zeta, labels = result["fn"], result["zeta"], result["mode_labels"]
    seed: dict[str, tuple[float, float]] = {}
    for label, f0, z in zip(labels, fn, zeta):
        seed[str(label).lower()] = (float(f0), float(z))
    if not all(np.isfinite(list(seed[m.lower()])).all() for m in MODE_ORDER):
        return None
    return seed


def eval_fit_psds(result: dict, freqs_hz: np.ndarray,
                  include_baseline: bool = True) -> np.ndarray:
    """Dispatch to the appropriate fit module's `eval_psds`."""
    method = result["method"]
    if method == "lorentzian_combined":
        return vibrations_lorentz.eval_psds(result, freqs_hz,
                                            include_baseline=include_baseline)
    if method == "body4dof":
        return vibrations_body4dof.eval_psds(result, freqs_hz)
    raise ValueError(f"Unknown fit method: {method!r}")


# ============================================================================
# Folder-run shorthand
# ============================================================================

def expand_runs(runs: list, root_folder: Path) -> list:
    """Expand `{"folder": ..., "filetype": ...}` entries into per-file runs.

    Mirrors the engine's folder-run expansion so `Run_Vibrations.py`
    accepts the same shorthand as the other `Run_*.py` workflows.
    Entries without a `folder` key are returned unchanged.
    """
    return _expand_folder_runs(list(runs), Path(root_folder))


# ============================================================================
# Plotting - style helpers
# ============================================================================

_PLOT_FONT = DataPlotter.PLOT_FONT
_GRID_MAJOR = DataPlotter.GRID_STYLE["major"]
_GRID_MINOR = DataPlotter.GRID_STYLE["minor"]
_INK = "#1A1A1A"
_MEAS_COLOR = "#2000BF"
_FIT_COLOR = "#D70000"
_MODE_COLOR = "#00AA55"
_RESID_COLOR = "#D70000"
_BASELINE_COLOR = "#888888"
_POS_BAR = "#2E86AB"
_NEG_BAR = "#E05263"
_SAVE_KW = dict(pad_inches=0.15, facecolor="white", bbox_inches="tight")
_DEFAULT_COLORS = [
    "#FF8000", "#2000BF", "#D70000", "#008CFF",
    "#00CC88", "#CC0066", "#FFD700", "#4C00BF",
]


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")


def _plots_dir(output_dir: Path | None) -> Path:
    base = output_dir if output_dir else Path(".")
    out = base / "plots" / "vibrations"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _configure_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = (_PLOT_FONT["family"] if _PLOT_FONT["family"] in available
            else _PLOT_FONT["fallback"][0])
    plt.rcParams.update({
        "font.family": font,
        "font.sans-serif": [_PLOT_FONT["family"]] + _PLOT_FONT["fallback"],
        "axes.titlesize": _PLOT_FONT["title_size"],
        "axes.titleweight": "bold",
        "axes.labelsize": _PLOT_FONT["label_size"],
        "axes.labelweight": "bold",
        "axes.edgecolor": _INK,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "xtick.labelsize": _PLOT_FONT["tick_size"],
        "ytick.labelsize": _PLOT_FONT["tick_size"],
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "text.color": _INK,
        "legend.fontsize": _PLOT_FONT["legend_size"],
        "figure.titlesize": _PLOT_FONT["figure_title_size"],
        "figure.titleweight": "bold",
    })


def _style_axis(ax, grid_axis: str = "both") -> None:
    ax.grid(True, which="major", axis=grid_axis, **_GRID_MAJOR)
    ax.grid(True, which="minor", axis=grid_axis, **_GRID_MINOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _style_dof_row(ax, dof: int, ylabel_suffix: str = "(PSD)",
                   ylim_zero: bool = True) -> None:
    suffix = f"\n{ylabel_suffix}" if ylabel_suffix else ""
    ax.set_ylabel(f"{DOF_LABELS[dof]}{suffix}", fontsize=9.5,
                  fontweight="bold", rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.085, 0.5)
    if ylim_zero:
        ax.set_ylim(bottom=0)
    _style_axis(ax, grid_axis="y")


def _add_suptitle(fig, event: str, run_name: str, plot_type: str,
                  method: str | None = None,
                  extras: str | None = None) -> None:
    fig.suptitle(plot_type.upper(),
                 fontsize=_PLOT_FONT["figure_title_size"],
                 fontweight="bold", color=_INK, y=0.995)
    sub_parts = [s for s in (event, run_name,
                             f"method={method}" if method else None) if s]
    line = "  -  ".join(sub_parts)
    if extras:
        line = f"{line}    |    {extras}" if line else extras
    if line:
        fig.text(0.5, 0.965, line, ha="center", va="top",
                 fontsize=_PLOT_FONT["label_size"], color=_INK)


def _add_legend(ax, loc: str = "upper right") -> None:
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
        loc=loc, fancybox=True, framealpha=0.92, edgecolor="#3C3C3C",
        borderpad=0.55, handlelength=1.8,
        prop={"family": _PLOT_FONT["family"], "weight": "bold",
              "size": _PLOT_FONT["legend_size"]},
    )
    legend.get_frame().set_linewidth(1.4)


def _format_lorentz_line(label: str, f0: float, sigma_f0: float,
                         zeta: float, sigma_zeta: float) -> str:
    sf_ok = sigma_f0 is not None and np.isfinite(sigma_f0)
    sz_ok = sigma_zeta is not None and np.isfinite(sigma_zeta)
    f_str = (f"$f_0$={f0:.2f}\u00b1{sigma_f0:.2f}Hz"
             if sf_ok else f"$f_0$={f0:.2f}Hz")
    z_str = (f"$\\zeta$={zeta:.3f}\u00b1{sigma_zeta:.3f}"
             if sz_ok else f"$\\zeta$={zeta:.3f}")
    prefix = f"  {label}: " if label else "  "
    return f"{prefix}{f_str}  {z_str}"


def _add_lorentz_info_box(ax, fmin: float, fmax: float, items: list,
                          halign: str = "right", valign: str = "top",
                          fontsize: float = 9.0) -> None:
    """items: iterable of (label, f0, sigma_f0, zeta, sigma_zeta, color)."""
    header = TextArea(
        f"Lorentz fit [{fmin:.1f}\u2013{fmax:.1f} Hz]",
        textprops=dict(color=_INK, fontsize=fontsize, fontweight="bold",
                       family=_PLOT_FONT["family"]),
    )
    children = [header]
    for label, f0, sf, z, sz, color in items:
        children.append(TextArea(
            _format_lorentz_line(label, f0, sf, z, sz),
            textprops=dict(color=color, fontsize=fontsize, fontweight="bold",
                           family=_PLOT_FONT["family"]),
        ))
    vpacker = VPacker(children=children, pad=2, sep=2)
    xy = (0.985 if halign == "right" else 0.015,
          0.965 if valign == "top" else 0.035)
    ab = AnnotationBbox(
        vpacker, xy=xy, xycoords="axes fraction",
        box_alignment=(1.0 if halign == "right" else 0.0,
                       1.0 if valign == "top" else 0.0),
        bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.92, edgecolor="#3C3C3C", linewidth=1.4),
        frameon=True, pad=0,
    )
    ab.set_zorder(11)
    ax.add_artist(ab)


def _peak_normalise(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(arr))
    return arr / peak if peak > 0 else arr


# ============================================================================
# Plotting - figures
# ============================================================================

def generate_mode_shape_plot(result: dict, output_dir: Path | None = None,
                             event: str = "", run_name: str = "",
                             output_dpi: int = 300) -> None:
    """Body-coordinate mode-shape bar chart, one panel per mode."""
    _configure_style()
    plots_dir = _plots_dir(output_dir)
    fn, zeta, labels = result["fn"], result["zeta"], result["mode_labels"]
    shapes = result.get("mode_shapes")
    n_modes = len(fn)
    if n_modes == 0 or shapes is None:
        return
    safe = _safe_name(run_name) if run_name else "fit"
    fig, axes = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4.6),
                             sharey=True)
    if n_modes == 1:
        axes = [axes]
    dof_short = ["z_F", "th_F", "z_R", "th_R"]
    for i in range(n_modes):
        shape = np.real(shapes[:, i])
        shape = shape / np.max(np.abs(shape))
        ax = axes[i]
        ax.barh(range(4), shape,
                color=[_POS_BAR if s >= 0 else _NEG_BAR for s in shape],
                edgecolor=_INK, linewidth=0.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels(dof_short)
        ax.set_title(f"{labels[i]}\n{fn[i]:.2f} Hz  z={zeta[i]:.4f}",
                     fontsize=10)
        ax.axvline(0, color=_INK, linewidth=0.6)
        ax.set_xlim(-1.2, 1.2)
        _style_axis(ax, grid_axis="x")
    _add_suptitle(fig, event, run_name, "Mode Shapes - Body Coords",
                  method=result["method"])
    plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.86))
    fig.savefig(plots_dir / f"vibrations_mode_shapes_body_{safe}.png",
                dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  Mode-shape plot saved: vibrations_mode_shapes_body_%s.png",
             safe)


def generate_diagnosis_plot(result: dict, freqs_fit: np.ndarray,
                            psds_fit: np.ndarray, fmin: float, fmax: float,
                            run_name: str, output_dir: Path | None,
                            event: str = "", output_dpi: int = 300) -> None:
    """Per-run measured-vs-fitted PSDs with a residual subplot."""
    _configure_style()
    plots_dir = _plots_dir(output_dir)
    H_sq_fit = eval_fit_psds(result, freqs_fit)
    # Per-row least-squares scale so alpha_d * H_sq_fit[d] ~= psds_fit[d].
    # The scaled fit and the scaled sloped baseline (drawn below when
    # available) use the same alpha_d, so the reader can visually decompose
    # the fit into peaks + broadband floor.
    scale_vec = np.empty(H_sq_fit.shape[0], dtype=float)
    for d in range(H_sq_fit.shape[0]):
        denom = float(np.dot(H_sq_fit[d], H_sq_fit[d]))
        scale_vec[d] = (float(np.dot(psds_fit[d], H_sq_fit[d])) / denom
                        if denom > 0 else 1.0)
    H_sq_scaled = H_sq_fit * scale_vec[:, None]
    fn = result["fn"]
    zeta = result.get("zeta")
    labels = result["mode_labels"]
    method = result["method"]
    is_lorentz = (method == "lorentzian_combined")
    # V2 sloped baseline: draw the fitted per-trace power-law floor when
    # the result dict carries the extra keys. Body4dof results silently
    # skip this branch.
    baselines = result.get("baselines")
    slopes = result.get("baseline_slopes")
    f_ref = result.get("baseline_f_ref")
    show_baseline = (is_lorentz and slopes is not None
                     and baselines is not None and f_ref is not None)

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5]})

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        ax = axes[row]
        meas = psds_fit[dof]
        model = H_sq_scaled[dof]
        ax.plot(freqs_fit, meas, color=_MEAS_COLOR, linewidth=1.6, alpha=0.85,
                label="Measured" if row == 0 else None)
        ax.plot(freqs_fit, model, color=_FIT_COLOR, linewidth=1.6, alpha=0.85,
                label="Fitted" if row == 0 else None)
        ax.fill_between(freqs_fit, meas, model, color=_FIT_COLOR, alpha=0.12)
        if show_baseline:
            base_curve = (float(baselines[dof])
                          * (freqs_fit / float(f_ref)) ** float(slopes[dof])
                          * scale_vec[dof])
            ax.plot(freqs_fit, base_curve, color=_BASELINE_COLOR,
                    linestyle="--", linewidth=1.0, alpha=0.6,
                    label="Baseline" if row == 0 else None)
        if is_lorentz:
            mode_rows = [0, 1] if dof in (0, 2) else [2, 3]
            mode_names = (["Heave", "Pitch"] if dof in (0, 2)
                          else ["Roll", "Warp"])
            sig_fn = result.get("sigma_fn")
            sig_z = result.get("sigma_zeta")
            items = []
            for mr, mname in zip(mode_rows, mode_names):
                f0 = float(result["params"][mr, 0])
                z = float(result["params"][mr, 1])
                sf = (float(sig_fn[mr]) if sig_fn is not None
                      and mr < len(sig_fn) else float("nan"))
                sz = (float(sig_z[mr]) if sig_z is not None
                      and mr < len(sig_z) else float("nan"))
                if fmin <= f0 <= fmax:
                    ax.axvline(f0, color=_MODE_COLOR, linestyle="--",
                               linewidth=0.9, alpha=0.7)
                items.append((mname, f0, sf, z, sz, _FIT_COLOR))
            _add_lorentz_info_box(ax, fmin, fmax, items,
                                  halign="right", valign="top")
        else:
            for i, f_n in enumerate(fn):
                if not (fmin <= f_n <= fmax):
                    continue
                z_str = (f" z={zeta[i]:.3f}"
                         if zeta is not None and not np.isnan(zeta[i]) else "")
                ax.axvline(f_n, color=_MODE_COLOR, linestyle="--",
                           linewidth=0.9, alpha=0.7,
                           label=(f"{labels[i]} {f_n:.1f} Hz{z_str}"
                                  if row == 0 else None))
        _style_dof_row(ax, dof)
        if row == 0:
            _add_legend(ax, loc="upper left" if is_lorentz else "upper right")

    # Synchronise y-axis limits within each subsystem so front and rear share
    # a common scale. Heave (rows 0-1) and Roll (rows 2-3) are scaled
    # independently — their force/acceleration magnitudes differ.
    for pair in ((0, 1), (2, 3)):
        ymin = min(axes[r].get_ylim()[0] for r in pair)
        ymax = max(axes[r].get_ylim()[1] for r in pair)
        for r in pair:
            axes[r].set_ylim(ymin, ymax)

    # Residual subplot at the bottom.
    ax_res = axes[4]
    residual = sum((_peak_normalise(psds_fit[d])
                    - _peak_normalise(H_sq_fit[d])) ** 2 for d in range(4))
    ax_res.fill_between(freqs_fit, 0, residual, color=_RESID_COLOR, alpha=0.3)
    ax_res.plot(freqs_fit, residual, color=_RESID_COLOR, linewidth=1.0,
                alpha=0.7)
    total_sse = float(np.trapezoid(residual, freqs_fit))
    ax_res.text(0.985, 0.92, f"integral SSE = {total_sse:.3f}",
                transform=ax_res.transAxes, fontsize=8.5, fontweight="bold",
                family="monospace", color=_INK, va="top", ha="right")
    ax_res.set_ylabel("Residual\n(SSE)", fontsize=9.5, fontweight="bold",
                      rotation=0, ha="right", va="center")
    ax_res.yaxis.set_label_coords(-0.085, 0.5)
    ax_res.set_ylim(bottom=0)
    ax_res.yaxis.set_major_locator(plt.MaxNLocator(3))
    _style_axis(ax_res, grid_axis="y")

    axes[-1].set_xlabel("Frequency [Hz]")
    _add_suptitle(fig, event, run_name, "Modal Fit - Diagnosis",
                  method=method, extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.955))
    safe = _safe_name(run_name)
    fig.savefig(plots_dir / f"vibrations_diag_{safe}.png", dpi=output_dpi,
                **_SAVE_KW)
    plt.close(fig)
    log.info("  Diagnosis plot saved: vibrations_diag_%s.png", safe)


def plot_comparison(results: list, fmin: float = 1.0, fmax: float = 19.0,
                    event: str = "", output_dir: Path | None = None,
                    output_dpi: int = 300) -> None:
    """Overlay fitted PSDs from multiple runs on the four body-DOF axes."""
    _configure_style()
    if not results:
        log.warning("No results to compare.")
        return
    plots_dir = _plots_dir(output_dir)
    freqs_plot = np.linspace(fmin, fmax, 500)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        meas_fit = res.get("psds_fit")
        freqs_fit = res.get("freqs_fit")
        if meas_fit is None or freqs_fit is None:
            H_sq_plot = eval_fit_psds(res, freqs_plot)
        else:
            # Re-scale each DOF row so the fitted curve sits over the
            # measured data at its own fit window.
            H_sq_at_fit = eval_fit_psds(res, freqs_fit)
            scale = np.empty(meas_fit.shape[0])
            for d in range(meas_fit.shape[0]):
                denom = float(np.dot(H_sq_at_fit[d], H_sq_at_fit[d]))
                scale[d] = (float(np.dot(meas_fit[d], H_sq_at_fit[d])) / denom
                            if denom > 0 else 1.0)
            H_sq_plot = eval_fit_psds(res, freqs_plot) * scale[:, None]
        for row, dof in enumerate(_PLOT_DOF_ORDER):
            axes[row].plot(freqs_plot, H_sq_plot[dof], color=color,
                           linewidth=1.8, alpha=0.85, label=res["name"])

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        _style_dof_row(axes[row], dof, ylabel_suffix="")
    axes[-1].set_xlabel("Frequency [Hz]")

    # Figure-level mode-table info box.
    mode_groups: OrderedDict[str, list] = OrderedDict()
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        sig_fn = res.get("sigma_fn")
        sig_z = res.get("sigma_zeta")
        for i, f_n in enumerate(res["fn"]):
            if not (fmin <= f_n <= fmax):
                continue
            sf = (float(sig_fn[i]) if sig_fn is not None
                  and i < len(sig_fn) else float("nan"))
            sz = (float(sig_z[i]) if sig_z is not None
                  and i < len(sig_z) else float("nan"))
            mode_groups.setdefault(res["mode_labels"][i], []).append({
                "f": float(f_n), "sf": sf,
                "z": float(res["zeta"][i]), "sz": sz,
                "color": color, "name": res["name"],
            })

    if mode_groups:
        legend_fs = _PLOT_FONT["legend_size"]
        n_entries = sum(1 + len(e) for e in mode_groups.values())
        info_fs = (legend_fs if n_entries <= 12
                   else (legend_fs - 1 if n_entries <= 16 else legend_fs - 2))
        max_name_len = max(len(e["name"]) for entries in mode_groups.values()
                           for e in entries)
        info_lines: list[tuple[str, str]] = []
        for mode_name, entries in mode_groups.items():
            info_lines.append((mode_name, _INK))
            for entry in entries:
                line = _format_lorentz_line(
                    entry["name"].ljust(max_name_len),
                    entry["f"], entry["sf"], entry["z"], entry["sz"])
                info_lines.append((line, entry["color"]))
        text_areas = [TextArea(text,
                               textprops=dict(color=c, fontsize=info_fs,
                                              fontweight="bold",
                                              family="monospace"))
                      for text, c in info_lines]
        vpacker = VPacker(children=text_areas, pad=6, sep=2)
        # 0.945 (not 0.99) because bbox_inches="tight" sometimes clips
        # fig-level AnnotationBboxes flush against the figure edge.
        ab = AnnotationBbox(
            vpacker, xy=(0.97, 0.945), xycoords="figure fraction",
            box_alignment=(1.0, 1.0), frameon=True, pad=0,
            bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.92, edgecolor="#3C3C3C", linewidth=1.4),
        )
        ab.set_zorder(10)
        fig.add_artist(ab)

    methods = sorted({r.get("method", "?") for r in results})
    method_str = (methods[0] if len(methods) == 1
                  else "mixed (" + ",".join(methods) + ")")
    _add_suptitle(fig, event, f"{len(results)} runs",
                  "Modal Fit - Comparison",
                  method=method_str, extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.95))
    fig.savefig(plots_dir / "vibrations_comparison_fit.png",
                dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)

    log.info("  %-20s %-12s %-12s %-12s %-12s",
             "Run", "Heave [Hz]", "Pitch [Hz]", "Roll [Hz]", "Warp [Hz]")
    for res in results:
        fn, modes = res["fn"], res["mode_labels"]

        def _freq_for(label: str) -> str:
            for i, m in enumerate(modes):
                if m == label:
                    return f"{fn[i]:.2f}"
            return "-"
        log.info("  %-20s %-12s %-12s %-12s %-12s",
                 res["name"], _freq_for("Heave"), _freq_for("Pitch"),
                 _freq_for("Roll"), _freq_for("Warp"))
    log.info("  Plots saved to: %s", plots_dir)


# ============================================================================
# Logging helpers
# ============================================================================

def _log_modes(result: dict) -> None:
    sig_fn = result.get("sigma_fn")
    sig_z = result.get("sigma_zeta")
    boot = result.get("bootstrap_meta") or {}
    n_eff = int(boot.get("n_effective", 0))
    fn_ci = result.get("fn_ci") if n_eff > 0 else None
    zeta_ci = result.get("zeta_ci") if n_eff > 0 else None
    if n_eff > 0:
        note = f"bootstrap 95% CI, n={n_eff}"
    else:
        note = "parametric 1\u03c3"
    log.info("  %-8s %-24s %-26s  (\u00b1 = %s)",
             "Mode", "Freq [Hz]", "Damp", note)
    for i in range(len(result["fn"])):
        f = float(result["fn"][i])
        z = float(result["zeta"][i])
        f_lo = f_hi = z_lo = z_hi = np.nan
        if fn_ci is not None:
            try:
                f_lo = float(fn_ci[0][i])
                f_hi = float(fn_ci[1][i])
            except (IndexError, TypeError, ValueError):
                pass
        if zeta_ci is not None:
            try:
                z_lo = float(zeta_ci[0][i])
                z_hi = float(zeta_ci[1][i])
            except (IndexError, TypeError, ValueError):
                pass
        if np.isfinite(f) and np.isfinite(f_lo) and np.isfinite(f_hi):
            f_str = f"{f:6.3f} -{max(f - f_lo, 0.0):5.3f} +{max(f_hi - f, 0.0):5.3f}"
        else:
            sf = (float(sig_fn[i]) if sig_fn is not None and i < len(sig_fn)
                  else float("nan"))
            f_str = (f"{f:6.3f} \u00b1 {sf:5.3f}"
                     if np.isfinite(f) and np.isfinite(sf)
                     else (f"{f:6.3f}" if np.isfinite(f) else "N/A"))
        if np.isfinite(z) and np.isfinite(z_lo) and np.isfinite(z_hi):
            z_str = (f"{z:6.4f} -{max(z - z_lo, 0.0):6.4f} "
                     f"+{max(z_hi - z, 0.0):6.4f}")
        else:
            sz = (float(sig_z[i]) if sig_z is not None and i < len(sig_z)
                  else float("nan"))
            z_str = (f"{z:6.4f} \u00b1 {sz:6.4f}"
                     if np.isfinite(z) and np.isfinite(sz)
                     else (f"{z:6.4f}" if np.isfinite(z) else "N/A"))
        log.info("  %-8s %-24s %-26s",
                 result["mode_labels"][i], f_str, z_str)
    r2 = result.get("r_squared")
    if (isinstance(r2, (tuple, list)) and len(r2) == 2
            and np.isfinite(r2[0]) and np.isfinite(r2[1])):
        log.info("  Log-space R^2:  heave/pitch=%.3f  roll/warp=%.3f",
                 r2[0], r2[1])
    if result["method"] == "body4dof":
        log.info("  Body params:")
        for i, (name, unit) in enumerate(zip(vibrations_body4dof.PARAM_NAMES,
                                             vibrations_body4dof.PARAM_UNITS)):
            log.info("    %-6s %-12.1f %s", name, result["params"][i], unit)


def _warn_edge_fits(result: dict, fr: dict, fmin: float, fmax: float,
                    delta_f: float, edge_bins: float = 1.5) -> None:
    tol = edge_bins * delta_f
    for label, f0 in zip(result["mode_labels"], result["fn"]):
        if not np.isfinite(f0):
            continue
        lo, hi, _ = fr[label.lower()]
        if abs(f0 - fmin) <= tol:
            log.warning("  %s fit at %.2f Hz hit lower window edge "
                        "F_MIN=%.2f Hz - widen the window.",
                        label, f0, fmin)
        elif abs(f0 - fmax) <= tol:
            log.warning("  %s fit at %.2f Hz hit upper window edge "
                        "F_MAX=%.2f Hz - likely wheel-hop bleed; widen "
                        "EXPECTED_FREQS band or lower F_MAX.",
                        label, f0, fmax)
        elif f0 < lo - tol or f0 > hi + tol:
            log.info("  %s fit at %.2f Hz is outside expected band "
                     "(%.1f-%.1f Hz) but within fit window - expected band "
                     "may need updating.", label, f0, lo, hi)


# ============================================================================
# Top-level entry points
# ============================================================================

def run_fit_from_arrays(
    corners_raw: np.ndarray,
    fs: float = RESAMPLE_RATE,
    source_type: str | None = None,
    fmin: float = 1.0, fmax: float = 12.0,
    nperseg: int | str = 1024,
    total_mass: float | None = None,
    wheelbase: float | None = None,
    pitch_inertia: float | None = None,
    roll_inertia: float | None = None,
    show_plots: bool = True,
    output_dir: Path | None = None,
    label: str = "",
    displacement_mode: bool = False,
    expected_freqs: dict | None = None,
    method: str = "lorentzian_combined",
    event: str = "",
    output_dpi: int = 300,
    disp_corners_raw: np.ndarray | None = None,
    bootstrap_ci: bool = False,
    bootstrap_n: int = 400,
    bootstrap_seed: int = 0,
    min_averages_target: int = 200,
) -> dict:
    """Run the modal fit on an already-loaded (4, N) corner array.

    `corners_raw` must be in `[FL, FR, RL, RR]` order. Sign conventions
    are applied internally based on `source_type` (skipped when
    `displacement_mode=True`).
    """
    # ---- Apply per-source pushrod signs unless we're in displacement mode.
    if not displacement_mode:
        signs = _PUSHROD_CORNER_SIGNS.get(source_type or "",
                                          _DEFAULT_CORNER_SIGNS)
        corners = corners_raw * signs[:, None]
    else:
        corners = corners_raw

    n_samples = corners.shape[1]
    if nperseg == "auto":
        nperseg = auto_nperseg(n_samples, sample_rate=fs,
                               min_averages_target=min_averages_target)
        log.info("  Auto NPERSEG: %d (delta_f=%.3f Hz, ~%d averages)",
                 nperseg, fs / nperseg,
                 int(2 * n_samples / nperseg - 1))
    # Number of Welch segments (50% overlap is scipy's default for welch).
    # Used by the Lorentz fit to set absolute parameter sigmas from the
    # known log-PSD noise variance trigamma(K).
    n_avg = max(int(2 * n_samples / nperseg - 1), 1)
    log.info("  %s: %d samples (%.1f s), fit %.1f-%.1f Hz, "
             "method=%s, nperseg=%d",
             label or "fit", n_samples, n_samples / fs,
             fmin, fmax, method, nperseg)

    # ---- Welch -> body PSDs, crop to fit window, smooth + normalise.
    # Fit-only smoothing: medfilt (kernel=5) to kill single-bin spikes, then
    # a 3-bin moving-average pass for an additional light low-pass. Peak
    # location and Lorentzian width are preserved; only the bin-to-bin
    # ripple is reduced. `psds_fit` (raw) is still passed to body4dof.
    freqs, psds = compute_body_psds(corners, fs, nperseg=nperseg)
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[fit_mask]
    psds_fit = psds[:, fit_mask]
    _ma3 = np.ones(3) / 3.0
    psds_smooth = np.stack([
        np.convolve(signal.medfilt(p, kernel_size=5), _ma3, mode="same")
        for p in psds_fit
    ])
    global_peak = float(np.max(psds_smooth))
    if global_peak <= 0.0:
        global_peak = 1.0
    meas_norm = psds_smooth / global_peak

    # ---- Coherence weights (one 1-D array per subsystem).
    coh_hp, coh_rw = compute_coherences(corners, fs, nperseg=nperseg)
    coh_hp = np.maximum(coh_hp[fit_mask], 0.3)
    coh_rw = np.maximum(coh_rw[fit_mask], 0.3)

    fr = _normalise_expected_freqs(expected_freqs)
    log.info("  Expected freq bands: %s",
             ", ".join(f"{m}={fr[m][0]:.1f}-{fr[m][1]:.1f}Hz"
                       for m in ("heave", "pitch", "roll", "warp")))

    # ---- Optional displacement cross-check for body4dof roll/warp.
    disp_norm = None
    if disp_corners_raw is not None:
        _, disp_psds = compute_body_psds(disp_corners_raw, fs, nperseg=nperseg)
        disp_norm = np.stack([_peak_normalise(p)
                              for p in disp_psds[:, fit_mask]])

    # ---- Dispatch to the chosen fit method.
    if method == "lorentzian_combined":
        result = vibrations_lorentz.run_fit(
            freqs_fit, meas_norm, fr,
            coh_hp=coh_hp, coh_rw=coh_rw, n_avg=n_avg,
        )
    elif method == "body4dof":
        # Pre-fit Lorentz to seed the DE basin with realistic (f0, zeta)
        # estimates. This is optional - body4dof falls back to band
        # centres if we omit it - but it markedly improves convergence.
        seed_modes = None
        try:
            seed_result = vibrations_lorentz.run_fit(
                freqs_fit, meas_norm, fr, coh_hp=coh_hp, coh_rw=coh_rw,
                n_avg=n_avg,
            )
            seed_modes = _seed_modes_from_lorentz(seed_result)
        except Exception as exc:  # noqa: BLE001 - seed is optional
            log.warning("  Lorentz pre-seed failed (%s); "
                        "falling back to band centres", exc)
        result = vibrations_body4dof.run_fit(
            freqs_fit, meas_norm, psds_fit, fr,
            seed_modes=seed_modes,
            total_mass=total_mass, wheelbase=wheelbase,
            pitch_inertia=pitch_inertia, roll_inertia=roll_inertia,
            disp_psds_norm=disp_norm,
            coh_hp=coh_hp, coh_rw=coh_rw,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Use 'lorentzian_combined' or 'body4dof'."
        )

    # ---- Optional segment block-bootstrap CIs (Option B).
    #
    # Fusion policy (2026-07-07): when the bootstrap rejection rate exceeds
    # 70 % (frac_ok = n_effective/n_boot < 0.3), the surviving draws are so
    # few that their percentile CI is dominated by sampling noise rather
    # than genuine parameter uncertainty. In that regime the parametric
    # Fisher-information CI (built from trigamma(K) log-PSD noise variance)
    # is the honest lower bound and reporting it is more informative than
    # a wide, unstable bootstrap band. See tools/diagnose_zeta_ci.py EXP G.
    _FUSION_MIN_FRAC_OK = 0.30
    _BOOTSTRAP_BLOCK_SIZE = 1  # matches bootstrap_modal_fit default
    if bootstrap_ci and method == "lorentzian_combined":
        try:
            from . import vibrations_bootstrap
            ci = vibrations_bootstrap.bootstrap_modal_fit(
                corners, fs, nperseg, fmin, fmax, fr,
                coh_hp, coh_rw, point_result=result,
                n_boot=int(bootstrap_n), rng=int(bootstrap_seed),
                block_size=_BOOTSTRAP_BLOCK_SIZE,
            )
            n_eff = int(ci["n_boot_effective"])
            frac_ok = n_eff / max(int(bootstrap_n), 1)
            result["bootstrap_meta"] = {
                "n_boot": int(bootstrap_n),
                "n_effective": n_eff,
                "block_size": _BOOTSTRAP_BLOCK_SIZE,
                "seed": int(bootstrap_seed),
                "frac_ok": frac_ok,
                "fusion_policy": (
                    "bootstrap" if frac_ok >= _FUSION_MIN_FRAC_OK
                    else "parametric_fallback"
                ),
            }
            if frac_ok < _FUSION_MIN_FRAC_OK:
                # Retain the samples + raw percentile arrays under _boot
                # keys for diagnostics, but drop the primary CI keys so
                # _log_modes and downstream sigma consumers fall back to
                # the parametric 1-sigma path.
                result["fn_ci_boot"] = (ci["fn_lo"], ci["fn_hi"])
                result["zeta_ci_boot"] = (ci["zeta_lo"], ci["zeta_hi"])
                result["amp_front_ci_boot"] = (ci["amp_front_lo"], ci["amp_front_hi"])
                result["amp_rear_ci_boot"] = (ci["amp_rear_lo"], ci["amp_rear_hi"])
                log.warning(
                    "  Bootstrap fusion: %d/%d draws OK (%.0f%%) < %.0f%% "
                    "threshold; reporting parametric CRLB sigmas instead "
                    "of unstable bootstrap CI.",
                    n_eff, int(bootstrap_n), 100.0 * frac_ok,
                    100.0 * _FUSION_MIN_FRAC_OK,
                )
                # Force _log_modes to take the parametric branch.
                result["bootstrap_meta"]["n_effective"] = 0
            else:
                result["fn_ci"] = (ci["fn_lo"], ci["fn_hi"])
                result["zeta_ci"] = (ci["zeta_lo"], ci["zeta_hi"])
                result["amp_front_ci"] = (ci["amp_front_lo"], ci["amp_front_hi"])
                result["amp_rear_ci"] = (ci["amp_rear_lo"], ci["amp_rear_hi"])
                # Overwrite the parametric sigmas with the bootstrap CI
                # half-width so every downstream consumer (log table,
                # diagnosis-plot info box, injected modal_<mode>_*_sigma
                # channels) shows the honest empirical uncertainty. Legacy
                # parametric sigmas kept under `sigma_fn_parametric` etc.
                for key, ci_key in (("sigma_fn",         "fn_ci"),
                                    ("sigma_zeta",       "zeta_ci"),
                                    ("sigma_amp_front",  "amp_front_ci"),
                                    ("sigma_amp_rear",   "amp_rear_ci")):
                    lo_arr, hi_arr = result[ci_key]
                    lo = np.asarray(lo_arr, dtype=float)
                    hi = np.asarray(hi_arr, dtype=float)
                    hw = 0.5 * (hi - lo)
                    orig = np.asarray(result.get(key), dtype=float)
                    result[f"{key}_parametric"] = orig
                    merged = np.where(np.isfinite(hw), hw, orig)
                    result[key] = merged
        except Exception as exc:  # noqa: BLE001 - CI is optional
            log.warning("  Bootstrap CI failed (%s); falling back to "
                        "parametric sigmas only.", exc)

    _log_modes(result)

    result["psds_fit"] = psds_fit
    result["freqs_fit"] = freqs_fit
    result["source_type"] = source_type
    _warn_edge_fits(result, fr, fmin, fmax, fs / nperseg)

    if output_dir is not None:
        generate_diagnosis_plot(result, freqs_fit, psds_fit, fmin, fmax,
                                label, output_dir,
                                event=event, output_dpi=output_dpi)
        if show_plots:
            generate_mode_shape_plot(result, output_dir=output_dir,
                                     event=event, run_name=label,
                                     output_dpi=output_dpi)
    return result


def run_fit(filepath: Path, fs: float = RESAMPLE_RATE,
            fmin: float = 1.0, fmax: float = 12.0,
            nperseg: int | str = 1024,
            total_mass: float | None = None,
            wheelbase: float | None = None,
            pitch_inertia: float | None = None,
            roll_inertia: float | None = None,
            show_plots: bool = True,
            output_dir: Path | None = None,
            run_name: str | None = None,
            displacement_mode: bool = False,
            expected_freqs: dict | None = None,
            method: str = "lorentzian_combined",
            event: str = "",
            source_type: str | None = None,
            output_dpi: int = 300) -> dict:
    """File-based entry point: load `filepath`, then delegate to
    `run_fit_from_arrays`."""
    label = run_name or filepath.stem
    primary_channels = (DISPLACEMENT_CHANNELS if displacement_mode
                        else FORCE_CHANNELS)
    _preflight_check(filepath, primary_channels)
    if source_type is None:
        source_type = _detect_source_type(filepath)
    log.info("Loading: %s  [source=%s]", filepath.name, source_type)
    if displacement_mode:
        log.info("  Displacement mode: fitting damperpot displacement PSDs")
        corners = _load_raw_corners(filepath, DISPLACEMENT_CHANNELS, fs)
    else:
        corners = _load_raw_corners(filepath, FORCE_CHANNELS, fs)

    disp_corners = None
    if not displacement_mode and method == "body4dof":
        try:
            disp_corners = _load_raw_corners(filepath, DISPLACEMENT_CHANNELS,
                                             fs)
            log.info("  Displacement PSDs loaded for roll/warp cross-check")
        except Exception:
            log.info("  No displacement data available")

    return run_fit_from_arrays(
        corners, fs=fs, source_type=source_type,
        fmin=fmin, fmax=fmax, nperseg=nperseg,
        total_mass=total_mass, wheelbase=wheelbase,
        pitch_inertia=pitch_inertia, roll_inertia=roll_inertia,
        show_plots=show_plots, output_dir=output_dir,
        label=label, displacement_mode=displacement_mode,
        expected_freqs=expected_freqs, method=method, event=event,
        output_dpi=output_dpi, disp_corners_raw=disp_corners,
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="4-DOF body-mode fitting.")
    parser.add_argument("data_file",
                        help="Path to CSV/parquet data file with FPushrod "
                             "channels.")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=19.0)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--total-mass", type=float, default=None)
    parser.add_argument("--wheelbase", type=float, default=None)
    parser.add_argument("--pitch-inertia", type=float, default=None)
    parser.add_argument("--roll-inertia", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--displacement-mode", action="store_true")
    parser.add_argument("--expected-freqs", type=float, nargs=4, default=None,
                        metavar=("HEAVE", "PITCH", "ROLL", "WARP"))
    parser.add_argument("--method", choices=("lorentzian_combined", "body4dof"),
                        default="lorentzian_combined")
    parser.add_argument("--event", type=str, default="")
    parser.add_argument("--source-type", choices=("DLS", "CAR", "DIL"),
                        default=None,
                        help="Override source-type sign convention. "
                             "Auto-detected from filename if omitted.")
    parser.add_argument("--output-dpi", type=int, default=300)
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        sys.exit(1)

    expected_freqs = None
    if args.expected_freqs is not None:
        h, p, r, w = args.expected_freqs
        expected_freqs = {"heave": h, "pitch": p, "roll": r, "warp": w}

    run_fit(
        filepath=data_path, fs=args.fs,
        fmin=args.fmin, fmax=args.fmax, nperseg=args.nperseg,
        total_mass=args.total_mass, wheelbase=args.wheelbase,
        pitch_inertia=args.pitch_inertia, roll_inertia=args.roll_inertia,
        show_plots=not args.no_plots,
        displacement_mode=args.displacement_mode,
        expected_freqs=expected_freqs, method=args.method,
        event=args.event, source_type=args.source_type,
        output_dpi=args.output_dpi,
    )


if __name__ == "__main__":
    main()
