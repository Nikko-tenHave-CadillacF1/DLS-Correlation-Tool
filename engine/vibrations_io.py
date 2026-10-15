"""Vibrations I/O, plotting and fit orchestration.

This module is the glue between raw measurement files (CSV or parquet)
and the two pure-math fit pipelines:

    engine.vibrations_lorentz   - per-mode shared-(f0, zeta) Lorentzian fit
    engine.vibrations_body4dof  - two-axle model with absolute rates (needs damper travel)

Responsibilities
----------------
1. **Loading** - CSV / parquet input, channel resolution with DLS
   underscore-alias support, sign convention from `CHANNEL_TRANSFORMS`.
2. **Pre-processing** - high-pass filter, body-frame transform via
   `T_BODY`, Welch PSDs, force/travel FRFs, expected-band sanitisation.
3. **Dispatch** - calls `vibrations_lorentz.run_fit` directly, OR
   `vibrations_body4dof.run_fit` with the Lorentzian poles and the
   force/travel FRFs.
4. **Plotting** - measured vs fitted, one row per body DOF; a single run
   or several overlaid:
       vibrations_diag_<run>.png         body PSDs vs modal fit
       vibrations_axle_<run>.png         force/travel response vs fitted k, c (body4dof)
       vibrations_comparison_fit.png     all runs' modal fits (Run_Vibrations)
       vibrations_comparison_axle.png    all runs' axle fits (Run_Vibrations, body4dof)
5. **CLI** - `python -m engine.vibrations_io <file> [options]`.

The two fit modules import NOTHING from each other and NOTHING from
this file, which keeps each fit method independently auditable.
"""

from __future__ import annotations

import argparse
import sys
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

from channel_config import CHANNEL_TRANSFORMS, RESAMPLE_RATE
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
DISPLACEMENT_CHANNELS = ["xDamperPotFL", "xDamperPotFR", "xDamperPotRL", "xDamperPotRR"]

# Corner-to-body transform: [FL, FR, RL, RR] -> [z_F, theta_F, z_R, theta_R].
T_BODY = np.array(
    [
        [0.5, 0.5, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 1, -1],
    ]
)

DOF_LABELS = ["Heave Front (z_F)", "Roll Front (th_F)", "Heave Rear (z_R)", "Roll Rear (th_R)"]
_PLOT_DOF_ORDER = [0, 2, 1, 3]  # Heave_F, Heave_R, Roll_F, Roll_R
MODE_ORDER = ("Heave", "Pitch", "Roll", "Warp")

# Default expected modal bands (lo, hi) in Hz. Overridden per-run via
# the `expected_freqs` argument.
_DEFAULT_EXPECTED_FREQS = {
    "heave": (2.0, 6.0),
    "pitch": (6.0, 11.0),
    "roll": (3.0, 6.0),
    "warp": (5.0, 11.0),
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
        logical,
        logical.lower(),
        logical.upper(),
        f"_{logical}",
        f"_{logical.lower()}",
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
                "Reading .parquet requires 'pyarrow' or 'fastparquet'. Install one via: pip install pyarrow"
            ) from exc
        return [str(c).strip() for c in fastparquet.ParquetFile(str(filepath)).columns]


def _read_parquet_channels(filepath: Path, channels: list, fs: float) -> pd.DataFrame:
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
        raise ValueError(f"Parquet '{filepath.name}' is missing required channels: {missing}")
    time_raw = _resolve_parquet_column(raw_cols, "tLap")
    cols_to_load = list(rename_map)
    if time_raw is not None and time_raw not in cols_to_load:
        cols_to_load.append(time_raw)
    df = pd.read_parquet(filepath, columns=cols_to_load)
    if time_raw is not None:
        rename_map[time_raw] = "tLap"
    df = df.rename(columns=rename_map)
    if "tLap" in df.columns:
        df = resample_to_uniform_rate(df, target_rate=fs, time_col="tLap", run_name=filepath.stem)
    return df


def _load_raw_corners(filepath: Path, channels: list, fs: float) -> np.ndarray:
    """Load + sanitise + high-pass filter four corner channels.

    Returns a (4, N) array in the order requested via `channels`. Sign
    conventions are NOT applied here - the caller is responsible.
    """
    if filepath.suffix.lower() == ".parquet":
        df = _read_parquet_channels(filepath, channels, fs)
    else:
        df = pd.read_csv(filepath, sep=",", skiprows=[0, 2], header=0, low_memory=False)
    for ch in channels:
        df[ch] = sanitize_numeric_series(df[ch])
    df[channels] = df[channels].interpolate(method="linear", limit=100, axis=0)
    df = df.dropna(subset=channels)
    for ch in channels:
        filtered, ok = _apply_butterworth_filter_to_data(df[ch].values, cutoff=2, order=2, sample_rate=fs, btype="high")
        if ok:
            df[ch] = filtered
        else:
            log.warning("High-pass filter failed for channel '%s'.", ch)
    return df[channels].astype(float).values.T


def _apply_transforms(corners: np.ndarray, channels: list, source_type: str) -> np.ndarray:
    """Apply the engine's per-source channel transforms (pushrod compression-positive sign)."""
    transforms = CHANNEL_TRANSFORMS.get(source_type, {})
    return np.stack([transforms[ch](row) if ch in transforms else row for ch, row in zip(channels, corners)])


def _preflight_check(filepath: Path, channels: list) -> None:
    """Validate file existence + channel presence before loading."""
    if not filepath.exists():
        raise FileNotFoundError(f"Vibrations input not found: {filepath}")
    if filepath.suffix.lower() == ".parquet":
        raw_cols = _parquet_schema_columns(filepath)
        missing = [c for c in channels if _resolve_parquet_column(raw_cols, c) is None]
    else:
        header = pd.read_csv(filepath, sep=",", nrows=0, skiprows=[0, 2], header=0, low_memory=False).columns.tolist()
        present = set(header)
        missing = [c for c in channels if c not in present]
    if missing:
        raise ValueError(f"'{filepath.name}' is missing required channels: {missing}")


# ============================================================================
# Body-frame transform + PSDs
# ============================================================================


def _to_body_frame(corner: np.ndarray) -> np.ndarray:
    """[FL, FR, RL, RR] -> [z_F, theta_F, z_R, theta_R]."""
    return T_BODY @ corner


def compute_body_psds(
    corner: np.ndarray, fs: float, nperseg: int = 1024, *, return_segments: bool = False
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
        body,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        mode="psd",
        axis=1,
    )
    return f, sxx


def compute_axle_spectra(
    corners: np.ndarray, disp_corners: np.ndarray, fs: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Body-frame force/travel response and front-rear force cross-spectra.

    Returns ``(freqs, frf, coh2, force_csd)``: ``frf = S_xF / S_xx`` and its
    squared coherence per body DOF (4, nf), and the front-to-rear force
    cross-spectra ``[S(z_F, z_R), S(theta_F, theta_R)]`` (2, nf).
    """
    force, disp = _to_body_frame(corners), _to_body_frame(disp_corners)
    kw = dict(fs=fs, window="hann", nperseg=nperseg, axis=-1)
    freqs, s_xx = signal.welch(disp, **kw)
    _, s_ff = signal.welch(force, **kw)
    _, s_xf = signal.csd(disp, force, **kw)
    _, force_csd = signal.csd(force[[0, 1]], force[[2, 3]], **kw)
    coh2 = np.abs(s_xf) ** 2 / np.maximum(s_xx * s_ff, 1e-300)
    return freqs, s_xf / np.maximum(s_xx, 1e-300), coh2, force_csd


# ============================================================================
# Expected-band sanitisation
# ============================================================================


def _normalise_expected_freqs(expected_freqs: dict | None) -> dict:
    """Promote user-supplied bands into `{mode: (lo, hi, mid)}` tuples.

    Accepts either `(lo, hi)` tuples, scalar centres (`+/- 15 %`), or
    falls back to `_DEFAULT_EXPECTED_FREQS`. Each band is the range in
    which that mode's peak is searched.
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


def eval_fit_psds(result: dict, freqs_hz: np.ndarray, include_baseline: bool = True) -> np.ndarray:
    """Dispatch to the appropriate fit module's `eval_psds`."""
    method = result["method"]
    if method == "lorentzian_combined":
        return vibrations_lorentz.eval_psds(result, freqs_hz, include_baseline=include_baseline)
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
# Plotting
# ============================================================================

_PLOT_FONT = DataPlotter.PLOT_FONT
_GRID_MAJOR = DataPlotter.GRID_STYLE["major"]
_GRID_MINOR = DataPlotter.GRID_STYLE["minor"]
_INK = "#1A1A1A"
_MEAS_COLOR = "#2000BF"
_FIT_COLOR = "#D70000"
_SHADE_COLOR = "#00AA55"
_FLOOR_COLOR = "#888888"
_RUN_COLORS = ("#FF8000", "#2000BF", "#D70000", "#008CFF", "#00CC88", "#CC0066", "#FFD700", "#4C00BF")
_SAVE_KW = dict(pad_inches=0.15, facecolor="white", bbox_inches="tight")
_DOF_MODES = {0: (0, 1), 2: (0, 1), 1: (2, 3), 3: (2, 3)}  # body DOF -> its subsystem's mode rows


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")


def _plots_dir(output_dir: Path | None) -> Path:
    out = (output_dir or Path(".")) / "plots" / "vibrations"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _configure_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = _PLOT_FONT["family"] if _PLOT_FONT["family"] in available else _PLOT_FONT["fallback"][0]
    plt.rcParams.update(
        {
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
        }
    )


def _run_colors(results: list) -> list[tuple[str, str]]:
    """(measured, fitted) colour per run: blue/red for a single run, the run's own colour when overlaid."""
    if len(results) == 1:
        return [(_MEAS_COLOR, _FIT_COLOR)]
    return [(c, c) for c in (r.get("color") or _RUN_COLORS[i % len(_RUN_COLORS)] for i, r in enumerate(results))]


def _new_figure(results: list, ncols: int, title: str, event: str):
    """4 x ncols grid, one row per body DOF, with titles, row labels and grids in place."""
    _configure_style()
    fig, axes = plt.subplots(4, ncols, figsize=(11 + 3 * (ncols - 1), 11), sharex=True, squeeze=False)
    run = results[0]["name"] if len(results) == 1 else f"{len(results)} runs"
    methods = "/".join(sorted({r["method"] for r in results}))
    fig.suptitle(title.upper(), fontsize=_PLOT_FONT["figure_title_size"], fontweight="bold", y=0.995)
    subtitle = "  -  ".join(s for s in (event, run, f"method={methods}") if s)
    fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=_PLOT_FONT["label_size"])
    for row, dof in enumerate(_PLOT_DOF_ORDER):
        for ax in axes[row]:
            ax.grid(True, which="major", **_GRID_MAJOR)
            ax.grid(True, which="minor", **_GRID_MINOR)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[row, 0].set_ylabel(DOF_LABELS[dof], fontsize=9.5, rotation=0, ha="right", va="center")
        axes[row, 0].yaxis.set_label_coords(-0.085 if ncols == 1 else -0.15, 0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")
    return fig, axes


def _save(fig, output_dir: Path | None, stem: str, dpi: int) -> None:
    fig.tight_layout(pad=0.3, h_pad=0.4, rect=(0.045, 0, 1, 0.955))
    path = _plots_dir(output_dir) / f"{_safe_name(stem)}.png"
    fig.savefig(path, dpi=dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  Plot saved: %s", path.name)


def _add_legend(ax) -> None:
    if not ax.get_legend_handles_labels()[0]:
        return
    legend = ax.legend(
        loc="upper left",
        fancybox=True,
        framealpha=0.92,
        edgecolor="#3C3C3C",
        prop={"family": _PLOT_FONT["family"], "weight": "bold", "size": _PLOT_FONT["legend_size"]},
    )
    legend.get_frame().set_linewidth(1.4)


def _info_box(ax, header: str, lines: list[tuple[str, str]], fontsize: float = 9.0) -> None:
    """Framed text box in the top-right corner: a header and coloured ``(text, color)`` lines."""
    style = dict(fontsize=fontsize, fontweight="bold", family=_PLOT_FONT["family"])
    children = [TextArea(header, textprops=dict(color=_INK, **style))]
    children += [TextArea(text, textprops=dict(color=color, **style)) for text, color in lines]
    box = AnnotationBbox(
        VPacker(children=children, pad=2, sep=2),
        xy=(0.985, 0.965),
        xycoords="axes fraction",
        box_alignment=(1.0, 1.0),
        bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#3C3C3C", linewidth=1.4),
        frameon=True,
        pad=0,
        zorder=11,
    )
    ax.add_artist(box)


def _pm(value: float, sigma: float, fmt: str) -> str:
    """``value +/- sigma`` with the sigma dropped when it is not finite."""
    text = format(value, fmt)
    return f"{text}\u00b1{format(sigma, fmt)}" if np.isfinite(sigma) else text


def _view(results: list) -> tuple[float, float]:
    """Frequency range spanning the peak-search bands and fitted bins (+/- 1 Hz) of every run."""
    lo = min(float(r["freqs_fit"][0]) for r in results)
    hi = max(float(r["freqs_fit"][-1]) for r in results)
    edges = []
    for r in results:
        edges += [b for band in r.get("bands", {}).values() for b in band[:2]]
        mask = r.get("fit_mask")
        if mask is not None and mask.any():
            used = r["freqs_fit"][mask.any(axis=0)]
            edges += [float(used[0]), float(used[-1])]
    if not edges:
        return lo, hi
    return max(min(edges) - 1.0, lo), min(max(edges) + 1.0, hi)


def _model_scale(result: dict) -> np.ndarray:
    """Per-DOF factor taking the fitted model PSD to measured units."""
    if result.get("psd_scale") is not None:
        return np.full(4, float(result["psd_scale"]))
    # body4dof models only the shape: least-squares scale onto the measured PSD.
    model, meas = eval_fit_psds(result, result["freqs_fit"]), result["psds_fit"]
    return np.sum(meas * model, axis=1) / np.maximum(np.sum(model * model, axis=1), 1e-300)


def plot_modal_fits(
    results: list, stem: str, event: str = "", output_dir: Path | None = None, output_dpi: int = 300
) -> None:
    """Measured vs fitted body PSDs, one row per body DOF.

    A single run is the diagnosis view (fitted bins shaded, noise floor
    dashed); several runs are overlaid in their own colours. Dashed vertical
    lines mark each mode's f0; the info box sits on the front row of each
    subsystem since front and rear share f0 and zeta.
    """
    single = len(results) == 1
    colors = _run_colors(results)
    f_lo, f_hi = _view(results)
    f_dense = np.linspace(f_lo, f_hi, 400)
    fig, axes = _new_figure(results, 1, "Modal Fit", event)
    axes = axes[:, 0]
    for res, (c_meas, c_fit) in zip(results, colors):
        f = res["freqs_fit"]
        view = (f >= f_lo) & (f <= f_hi)
        scale = _model_scale(res)
        model = eval_fit_psds(res, f_dense) * scale[:, None]
        mask = res.get("fit_mask")
        for row, dof in enumerate(_PLOT_DOF_ORDER):
            ax, first = axes[row], row == 0
            if single and mask is not None:
                ax.fill_between(
                    f[view],
                    0,
                    1,
                    where=mask[dof][view],
                    step="mid",
                    transform=ax.get_xaxis_transform(),
                    color=_SHADE_COLOR,
                    alpha=0.10,
                    linewidth=0,
                    label="Fitted bins" if first else None,
                )
            ax.plot(
                f[view],
                res["psds_fit"][dof][view],
                color=c_meas,
                lw=1.2 if single else 0.8,
                marker="o",
                ms=2.5,
                alpha=1.0 if single else 0.45,
                label="Measured" if single and first else None,
            )
            ax.plot(f_dense, model[dof], color=c_fit, lw=1.8, label=("Fitted" if single else res["name"]) if first else None)
            if single and res.get("baselines") is not None:
                floor = float(res["baselines"][dof]) * scale[dof]
                ax.axhline(floor, color=_FLOOR_COLOR, linestyle="--", linewidth=1.0, label="Floor" if first else None)
            for m in _DOF_MODES[dof]:
                ax.axvline(float(res["fn"][m]), color=c_fit, linestyle="--", linewidth=0.9)

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        if dof not in (0, 1):
            continue
        lines = []
        for res, (_, c_fit) in zip(results, colors):
            peak_found = res.get("peak_found")
            for m in _DOF_MODES[dof]:
                name = res["mode_labels"][m] if single else f"{res['name']} {res['mode_labels'][m]}"
                if peak_found is not None and not peak_found[m]:
                    name += " (no clear peak)"
                f0 = _pm(float(res["fn"][m]), float(res["sigma_fn"][m]), ".2f")
                zeta = _pm(float(res["zeta"][m]), float(res["sigma_zeta"][m]), ".3f")
                lines.append((f"{name}: $f_0$={f0} Hz  $\\zeta$={zeta}", c_fit))
        _info_box(axes[row], "Modal fit", lines)

    # Front and rear share a y-scale within each subsystem.
    for pair in ((0, 1), (2, 3)):
        top = max(axes[r].get_ylim()[1] for r in pair)
        for r in pair:
            axes[r].set_ylim(0, top)
    axes[-1].set_xlim(f_lo, f_hi)
    _add_legend(axes[0])
    _save(fig, output_dir, stem, output_dpi)


def plot_axle_fits(
    results: list, stem: str, event: str = "", output_dir: Path | None = None, output_dpi: int = 300
) -> None:
    """Measured pushrod-force / damper-travel response vs the fitted axle rate k and damping c.

    One row per body DOF; ``Re(H)`` against k on the left, ``Im(H)/w`` against
    c on the right. Runs without an axle fit (not body4dof) are skipped.
    """
    results = [r for r in results if "frf" in r]
    if not results:
        return
    single = len(results) == 1
    colors = _run_colors(results)
    panels = (("k", "N/mm", "Stiffness  Re(H)", ".0f"), ("c", "N.s/mm", "Damping  Im(H)/$\\omega$", ".1f"))
    fig, axes = _new_figure(results, 2, "Axle Fit - Pushrod Force / Damper Travel", event)
    for col, (_, unit, title, _) in enumerate(panels):
        axes[0, col].set_title(f"{title}  [{unit}]")
    for res, (c_meas, c_fit) in zip(results, colors):
        f, frf = res["freqs_fit"], res["frf"]
        measured = {"k": frf.real, "c": frf.imag / (2.0 * np.pi * f)}
        for row, dof in enumerate(_PLOT_DOF_ORDER):
            first = row == 0
            for col, (key, _, _, _) in enumerate(panels):
                ax = axes[row, col]
                ax.plot(
                    f,
                    measured[key][dof],
                    color=c_meas,
                    lw=1.2 if single else 0.8,
                    marker="o",
                    ms=2.5,
                    alpha=1.0 if single else 0.45,
                    label="Measured" if single and first and col == 0 else None,
                )
                fit_label = ("Fitted" if single else res["name"]) if first and col == 0 else None
                ax.axhline(float(res[f"axle_{key}"][dof]), color=c_fit, lw=1.8, label=fit_label)

    for row, dof in enumerate(_PLOT_DOF_ORDER):
        for col, (key, unit, _, fmt) in enumerate(panels):
            lines = []
            for res, (_, c_fit) in zip(results, colors):
                value = _pm(float(res[f"axle_{key}"][dof]), float(res[f"sigma_axle_{key}"][dof]), fmt)
                lines.append((f"{'' if single else res['name'] + ': '}{key} = {value} {unit}", c_fit))
            _info_box(axes[row, col], "Axle fit", lines)
    _add_legend(axes[0, 0])
    _save(fig, output_dir, stem, output_dpi)


def plot_comparison(results: list, event: str = "", output_dir: Path | None = None, output_dpi: int = 300) -> None:
    """Overlay the modal fits (and, for body4dof runs, the axle fits) of several runs."""
    if not results:
        log.warning("No results to compare.")
        return
    plot_modal_fits(results, "vibrations_comparison_fit", event, output_dir, output_dpi)
    plot_axle_fits(results, "vibrations_comparison_axle", event, output_dir, output_dpi)
    log.info("  %-20s" + " %-11s" * 4, "Run", *(f"{m} [Hz]" for m in MODE_ORDER))
    for res in results:
        log.info("  %-20s" + " %-11.2f" * 4, res["name"], *res["fn"])



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
    log.info("  %-8s %-24s %-26s  (\u00b1 = %s)", "Mode", "Freq [Hz]", "Damp", note)
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
            sf = float(sig_fn[i]) if sig_fn is not None and i < len(sig_fn) else float("nan")
            f_str = (
                f"{f:6.3f} \u00b1 {sf:5.3f}"
                if np.isfinite(f) and np.isfinite(sf)
                else (f"{f:6.3f}" if np.isfinite(f) else "N/A")
            )
        if np.isfinite(z) and np.isfinite(z_lo) and np.isfinite(z_hi):
            z_str = f"{z:6.4f} -{max(z - z_lo, 0.0):6.4f} +{max(z_hi - z, 0.0):6.4f}"
        else:
            sz = float(sig_z[i]) if sig_z is not None and i < len(sig_z) else float("nan")
            z_str = (
                f"{z:6.4f} \u00b1 {sz:6.4f}"
                if np.isfinite(z) and np.isfinite(sz)
                else (f"{z:6.4f}" if np.isfinite(z) else "N/A")
            )
        log.info("  %-8s %-24s %-26s", result["mode_labels"][i], f_str, z_str)
    r2 = result.get("r_squared")
    r2_name = "Force/travel FRF R^2" if result["method"] == "body4dof" else "Log-space R^2"
    if isinstance(r2, (tuple, list)) and len(r2) == 2 and np.isfinite(r2[0]) and np.isfinite(r2[1]):
        log.info("  %s:  heave/pitch=%.3f  roll/warp=%.3f", r2_name, r2[0], r2[1])
    if result["method"] == "body4dof":
        f_meas, z_meas = result["poles_meas"]
        log.info("  %-8s %-18s %-22s %s", "Mode", "Lorentz f0 / zeta", "rear/front shape", "force phase model/meas")
        for i, label in enumerate(result["mode_labels"]):
            r = result["mode_shapes"][2 if i < 2 else 3, i]  # rear component; front = 1
            log.info(
                "  %-8s %6.3f / %-9.4f %5.2f @ %+5.0f deg      %+5.0f / %+5.0f deg",
                label,
                f_meas[i],
                z_meas[i],
                abs(r),
                np.degrees(np.angle(r)),
                result["phase_model"][i],
                result["phase_meas"][i],
            )
        log.info("  Axle parameters (per-corner equivalent, damper-travel coordinates):")
        for name, value in result["axle_params"].items():
            sigma = result["axle_param_sigmas"][name]
            log.info("    %-12s %10.3f \u00b1 %-9.3f %s", name, value, sigma, vibrations_body4dof.PARAM_UNITS[name[0]])


# ============================================================================
# Top-level entry points
# ============================================================================


def _analysis_mask(freqs: np.ndarray, fmin: float | None, fmax: float | None) -> np.ndarray:
    """Bins used for fitting: everything above DC unless a window is given."""
    mask = freqs > 0.0
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    return mask


def run_fit_from_arrays(
    corners: np.ndarray,
    fs: float = RESAMPLE_RATE,
    source_type: str | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    nperseg: int | str = 1024,
    show_plots: bool = True,
    output_dir: Path | None = None,
    label: str = "",
    displacement_mode: bool = False,
    expected_freqs: dict | None = None,
    method: str = "lorentzian_combined",
    event: str = "",
    output_dpi: int = 300,
    disp_corners: np.ndarray | None = None,
    bootstrap_ci: bool = False,
    bootstrap_n: int = 400,
    bootstrap_seed: int = 0,
    min_averages_target: int = 200,
) -> dict:
    """Run the modal fit on an already-loaded (4, N) corner array.

    `corners` must be in `[FL, FR, RL, RR]` order with the engine sign
    convention already applied (`CHANNEL_TRANSFORMS`, pushrods compression
    positive). ``body4dof`` also needs ``disp_corners``: damper travel of the
    same samples. ``expected_freqs`` bands are where each mode's peak is
    searched; ``fmin``/``fmax`` optionally crop the spectrum (body4dof
    defaults to 1-13 Hz, below wheel hop).
    """
    if method == "body4dof":
        if displacement_mode:
            raise ValueError("body4dof fits pushrod forces against damper travel; disable displacement_mode.")
        if disp_corners is None:
            raise ValueError("body4dof needs damper travel (xDamperPot*) alongside the pushrod forces.")
        fmin = 1.0 if fmin is None else fmin
        fmax = 13.0 if fmax is None else fmax
    n_samples = corners.shape[1]
    if nperseg == "auto":
        nperseg = auto_nperseg(n_samples, sample_rate=fs, min_averages_target=min_averages_target)
        log.info(
            "  Auto NPERSEG: %d (delta_f=%.3f Hz, ~%d averages)",
            nperseg,
            fs / nperseg,
            int(2 * n_samples / nperseg - 1),
        )
    # Number of Welch segments (50% overlap is scipy's default for welch).
    # Used by the Lorentz fit to set absolute parameter sigmas from the
    # known log-PSD noise variance trigamma(K).
    n_avg = max(int(2 * n_samples / nperseg - 1), 1)
    window = "" if fmin is None and fmax is None else f", window {fmin or 0:.1f}-{fmax or fs / 2:.1f} Hz"
    log.info(
        "  %s: %d samples (%.1f s), nperseg=%d (df=%.2f Hz)%s",
        label or "fit",
        n_samples,
        n_samples / fs,
        nperseg,
        fs / nperseg,
        window,
    )

    fr = _normalise_expected_freqs(expected_freqs)
    log.info(
        "  Peak search bands: %s",
        ", ".join(f"{m}={fr[m][0]:.1f}-{fr[m][1]:.1f}Hz" for m in ("heave", "pitch", "roll", "warp")),
    )

    # ---- Welch -> body PSDs (no smoothing: it flattens peaks), normalised to the peak inside the bands.
    freqs, psds = compute_body_psds(corners, fs, nperseg=nperseg)
    fit_mask = _analysis_mask(freqs, fmin, fmax)
    freqs_fit = freqs[fit_mask]
    psds_fit = psds[:, fit_mask]
    in_bands = np.zeros(freqs_fit.size, dtype=bool)
    for lo, hi, _ in fr.values():
        in_bands |= (freqs_fit >= lo) & (freqs_fit <= hi)
    global_peak = float(np.max(psds_fit[:, in_bands] if in_bands.any() else psds_fit))
    if global_peak <= 0.0:
        global_peak = 1.0
    meas_norm = psds_fit / global_peak

    # ---- Dispatch to the chosen fit method.
    if method == "lorentzian_combined":
        result = vibrations_lorentz.run_fit(freqs_fit, meas_norm, fr, n_avg=n_avg, welch_hann=True)
        result["psd_scale"] = global_peak
        for key in ("amp_front", "amp_rear", "sigma_amp_front", "sigma_amp_rear"):
            result[key] = result[key] * global_peak
    elif method == "body4dof":
        # The Lorentzian fit measures the poles; the force/travel FRFs measure the axle rates.
        poles = vibrations_lorentz.run_fit(freqs_fit, meas_norm, fr, n_avg=n_avg, welch_hann=True)
        _, frf, coh2, force_csd = compute_axle_spectra(corners, disp_corners, fs, nperseg)
        frf, coh2, force_csd = frf[:, fit_mask], coh2[:, fit_mask], force_csd[:, fit_mask]
        flipped = np.median(frf.real, axis=1) < 0.0
        if flipped.any():
            log.warning(
                "  Damper travel opposes pushrod force on %s; travel sign inverted.",
                ", ".join(DOF_LABELS[d] for d in np.flatnonzero(flipped)),
            )
            frf[flipped] *= -1.0
        result = vibrations_body4dof.run_fit(freqs_fit, frf, coh2, n_avg, poles, force_csd)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'lorentzian_combined' or 'body4dof'.")

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
                corners,
                fs,
                nperseg,
                fmin,
                fmax,
                fr,
                point_result=result,
                n_boot=int(bootstrap_n),
                rng=int(bootstrap_seed),
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
                "fusion_policy": ("bootstrap" if frac_ok >= _FUSION_MIN_FRAC_OK else "parametric_fallback"),
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
                    n_eff,
                    int(bootstrap_n),
                    100.0 * frac_ok,
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
                for key, ci_key in (
                    ("sigma_fn", "fn_ci"),
                    ("sigma_zeta", "zeta_ci"),
                    ("sigma_amp_front", "amp_front_ci"),
                    ("sigma_amp_rear", "amp_rear_ci"),
                ):
                    lo_arr, hi_arr = result[ci_key]
                    lo = np.asarray(lo_arr, dtype=float)
                    hi = np.asarray(hi_arr, dtype=float)
                    hw = 0.5 * (hi - lo)
                    orig = np.asarray(result.get(key), dtype=float)
                    result[f"{key}_parametric"] = orig
                    merged = np.where(np.isfinite(hw), hw, orig)
                    result[key] = merged
        except Exception as exc:  # noqa: BLE001 - CI is optional
            log.warning("  Bootstrap CI failed (%s); falling back to parametric sigmas only.", exc)

    _log_modes(result)

    result["psds_fit"] = psds_fit
    result["freqs_fit"] = freqs_fit
    result["bands"] = fr
    result["name"] = label or "fit"
    result["source_type"] = source_type

    if output_dir is not None and show_plots:
        try:
            plot_modal_fits([result], f"vibrations_diag_{result['name']}", event, output_dir, output_dpi)
            plot_axle_fits([result], f"vibrations_axle_{result['name']}", event, output_dir, output_dpi)
        except OSError as exc:
            # A locked/open PNG must not discard the fit itself.
            log.warning("  Could not save modal fit plot for %s: %s", label, exc)
    return result


def run_fit(
    filepath: Path,
    fs: float = RESAMPLE_RATE,
    fmin: float | None = None,
    fmax: float | None = None,
    nperseg: int | str = 1024,
    show_plots: bool = True,
    output_dir: Path | None = None,
    run_name: str | None = None,
    displacement_mode: bool = False,
    expected_freqs: dict | None = None,
    method: str = "lorentzian_combined",
    event: str = "",
    source_type: str | None = None,
    output_dpi: int = 300,
) -> dict:
    """File-based entry point: load `filepath`, then delegate to
    `run_fit_from_arrays`."""
    label = run_name or filepath.stem
    if displacement_mode:
        channels = DISPLACEMENT_CHANNELS
    elif method == "body4dof":
        channels = FORCE_CHANNELS + DISPLACEMENT_CHANNELS
    else:
        channels = FORCE_CHANNELS
    _preflight_check(filepath, channels)
    if source_type is None:
        source_type = _detect_source_type(filepath)
    log.info("Loading: %s  [source=%s]", filepath.name, source_type)
    if displacement_mode:
        log.info("  Displacement mode: fitting damperpot displacement PSDs")
    # Forces and travel load together so both cover the same samples.
    data = _apply_transforms(_load_raw_corners(filepath, channels, fs), channels, source_type)

    return run_fit_from_arrays(
        data[:4],
        fs=fs,
        source_type=source_type,
        fmin=fmin,
        fmax=fmax,
        nperseg=nperseg,
        show_plots=show_plots,
        output_dir=output_dir,
        label=label,
        displacement_mode=displacement_mode,
        expected_freqs=expected_freqs,
        method=method,
        event=event,
        output_dpi=output_dpi,
        disp_corners=data[4:] if len(data) == 8 else None,
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Body-mode fitting.")
    parser.add_argument("data_file", help="Path to CSV/parquet data file with FPushrod channels.")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=19.0)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--displacement-mode", action="store_true")
    parser.add_argument(
        "--expected-freqs", type=float, nargs=4, default=None, metavar=("HEAVE", "PITCH", "ROLL", "WARP")
    )
    parser.add_argument("--method", choices=("lorentzian_combined", "body4dof"), default="lorentzian_combined")
    parser.add_argument("--event", type=str, default="")
    parser.add_argument(
        "--source-type",
        choices=("DLS", "CAR", "DIL"),
        default=None,
        help="Override source-type sign convention. Auto-detected from filename if omitted.",
    )
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
        filepath=data_path,
        fs=args.fs,
        fmin=args.fmin,
        fmax=args.fmax,
        nperseg=args.nperseg,
        show_plots=not args.no_plots,
        displacement_mode=args.displacement_mode,
        expected_freqs=expected_freqs,
        method=args.method,
        event=args.event,
        source_type=args.source_type,
        output_dpi=args.output_dpi,
    )


if __name__ == "__main__":
    main()
