
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel_config import RESAMPLE_RATE
from engine.logger import log
from engine.datafunctions import (
    sanitize_numeric_series,
    _apply_butterworth_filter_to_data,
    resample_to_uniform_rate,
    auto_nperseg,
)
from engine.plot_runtime import _expand_folder_runs
from engine.dataplotter import DataPlotter
from . import vibrations_lorentz, vibrations_body4dof

FORCE_CHANNELS = ["FPushrodFL", "FPushrodFR", "FPushrodRL", "FPushrodRR"]
DISPLACEMENT_CHANNELS = ["xDamperPotFL", "xDamperPotFR", "xDamperPotRL", "xDamperPotRR"]

_PUSHROD_CORNER_SIGNS = {
    "DLS": np.array([-1, -1, +1, +1]),
    "CAR": np.array([-1, -1, +1, +1]),
    "DIL": np.array([-1, -1, -1, -1]),
}
_DEFAULT_CORNER_SIGNS = _PUSHROD_CORNER_SIGNS["CAR"]

DOF_LABELS = ["Heave Front (z_F)", "Roll Front (th_F)",
              "Heave Rear (z_R)", "Roll Rear (th_R)"]
_PLOT_DOF_ORDER = [0, 2, 1, 3]
_MODE_ORDER = ["Heave", "Pitch", "Roll", "Warp"]

_EXPECTED_FREQS = {
    "heave": (2.0,  6.0),
    "pitch": (6.0, 11.0),
    "roll":  (3.0,  6.0),
    "warp":  (5.0, 11.0),
}

def _detect_source_type(filepath: Path) -> str:
    name = filepath.name
    upper = name.upper()
    if filepath.suffix.lower() == ".parquet" or "_DLS" in upper:
        return "DLS"
    if "GMDIL" in upper or "_DIL" in upper:
        return "DIL"
    if "MAC" in upper:
        return "CAR"
    return "CAR"

def _resolve_parquet_column(raw_cols: list, logical: str) -> str | None:
    raw_set = set(raw_cols)
    candidates = [
        logical, logical.lower(), logical.upper(),
        f"_{logical}", f"_{logical.lower()}",
        logical[0].upper() + logical[1:] if logical else logical,
    ]
    for c in candidates:
        if c in raw_set:
            return c
    lower_target = logical.lower()
    for raw in raw_cols:
        if raw.lower() == lower_target:
            return raw
        if raw.startswith("_") and raw[1:].lower() == lower_target:
            return raw
    return None

def _read_parquet_channels(filepath: Path, channels: list, fs: float) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
        raw_cols = [str(c).strip() for c in pq.read_schema(filepath).names]
    except ImportError:
        try:
            import fastparquet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Reading .parquet requires 'pyarrow' or 'fastparquet'. "
                "Install one via: pip install pyarrow"
            ) from exc
        raw_cols = [str(c).strip() for c in fastparquet.ParquetFile(str(filepath)).columns]
    rename_map = {}
    missing = []
    for ch in channels:
        raw = _resolve_parquet_column(raw_cols, ch)
        if raw is None:
            missing.append(ch)
            continue
        rename_map[raw] = ch
    if missing:
        raise ValueError(
            f"Parquet '{filepath.name}' is missing required channels: {missing}"
        )
    time_raw = _resolve_parquet_column(raw_cols, "tLap")
    cols_to_load = list(rename_map.keys())
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

def _load_channels(filepath: Path, channels: list, fs: float) -> np.ndarray:
    if filepath.suffix.lower() == ".parquet":
        df = _read_parquet_channels(filepath, channels, fs)
    else:
        df = pd.read_csv(filepath, sep=",", skiprows=[0, 2], header=0, low_memory=False)
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

def load_force_data(filepath: Path, fs: float = RESAMPLE_RATE,
                    source_type: str | None = None) -> np.ndarray:
    if source_type is None:
        source_type = _detect_source_type(filepath)
    signs = _PUSHROD_CORNER_SIGNS.get(source_type, _DEFAULT_CORNER_SIGNS)
    F_corner = _load_channels(filepath, FORCE_CHANNELS, fs)
    return F_corner * signs[:, None]

def load_displacement_data(filepath: Path, fs: float = RESAMPLE_RATE) -> np.ndarray:
    return _load_channels(filepath, DISPLACEMENT_CHANNELS, fs)

def _preflight_check(filepath: Path, channels: list) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"Vibrations input not found: {filepath}")
    if filepath.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            raw_cols = [str(c).strip() for c in pq.read_schema(filepath).names]
        except ImportError:
            try:
                import fastparquet  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "Reading .parquet requires 'pyarrow' or 'fastparquet'."
                ) from exc
            raw_cols = [str(c).strip() for c in fastparquet.ParquetFile(str(filepath)).columns]
        missing = [ch for ch in channels if _resolve_parquet_column(raw_cols, ch) is None]
    else:
        header = pd.read_csv(filepath, sep=",", nrows=0, skiprows=[0, 2],
                             header=0, low_memory=False).columns.tolist()
        present = set(header)
        missing = [ch for ch in channels if ch not in present]
    if missing:
        raise ValueError(
            f"'{filepath.name}' is missing required channels: {missing}"
        )

T_BODY = np.array([
    [0.5,  0.5,  0,    0  ],
    [1,   -1,    0,    0  ],
    [0,    0,    0.5,  0.5],
    [0,    0,    1,   -1  ],
])

def compute_body_psds(F_corner: np.ndarray, T: np.ndarray,
                      fs: float, nperseg: int = 1024):
    F_body = T @ F_corner
    freqs, psds = signal.welch(F_body, fs, nperseg=nperseg, axis=1)
    return freqs, psds

def compute_coherence_weights(F_corner: np.ndarray, T: np.ndarray,
                              fs: float, nperseg: int = 512,
                              smooth_bins: int = 5) -> np.ndarray:
    F_body = T @ F_corner
    _, coh_hp = signal.coherence(F_body[0], F_body[2], fs=fs, nperseg=nperseg)
    _, coh_rw = signal.coherence(F_body[1], F_body[3], fs=fs, nperseg=nperseg)
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins) / smooth_bins
        coh_hp = np.convolve(coh_hp, kernel, mode="same")
        coh_rw = np.convolve(coh_rw, kernel, mode="same")
    amp_hp = np.sqrt(np.clip(coh_hp, 0.0, 1.0))
    amp_rw = np.sqrt(np.clip(coh_rw, 0.0, 1.0))
    return np.stack([amp_hp, amp_rw, amp_hp, amp_rw])

def expand_runs(runs: list, root_folder: Path) -> list:
    """Expand `{"folder": ..., "filetype": ...}` entries into per-file runs.

    Mirrors the engine's folder-run expansion so `Run_Vibrations.py` accepts
    the same `RUNS = [{"folder": ".", "filetype": ".txt", "type": "CAR"}]`
    shorthand as the other `Run_*.py` workflows. Entries without a `folder`
    key are returned unchanged.
    """
    return _expand_folder_runs(list(runs), Path(root_folder))

def _normalise(arr: np.ndarray) -> np.ndarray:
    peak = np.max(arr)
    return arr / peak if peak > 0 else arr

def _scale_model_to_data(measured: np.ndarray, model: np.ndarray) -> np.ndarray:
    scaled = np.empty_like(model)
    for i in range(model.shape[0]):
        denom = np.dot(model[i], model[i])
        alpha = np.dot(measured[i], model[i]) / denom if denom > 0 else 1.0
        scaled[i] = alpha * model[i]
    return scaled

def _normalise_expected_freqs(expected_freqs: dict | None) -> dict:
    out = {}
    for mode in _MODE_ORDER:
        key = mode.lower()
        v = (expected_freqs or {}).get(key)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
        elif v is not None:
            f = float(v)
            lo, hi = f * 0.85, f * 1.15
        else:
            lo, hi = _EXPECTED_FREQS[key]
        out[key] = (lo, hi, 0.5 * (lo + hi))
    return out

def eval_fit_psds(result: dict, freqs_hz: np.ndarray,
                  include_baseline: bool = True) -> np.ndarray:
    method = result["method"]
    if method == "lorentzian_combined":
        return vibrations_lorentz.eval_psds(result, freqs_hz,
                                            include_baseline=include_baseline)
    if method == "body4dof":
        return vibrations_body4dof.eval_psds(result, freqs_hz)
    raise ValueError(f"Unknown fit method: {method!r}")

_PLOT_FONT = DataPlotter.PLOT_FONT
_GRID_MAJOR = DataPlotter.GRID_STYLE["major"]
_GRID_MINOR = DataPlotter.GRID_STYLE["minor"]
_INK = "#1A1A1A"
_MEAS_COLOR = "#2000BF"
_FIT_COLOR  = "#D70000"
_MODE_COLOR = "#00AA55"
_RESID_COLOR = "#D70000"
_POS_BAR = "#2E86AB"
_NEG_BAR = "#E05263"
_SAVE_KW = dict(pad_inches=0.15, facecolor="white", bbox_inches="tight")

def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")

def _add_suptitle(fig, event: str, run_name: str, plot_type: str,
                  method: str = None, extras: str = None) -> None:
    fig.suptitle(plot_type.upper(),
                 fontsize=_PLOT_FONT["figure_title_size"],
                 fontweight="bold", color=_INK, y=0.995)
    sub_parts = [s for s in (event, run_name, f"method={method}" if method else None) if s]
    line = "  -  ".join(sub_parts)
    if extras:
        line = f"{line}    |    {extras}" if line else extras
    if line:
        fig.text(0.5, 0.965, line, ha="center", va="top",
                 fontsize=_PLOT_FONT["label_size"], color=_INK)

def _configure_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = _PLOT_FONT["family"] if _PLOT_FONT["family"] in available else _PLOT_FONT["fallback"][0]
    plt.rcParams.update({
        "font.family": font, "font.sans-serif": [_PLOT_FONT["family"]] + _PLOT_FONT["fallback"],
        "axes.titlesize": _PLOT_FONT["title_size"], "axes.titleweight": "bold",
        "axes.labelsize": _PLOT_FONT["label_size"], "axes.labelweight": "bold",
        "axes.edgecolor": _INK, "axes.labelcolor": _INK, "axes.titlecolor": _INK,
        "xtick.color": _INK, "ytick.color": _INK,
        "xtick.labelsize": _PLOT_FONT["tick_size"], "ytick.labelsize": _PLOT_FONT["tick_size"],
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "text.color": _INK, "legend.fontsize": _PLOT_FONT["legend_size"],
        "figure.titlesize": _PLOT_FONT["figure_title_size"], "figure.titleweight": "bold",
    })

def _style_axis(ax, grid_axis="both"):
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

def _vibrations_plots_dir(output_dir: Path | None) -> Path:
    base = output_dir if output_dir else Path(".")
    plots_dir = base / "plots" / "vibrations"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def _plot_mode_shape_bars(ax, shape: np.ndarray, tick_labels: list,
                          title: str) -> None:
    ax.barh(range(4), shape,
            color=[_POS_BAR if s >= 0 else _NEG_BAR for s in shape],
            edgecolor=_INK, linewidth=0.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(tick_labels)
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color=_INK, linewidth=0.6)
    ax.set_xlim(-1.2, 1.2)
    _style_axis(ax, grid_axis="x")

def _add_legend(ax, loc="upper right"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
        loc=loc, fancybox=True, framealpha=0.92, edgecolor="#3C3C3C",
        borderpad=0.55, handlelength=1.8,
        prop={"family": _PLOT_FONT["family"], "weight": "bold", "size": _PLOT_FONT["legend_size"]},
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
    """Engine-style compact Lorentz fit info box.

    items: iterable of (label, f0, sigma_f0, zeta, sigma_zeta, color).
    """
    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
    header = TextArea(
        f"Lorentz fit [{fmin:.1f}\u2013{fmax:.1f} Hz]",
        textprops=dict(color=_INK, fontsize=fontsize,
                       fontweight="bold", family=_PLOT_FONT["family"]),
    )
    children = [header]
    for label, f0, sf, z, sz, color in items:
        children.append(TextArea(
            _format_lorentz_line(label, f0, sf, z, sz),
            textprops=dict(color=color, fontsize=fontsize,
                           fontweight="bold", family=_PLOT_FONT["family"]),
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

def generate_plots(result: dict, output_dir: Path = None,
                   event: str = "", run_name: str = "",
                   output_dpi: int = 300):
    _configure_style()
    plots_dir = _vibrations_plots_dir(output_dir)
    method = result["method"]
    fn = result["fn"]
    zeta = result["zeta"]
    mode_labels = result["mode_labels"]
    mode_shapes = result.get("mode_shapes")
    n_modes = len(fn)
    safe = _safe_name(run_name) if run_name else "fit"
    if n_modes == 0 or mode_shapes is None:
        return
    fig, axes = plt.subplots(1, n_modes, figsize=(3 * n_modes, 4.6), sharey=True)
    if n_modes == 1:
        axes = [axes]
    dof_short = ["z_F", "th_F", "z_R", "th_R"]
    for i in range(n_modes):
        shape = np.real(mode_shapes[:, i])
        shape = shape / np.max(np.abs(shape))
        _plot_mode_shape_bars(
            axes[i], shape, dof_short,
            f"{mode_labels[i]}\n{fn[i]:.2f} Hz  z={zeta[i]:.4f}",
        )
    _add_suptitle(fig, event, run_name, "Mode Shapes - Body Coords",
                  method=method)
    plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.86))
    fig.savefig(plots_dir / f"vibrations_mode_shapes_body_{safe}.png",
                dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  Mode-shape plot saved: vibrations_mode_shapes_body_%s.png", safe)

def _generate_diagnosis_plot(result: dict, freqs_fit, psds_fit,
                             fmin, fmax, run_name, output_dir,
                             event: str = "", output_dpi: int = 300):
    _configure_style()
    plots_dir = _vibrations_plots_dir(output_dir)
    H_sq_fit = eval_fit_psds(result, freqs_fit)
    H_sq_scaled = _scale_model_to_data(psds_fit, H_sq_fit)
    fn, mode_labels, method = result["fn"], result["mode_labels"], result["method"]
    zeta = result.get("zeta")
    is_lorentz = (method == "lorentzian_combined")
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
                z_str = f" z={zeta[i]:.3f}" if zeta is not None and not np.isnan(zeta[i]) else ""
                ax.axvline(f_n, color=_MODE_COLOR, linestyle="--", linewidth=0.9, alpha=0.7,
                           label=f"{mode_labels[i]} {f_n:.1f} Hz{z_str}" if row == 0 else None)
        _style_dof_row(ax, dof)
        if row == 0:
            _add_legend(ax, loc="upper left" if is_lorentz else "upper right")
    ax_res = axes[4]
    residual = sum((_normalise(psds_fit[d]) - _normalise(H_sq_fit[d]))**2 for d in range(4))
    ax_res.fill_between(freqs_fit, 0, residual, color=_RESID_COLOR, alpha=0.3)
    ax_res.plot(freqs_fit, residual, color=_RESID_COLOR, linewidth=1.0, alpha=0.7)
    total_sse = float(np.trapezoid(residual, freqs_fit))
    ax_res.text(0.985, 0.92, f"integral SSE = {total_sse:.3f}", transform=ax_res.transAxes,
                fontsize=8.5, fontweight="bold", family="monospace", color=_INK,
                va="top", ha="right")
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
    safe_name = _safe_name(run_name)
    fig.savefig(plots_dir / f"vibrations_diag_{safe_name}.png", dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  Diagnosis plot saved: vibrations_diag_%s.png", safe_name)

def run_fit(filepath: Path, fs: float = RESAMPLE_RATE,
            fmin: float = 1.0, fmax: float = 12.0,
            nperseg: int | str = 1024, total_mass: float = None,
            wheelbase: float = None, pitch_inertia: float = None,
            roll_inertia: float = None, show_plots: bool = True,
            output_dir: Path = None, run_name: str = None,
            displacement_mode: bool = False,
            expected_freqs: dict = None,
            method: str = "lorentzian_combined",
            event: str = "",
            source_type: str | None = None,
            output_dpi: int = 300) -> dict:
    label = run_name or filepath.stem
    primary_channels = DISPLACEMENT_CHANNELS if displacement_mode else FORCE_CHANNELS
    _preflight_check(filepath, primary_channels)
    if source_type is None:
        source_type = _detect_source_type(filepath)
    log.info("Loading: %s  [source=%s]", filepath.name, source_type)
    if displacement_mode:
        log.info("  Displacement mode: fitting damperpot displacement PSDs")
        primary = load_displacement_data(filepath, fs)
    else:
        primary = load_force_data(filepath, fs, source_type=source_type)
    disp_primary = None
    if not displacement_mode and method == "body4dof":
        try:
            disp_primary = load_displacement_data(filepath, fs)
            log.info("  Displacement PSDs loaded for roll/warp cross-check")
        except Exception:
            log.info("  No displacement data available")
    return run_fit_from_arrays(
        primary, fs=fs, source_type=source_type,
        fmin=fmin, fmax=fmax, nperseg=nperseg,
        total_mass=total_mass, wheelbase=wheelbase,
        pitch_inertia=pitch_inertia, roll_inertia=roll_inertia,
        show_plots=show_plots, output_dir=output_dir,
        label=label, displacement_mode=displacement_mode,
        expected_freqs=expected_freqs, method=method,
        event=event, output_dpi=output_dpi,
        disp_primary=disp_primary,
        _signs_already_applied=True,
    )


def run_fit_from_arrays(
    primary: np.ndarray, fs: float = RESAMPLE_RATE,
    source_type: str | None = None,
    fmin: float = 1.0, fmax: float = 12.0,
    nperseg: int | str = 1024, total_mass: float = None,
    wheelbase: float = None, pitch_inertia: float = None,
    roll_inertia: float = None, show_plots: bool = True,
    output_dir: Path = None, label: str = "",
    displacement_mode: bool = False,
    expected_freqs: dict = None,
    method: str = "lorentzian_combined",
    event: str = "",
    output_dpi: int = 300,
    disp_primary: np.ndarray | None = None,
    _signs_already_applied: bool = False,
) -> dict:
    """Run the modal fit on an already-loaded (4, N) array.

    Used by DataPlotter._run_modal_fits to avoid re-reading source files.
    The array layout must be `[FL, FR, RL, RR]` (force) or the equivalent
    damper-pot ordering. When `displacement_mode` is False and the data did
    not pass through `load_force_data`, the per-source pushrod sign
    convention is applied here based on `source_type`.
    """
    if not displacement_mode and not _signs_already_applied:
        signs = _PUSHROD_CORNER_SIGNS.get(source_type or "", _DEFAULT_CORNER_SIGNS)
        primary = primary * signs[:, None]
    n_samples = primary.shape[1]
    if nperseg == "auto":
        nperseg = auto_nperseg(n_samples)
        log.info("  Auto NPERSEG: %d (Δf=%.3f Hz, ~%d averages)",
                 nperseg, fs / nperseg,
                 int(2 * n_samples / nperseg - 1))
    log.info("  %s: %d samples (%.1f s), fit %.1f-%.1f Hz, method=%s, nperseg=%d",
             label or "fit", n_samples, n_samples / fs,
             fmin, fmax, method, nperseg)
    T = T_BODY
    freqs, psds = compute_body_psds(primary, T, fs, nperseg=nperseg)
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit = freqs[fit_mask]
    psds_fit = psds[:, fit_mask]
    psds_fit_smooth = np.stack([signal.medfilt(p, kernel_size=3) for p in psds_fit])
    global_peak = float(np.max(psds_fit_smooth))
    if global_peak <= 0.0:
        global_peak = 1.0
    meas_norm = psds_fit_smooth / global_peak
    coh_fit = compute_coherence_weights(primary, T, fs, nperseg=nperseg)[:, fit_mask]
    fr = _normalise_expected_freqs(expected_freqs)
    log.info("  Expected freq bands: %s",
             ", ".join(f"{m}={fr[m][0]:.1f}-{fr[m][1]:.1f}Hz"
                       for m in ("heave", "pitch", "roll", "warp")))
    disp_norm = None
    if disp_primary is not None:
        _, disp_psds = compute_body_psds(disp_primary, T, fs, nperseg=nperseg)
        disp_norm = np.stack([_normalise(p) for p in disp_psds[:, fit_mask]])
    if method == "lorentzian_combined":
        result = vibrations_lorentz.run_fit(freqs_fit, meas_norm, fr,
                                            coh_fit=coh_fit)
        _log_modes(result)
    elif method == "body4dof":
        result = vibrations_body4dof.run_fit(
            freqs_fit, meas_norm, psds_fit, fr,
            total_mass=total_mass, wheelbase=wheelbase,
            pitch_inertia=pitch_inertia, roll_inertia=roll_inertia,
            disp_norm=disp_norm, coh_fit=coh_fit,
        )
        _log_modes(result)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'lorentzian_combined' "
            "or 'body4dof'."
        )
    result["psds_fit"] = psds_fit
    result["freqs_fit"] = freqs_fit
    result["source_type"] = source_type
    _warn_edge_fits(result, fr, fmin, fmax, fs / nperseg)
    if output_dir is not None:
        _generate_diagnosis_plot(result, freqs_fit, psds_fit, fmin, fmax,
                                 label, output_dir,
                                 event=event, output_dpi=output_dpi)
        if show_plots:
            generate_plots(result, output_dir=output_dir,
                           event=event, run_name=label, output_dpi=output_dpi)
    return result

def _warn_edge_fits(result: dict, fr: dict, fmin: float, fmax: float,
                    delta_f: float, edge_bins: float = 1.5) -> None:
    tol = edge_bins * delta_f
    for label, f0 in zip(result["mode_labels"], result["fn"]):
        if not np.isfinite(f0):
            continue
        lo, hi, _ = fr[label.lower()]
        if abs(f0 - fmin) <= tol:
            log.warning("  %s fit at %.2f Hz hit lower window edge "
                        "F_MIN=%.2f Hz — widen the window.", label, f0, fmin)
        elif abs(f0 - fmax) <= tol:
            log.warning("  %s fit at %.2f Hz hit upper window edge "
                        "F_MAX=%.2f Hz — likely wheel-hop bleed; widen "
                        "EXPECTED_FREQS band or lower F_MAX.", label, f0, fmax)
        elif f0 < lo - tol or f0 > hi + tol:
            log.info("  %s fit at %.2f Hz is outside expected band "
                     "(%.1f-%.1f Hz) but within fit window — expected band "
                     "may need updating.", label, f0, lo, hi)

def _log_modes(result: dict) -> None:
    sig_fn = result.get("sigma_fn")
    sig_z = result.get("sigma_zeta")
    log.info("  %-8s %-18s %-20s", "Mode", "Freq [Hz]", "Damp")
    for i in range(len(result["fn"])):
        f = float(result["fn"][i])
        z = float(result["zeta"][i])
        sf = float(sig_fn[i]) if sig_fn is not None and i < len(sig_fn) else float("nan")
        sz = float(sig_z[i]) if sig_z is not None and i < len(sig_z) else float("nan")
        f_str = (f"{f:6.3f} \u00b1 {sf:5.3f}"
                 if np.isfinite(f) and np.isfinite(sf) else
                 (f"{f:6.3f}" if np.isfinite(f) else "N/A"))
        z_str = (f"{z:6.4f} \u00b1 {sz:6.4f}"
                 if np.isfinite(z) and np.isfinite(sz) else
                 (f"{z:6.4f}" if np.isfinite(z) else "N/A"))
        log.info("  %-8s %-18s %-20s", result["mode_labels"][i], f_str, z_str)
    r2 = result.get("r_squared")
    if isinstance(r2, (tuple, list)) and len(r2) == 2 \
            and np.isfinite(r2[0]) and np.isfinite(r2[1]):
        log.info("  Log-space R^2:  heave/pitch=%.3f  roll/warp=%.3f",
                 r2[0], r2[1])
    if result["method"] == "body4dof":
        from .vibrations_body4dof import PARAM_NAMES, PARAM_UNITS
        log.info("  Body params:")
        for i, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
            log.info("    %-6s %-12.1f %s", name, result["params"][i], unit)

_DEFAULT_COLORS = [
    "#FF8000", "#2000BF", "#D70000", "#008CFF",
    "#00CC88", "#CC0066", "#FFD700", "#4C00BF",
]

def plot_comparison(results: list, fmin: float = 1.0, fmax: float = 19.0,
                    event: str = "", output_dir: Path = None,
                    output_dpi: int = 300):
    _configure_style()
    if not results:
        log.warning("No results to compare.")
        return
    plots_dir = _vibrations_plots_dir(output_dir)
    freqs_plot = np.linspace(fmin, fmax, 500)
    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
    from collections import OrderedDict
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        meas_fit = res.get("psds_fit")
        freqs_fit = res.get("freqs_fit")
        if meas_fit is None or freqs_fit is None:
            H_sq_plot = eval_fit_psds(res, freqs_plot)
        else:
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
    mode_groups = OrderedDict()
    for idx, res in enumerate(results):
        color = res.get("color") or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
        sig_fn = res.get("sigma_fn")
        sig_z = res.get("sigma_zeta")
        for i, f_n in enumerate(res["fn"]):
            if fmin <= f_n <= fmax:
                sf = (float(sig_fn[i]) if sig_fn is not None
                      and i < len(sig_fn) else float("nan"))
                sz = (float(sig_z[i]) if sig_z is not None
                      and i < len(sig_z) else float("nan"))
                mode_groups.setdefault(res["mode_labels"][i], []).append(
                    {"f": float(f_n), "sf": sf,
                     "z": float(res["zeta"][i]), "sz": sz,
                     "color": color, "name": res["name"]})
    if mode_groups:
        legend_fs = _PLOT_FONT["legend_size"]
        n_entries = sum(1 + len(e) for e in mode_groups.values())
        info_fs = legend_fs if n_entries <= 12 else (legend_fs - 1 if n_entries <= 16 else legend_fs - 2)
        max_name_len = max(len(e["name"]) for entries in mode_groups.values() for e in entries)
        info_lines = []
        for mode_name, entries in mode_groups.items():
            info_lines.append((mode_name, _INK))
            for entry in entries:
                line = _format_lorentz_line(
                    entry["name"].ljust(max_name_len),
                    entry["f"], entry["sf"], entry["z"], entry["sz"])
                info_lines.append((line, entry["color"]))
        text_areas = [TextArea(text, textprops=dict(color=c, fontsize=info_fs,
                      fontweight="bold", family="monospace")) for text, c in info_lines]
        vpacker = VPacker(children=text_areas, pad=6, sep=2)
        ab = AnnotationBbox(vpacker, xy=(0.97, 0.945), xycoords="figure fraction",
                            box_alignment=(1.0, 1.0), frameon=True, pad=0,
                            bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white",
                                           alpha=0.92, edgecolor="#3C3C3C", linewidth=1.4))
        ab.set_zorder(10)
        fig.add_artist(ab)
    methods = sorted({r.get("method", "?") for r in results})
    method_str = methods[0] if len(methods) == 1 else "mixed (" + ",".join(methods) + ")"
    _add_suptitle(fig, event, f"{len(results)} runs", "Modal Fit - Comparison",
                  method=method_str, extras=f"{fmin:.1f}-{fmax:.1f} Hz")
    plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0.045, 0, 1, 0.95))
    fig.savefig(plots_dir / "vibrations_comparison_fit.png", dpi=output_dpi, **_SAVE_KW)
    plt.close(fig)
    log.info("  %-20s %-12s %-12s %-12s %-12s", "Run", "Heave [Hz]", "Pitch [Hz]", "Roll [Hz]", "Warp [Hz]")
    for res in results:
        fn, modes = res["fn"], res["mode_labels"]
        def find_freq(label):
            for i, m in enumerate(modes):
                if m == label:
                    return f"{fn[i]:.2f}"
            return "-"
        log.info("  %-20s %-12s %-12s %-12s %-12s",
                 res["name"], find_freq("Heave"), find_freq("Pitch"),
                 find_freq("Roll"), find_freq("Warp"))
    log.info("  Plots saved to: %s", plots_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description="4-DOF body-mode fitting.")
    parser.add_argument("data_file", help="Path to CSV data file with FPushrod channels.")
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
        show_plots=not args.no_plots, displacement_mode=args.displacement_mode,
        expected_freqs=expected_freqs, method=args.method,
        event=args.event, source_type=args.source_type,
        output_dpi=args.output_dpi,
    )

if __name__ == "__main__":
    main()
