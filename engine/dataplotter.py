from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import importlib.util
import logging
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

from . import datafunctions
from .logger import log
from .plot_generators_bar_box import BarBoxMixin
from .plot_generators_misc import HeatmapMixin, PsdHistMixin
from .plot_generators_scatter import ScatterMixin
from .plot_generators_waveform import WaveformMixin


def make_unique(names):
    counts = Counter(names)
    unique_names = []
    seen = {}
    for name in names:
        if counts[name] > 1:
            if name not in seen:
                seen[name] = 1
                unique_names.append(name)
            else:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
        else:
            unique_names.append(name)
    return unique_names


def _find_split_column(df: pd.DataFrame, column: str) -> str | None:
    """Resolve a split_by column name against a DataFrame, case-insensitively.

    Also tries the parquet-style ``_<column>`` / ``<column>`` underscore alias
    so users can write ``nRun`` even when the source column is ``_nRun``.
    """
    if column in df.columns:
        return column
    lower_map = {str(c).lower(): c for c in df.columns}
    target = column.lower()
    if target in lower_map:
        return lower_map[target]
    if not target.startswith("_") and ("_" + target) in lower_map:
        return lower_map["_" + target]
    if target.startswith("_") and target[1:] in lower_map:
        return lower_map[target[1:]]
    return None


def _format_split_key(value) -> str:
    """Render a split_by group value as a filesystem-friendly suffix."""
    if value is None:
        return "None"
    if isinstance(value, float):
        if pd.isna(value):
            return "NaN"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}".replace(".", "p").replace("-", "neg")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    s = str(value).strip()
    return s.replace(" ", "_") if s else "blank"


def _split_by_required_columns(spec) -> list[str] | None:
    """Return the source column(s) a split_by spec depends on.

    Returns ``None`` for callable specs (caller should disable column
    projection and load every column). Returns an empty list for falsy
    specs.
    """
    if not spec:
        return []
    if callable(spec):
        return None
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, dict):
        col = spec.get("column")
        return [col] if col else []
    return []


_CALC_DEP_CACHE = {}


def _extract_calculated_dependencies(func):
    key = id(func)
    cached = _CALC_DEP_CACHE.get(key)
    if cached is not None:
        return cached
    explicit = getattr(func, "__dls_deps__", None)
    if explicit is not None:
        deps = set(explicit)
        _CALC_DEP_CACHE[key] = deps
        return deps
    import inspect
    import re

    try:
        source = inspect.getsource(func)
    except Exception:
        log.debug(
            "Could not extract source for calc-channel func %r; "
            "dependency inference disabled. Decorate the func with "
            "`@calc_channel('col1', 'col2', ...)` to declare deps explicitly.",
            getattr(func, "__name__", repr(func)),
        )
        _CALC_DEP_CACHE[key] = set()
        return _CALC_DEP_CACHE[key]
    matches = re.findall(r"df\['([^']+)'\]|df\[\"([^\"]+)\"\]", source)
    deps = {m[0] or m[1] for m in matches}
    _CALC_DEP_CACHE[key] = deps
    return deps


def collect_referenced_channels(plot_definitions):
    referenced = set()

    def _add(value):
        if isinstance(value, str):
            referenced.add(value)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _add(v)

    for plot_group in plot_definitions or []:
        for plot_def in plot_group or []:
            kind = getattr(plot_def, "kind", None)
            if kind == "waveform":
                _add(plot_def.channels)
                _add(plot_def.x_channel)
                for m in getattr(plot_def, "markers", None) or []:
                    if getattr(m, "condition", None) is not None:
                        referenced.update(datafunctions.collect_gate_channels(m.condition))
            elif kind == "scatter":
                _add(plot_def.x_channel)
                _add(plot_def.y_channel)
                if isinstance(plot_def.best_fit, (list, tuple)):
                    referenced.update(datafunctions.collect_multi_fit_condition_channels(plot_def.best_fit))
                if plot_def.gate is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
                if isinstance(plot_def.color_gate, (list, tuple)) and len(plot_def.color_gate) >= 3:
                    referenced.update(datafunctions.collect_gate_channels(tuple(plot_def.color_gate[:3])))
            elif kind == "psd":
                _add(plot_def.channel)
                if getattr(plot_def, "gate", None) is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
            elif kind == "histogram":
                _add(plot_def.channel)
            elif kind == "bar":
                for ch, _agg in datafunctions.normalize_bar_metric_specs(plot_def.metrics):
                    _add(ch)
                err_metrics = getattr(plot_def, "error_metrics", None)
                if err_metrics:
                    for em in err_metrics:
                        if em:
                            _add(em)
            elif kind == "box":
                _add(plot_def.channels)
                if plot_def.gate is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
            elif kind == "heatmap":
                _add(plot_def.x_channel)
                _add(plot_def.y_channel)
                if plot_def.z_channel:
                    _add(plot_def.z_channel)
                if plot_def.gate is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
            elif kind == "scatter3d":
                _add(plot_def.x_channel)
                _add(plot_def.y_channel)
                _add(plot_def.z_channel)
                if getattr(plot_def, "gate", None) is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
    return sorted(referenced)


def _prepare_slap_vcar_series(df):
    if "sLap" not in df.columns or "vCar" not in df.columns:
        return None, None
    s = pd.to_numeric(df["sLap"], errors="coerce")
    v = pd.to_numeric(df["vCar"], errors="coerce")
    tmp = pd.DataFrame({"s": s, "v": v}).dropna()
    if tmp.empty:
        return None, None
    tmp = tmp[tmp["s"] >= 0].sort_values("s")
    if tmp.empty:
        return None, None
    tmp = tmp.groupby("s", as_index=False)["v"].mean()
    if len(tmp) < 50:
        return None, None
    return tmp["s"].to_numpy(dtype=float), tmp["v"].to_numpy(dtype=float)


def _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset):
    transformed = oth_s * scale + offset
    lo = max(ref_s.min(), transformed.min())
    hi = min(ref_s.max(), transformed.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.arange(np.ceil(lo), np.floor(hi) + 1.0, 5.0)
    if grid.size < 100:
        return None
    ref_interp = np.interp(grid, ref_s, ref_v)
    oth_interp = np.interp(grid, transformed, oth_v)
    if np.std(ref_interp) < 1e-9 or np.std(oth_interp) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ref_interp, oth_interp)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    mae = float(np.mean(np.abs(ref_interp - oth_interp)))
    return corr, mae, grid.size


def estimate_slap_alignment(runs, run_data):
    lines = []
    if not runs:
        return lines
    base_name = runs[0]["name"].lower()
    if base_name not in run_data:
        return [f"{base_name.upper()}: baseline dataframe not loaded"]
    ref_s, ref_v = _prepare_slap_vcar_series(run_data[base_name])
    if ref_s is None:
        return [f"{base_name.upper()}: missing usable sLap/vCar for baseline"]
    ref_range = float(ref_s.max() - ref_s.min())
    if ref_range <= 0:
        return [f"{base_name.upper()}: invalid sLap range for baseline"]
    for run in runs[1:]:
        rn = run["name"].lower()
        if rn not in run_data:
            lines.append(f"{rn.upper()}: dataframe not loaded")
            continue
        oth_s, oth_v = _prepare_slap_vcar_series(run_data[rn])
        if oth_s is None:
            lines.append(f"{rn.upper()}: missing usable sLap/vCar")
            continue
        oth_range = float(oth_s.max() - oth_s.min())
        if oth_range <= 0:
            lines.append(f"{rn.upper()}: invalid sLap range")
            continue
        scale_guess = ref_range / oth_range
        offset_guess = float(ref_s.min() - oth_s.min() * scale_guess)
        best = None
        for scale in np.linspace(scale_guess - 0.01, scale_guess + 0.01, 5):
            for offset in np.linspace(offset_guess - 40, offset_guess + 40, 5):
                score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset)
                if score is None:
                    continue
                corr, mae, n = score
                key = (corr, -mae, n)
                if best is None or key > best["key"]:
                    best = {
                        "scale": float(scale),
                        "offset": float(offset),
                        "corr": corr,
                        "mae": mae,
                        "n": int(n),
                        "key": key,
                    }
        if best is not None:
            try:
                from scipy.optimize import minimize

                def _obj(x):
                    score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, x[0], x[1])
                    if score is None:
                        return 1e6
                    corr, mae, _n = score
                    return -corr + 1e-3 * mae

                result = minimize(
                    _obj,
                    x0=[best["scale"], best["offset"]],
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 200},
                )
                if result.success or result.nit > 0:
                    score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, float(result.x[0]), float(result.x[1]))
                    if score is not None:
                        corr, mae, n = score
                        key = (corr, -mae, n)
                        if key > best["key"]:
                            best = {
                                "scale": float(result.x[0]),
                                "offset": float(result.x[1]),
                                "corr": corr,
                                "mae": mae,
                                "n": int(n),
                                "key": key,
                            }
            except Exception:
                pass
        if best is None:
            lines.append(f"{rn.upper()}: could not estimate sLap mapping")
            continue
        drift_end = (best["scale"] - 1.0) * ref_range
        lines.append(
            f"{rn.upper()} vs {base_name.upper()}: "
            f"scale={best['scale']:.6f}, offset={best['offset']:+.2f} m, "
            f"end_drift_est={drift_end:+.2f} m, "
            f"vCar_corr={best['corr']:.4f}, vCar_mae={best['mae']:.2f} kph, "
            f"samples={best['n']}"
        )
    return lines


def compute_slap_alignment(runs, run_data, target_length=None):
    result = {}
    if not runs:
        return result
    if target_length is not None and target_length > 0:
        for run in runs:
            rn = run["name"].lower()
            if rn not in run_data:
                continue
            df = run_data[rn]
            if "sLap" not in df.columns:
                continue
            s = pd.to_numeric(df["sLap"], errors="coerce")
            s_clean = s.dropna()
            if s_clean.empty:
                continue
            s_range = float(s_clean.max() - s_clean.min())
            if s_range <= 0:
                continue
            scale = target_length / s_range
            if abs(scale - 1.0) < 0.001:
                continue
            result[rn] = scale
        return result
    if len(runs) < 2:
        return result
    base_run = None
    for r in runs:
        if r.get("baseline") or r.get("reference"):
            base_run = r
            break
    if base_run is None:
        base_run = runs[0]
    base_name = base_run["name"].lower()
    if base_name not in run_data:
        return result
    base_df = run_data[base_name]
    if "sLap" not in base_df.columns:
        return result
    base_s = pd.to_numeric(base_df["sLap"], errors="coerce").dropna()
    if base_s.empty:
        return result
    ref_range = float(base_s.max() - base_s.min())
    if ref_range <= 0:
        return result
    for run in runs:
        rn = run["name"].lower()
        if rn == base_name:
            continue
        if rn not in run_data:
            continue
        df = run_data[rn]
        if "sLap" not in df.columns:
            continue
        s = pd.to_numeric(df["sLap"], errors="coerce").dropna()
        if s.empty:
            continue
        s_range = float(s.max() - s.min())
        if s_range <= 0:
            continue
        scale = ref_range / s_range
        if abs(scale - 1.0) < 0.001:
            continue
        result[rn] = scale
    return result


class DataPlotter(WaveformMixin, ScatterMixin, PsdHistMixin, HeatmapMixin, BarBoxMixin):
    def __init__(
        self,
        root_folder: str | Path,
        runs: list[dict],
        plot_definitions: tuple | None = None,
        channel_mappings: dict | None = None,
        channel_transforms: dict | None = None,
        calculated_channels: dict | None = None,
        filters: dict | None = None,
        fig_size: dict | list | None = None,
        units_map: dict | None = None,
        plot_aspect_ratios: dict | None = None,
        sample_rate: float = 100,
        scatter_dot_size: float = 5,
        scatter_transparency: float = 0.7,
        scatter_max_points: int = 45000,
        bar_secondary_axis_ratio: float = 20.0,
        box_plot_settings: dict | None = None,
        output_dir: str | Path | None = None,
        verbose: bool = False,
        output_dpi: int = 300,
        resample_rate: float | None = None,
        vibrations_fit: dict | None = None,
        psd_min_averages_target: int = 200,
        debug_scatter3d_plots: list | None = None,
    ):
        if fig_size is None:
            fig_size = {
                "waveform": (15.5, 6.4),
                "scatter": (10, 8),
                "psd": (10, 8),
                "histogram": (10, 8),
                "bar": (10, 6),
                "box": (10, 6),
            }
        root_folder = Path(root_folder)
        output_dir = Path(output_dir) if output_dir is not None else root_folder
        _AUTO_COLORS = (
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        )
        for _i, _r in enumerate(runs):
            if isinstance(_r, dict):
                if "name" in _r and "run_id" not in _r:
                    _r["run_id"] = str(_r["name"]).lower()
                if not _r.get("color"):
                    _r["color"] = _AUTO_COLORS[_i % len(_AUTO_COLORS)]
        self.runs = runs
        self.verbose = verbose
        self.output_dpi = output_dpi
        self._configure_plot_style()
        self.BAR_SECONDARY_AXIS_RATIO = float(bar_secondary_axis_ratio)
        self.PLOT_DEFINITIONS = plot_definitions
        self.CHANNEL_MAPPINGS = channel_mappings
        self.CALCULATED_CHANNELS = calculated_channels
        self.CHANNEL_TRANSFORMS = channel_transforms
        self.units_map = units_map
        self.FILTER_SAMPLE_RATE = sample_rate
        self.FILTERS = filters
        if resample_rate is not None:
            try:
                rr = float(resample_rate)
            except (TypeError, ValueError):
                rr = 0.0
            self.RESAMPLE_RATE = rr
            if rr > 0:
                self.FILTER_SAMPLE_RATE = rr
        else:
            # `resample_rate=None` means "disable resampling" -- the per-run
            # detected sample rate is kept as-is. `FILTER_SAMPLE_RATE` (the
            # `sample_rate` constructor arg) stays in place purely as a
            # fallback for `_run_fs(run_name)` before per-run detection has
            # populated `run_sample_rates`. We deliberately do NOT copy
            # `sample_rate` into `RESAMPLE_RATE` here -- doing so silently
            # re-enabled resampling at the 100 Hz default and pulled
            # native-1000 Hz DLS runs down with it.
            self.RESAMPLE_RATE = 0.0
        self.SCATTER_DOT_SIZE = scatter_dot_size
        self.SCATTER_TRANSPARENCY = scatter_transparency
        self.SCATTER_MAX_POINTS = scatter_max_points
        if isinstance(fig_size, dict):
            default = fig_size.get("default", (10, 8))
            self.waveform_figsize = fig_size.get("waveform", fig_size.get("default", (15.5, 6.4)))
            self.scatter_FIGSIZE = fig_size.get("scatter", default)
            self.psd_FIGSIZE = fig_size.get("psd", default)
            self.histogram_FIGSIZE = fig_size.get("histogram", default)
            self.bar_FIGSIZE = fig_size.get("bar", fig_size.get("default", (10, 6)))
            self.boxplot_FIGSIZE = fig_size.get("box", self.bar_FIGSIZE)
        elif (
            isinstance(fig_size, (list, tuple))
            and len(fig_size) == 2
            and all(isinstance(v, (int, float)) for v in fig_size)
        ):
            size = (float(fig_size[0]), float(fig_size[1]))
            self.waveform_figsize = self.scatter_FIGSIZE = self.psd_FIGSIZE = size
            self.histogram_FIGSIZE = self.bar_FIGSIZE = self.boxplot_FIGSIZE = size
        else:
            self.waveform_figsize = fig_size[0]
            self.scatter_FIGSIZE = fig_size[1]
            self.psd_FIGSIZE = fig_size[2]
            self.histogram_FIGSIZE = fig_size[3]
            self.bar_FIGSIZE = fig_size[4] if len(fig_size) > 4 else (10, 6)
            self.boxplot_FIGSIZE = fig_size[5] if len(fig_size) > 5 else self.bar_FIGSIZE
        self.plot_aspect_ratios = plot_aspect_ratios or {}
        self.BOX_PLOT_SETTINGS = box_plot_settings or {}
        self.debug_scatter3d_plots = list(debug_scatter3d_plots or [])
        self.run_filepaths = {}
        self.run_data = {}
        self.run_units = {}
        self.run_required_cols = {}
        self.run_sample_rates = {}
        self.modal_results = {}
        self.VIBRATIONS_FIT = vibrations_fit
        self.PSD_MIN_AVERAGES_TARGET = int(psd_min_averages_target)
        self._gated_data_cache = {}
        self._psd_cache = {}
        self._outlier_log = []
        self._reverse_mappings = {}
        self._parquet_alias_cache = {}
        self._loaded = False
        self._preprocessed = False
        if self.CHANNEL_MAPPINGS:
            for source_type, mapping in self.CHANNEL_MAPPINGS.items():
                if mapping:
                    self._reverse_mappings[source_type] = {mapped: raw for raw, mapped in mapping.items()}
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.load_data(root_folder)
        self.preprocess_data()

    PLOT_FONT = {
        "family": "Montserrat",
        "fallback": ["DejaVu Sans", "Arial", "sans-serif"],
        "title_size": 14,
        "label_size": 11,
        "tick_size": 10,
        "legend_size": 11,
        "figure_title_size": 16,
    }
    GRID_STYLE = {
        "major": {"alpha": 0.30, "linewidth": 0.6},
        "minor": {"alpha": 0.22, "linewidth": 0.4},
    }

    def _configure_plot_style(self):
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        preferred_font = (
            self.PLOT_FONT["family"] if self.PLOT_FONT["family"] in available_fonts else self.PLOT_FONT["fallback"][0]
        )
        ink = "#1A1A1A"
        plt.rcParams.update(
            {
                "font.family": preferred_font,
                "font.sans-serif": [self.PLOT_FONT["family"]] + self.PLOT_FONT["fallback"],
                "axes.titlesize": self.PLOT_FONT["title_size"],
                "axes.titleweight": "bold",
                "axes.labelsize": self.PLOT_FONT["label_size"],
                "axes.labelweight": "bold",
                "axes.edgecolor": ink,
                "axes.labelcolor": ink,
                "axes.titlecolor": ink,
                "xtick.color": ink,
                "ytick.color": ink,
                "xtick.labelsize": self.PLOT_FONT["tick_size"],
                "ytick.labelsize": self.PLOT_FONT["tick_size"],
                "xtick.minor.visible": True,
                "ytick.minor.visible": True,
                "text.color": ink,
                "legend.fontsize": self.PLOT_FONT["legend_size"],
                "figure.titlesize": self.PLOT_FONT["figure_title_size"],
                "figure.titleweight": "bold",
            }
        )

    def _apply_grid(self, ax, which="both", axis="both"):
        if which in ("major", "both"):
            ax.grid(True, which="major", axis=axis, **self.GRID_STYLE["major"])
        if which in ("minor", "both"):
            ax.grid(True, which="minor", axis=axis, **self.GRID_STYLE["minor"])
        ax.set_axisbelow(True)

    def _run_fs(self, run_name: str) -> float:
        """Sample rate for a run: detected per-run value, else the global default.

        After ``_preprocess_data`` the per-run rate equals ``RESAMPLE_RATE`` when
        resampling is enabled, otherwise the rate inferred from the run's time
        column. ``FILTER_SAMPLE_RATE`` is only used as a fallback before detection
        has run (e.g. inside ``_clean_data`` on the very first call).
        """
        pair = self.run_sample_rates.get(run_name)
        if pair and pair[0]:
            return float(pair[0])
        return float(self.FILTER_SAMPLE_RATE)

    @staticmethod
    def _apply_2d_axis_limits(ax, axis_limits, *, log_scale_y=False, y_floor=1e-4):
        if not axis_limits:
            return False, False
        (xmin, xmax), (ymin, ymax) = axis_limits
        has_x = xmin is not None or xmax is not None
        has_y = ymin is not None or ymax is not None
        if has_x:
            ax.set_xlim(left=xmin, right=xmax)
        if has_y:
            if log_scale_y and ymin is not None:
                ymin = max(ymin, y_floor)
            ax.set_ylim(bottom=ymin, top=ymax)
        return has_x, has_y

    @staticmethod
    def _draw_horizontal_reference_lines(ax, refs, *, label=True):
        if not refs:
            return
        y0, y1 = ax.get_ylim()
        new_y0, new_y1 = y0, y1
        for v in refs:
            if not np.isfinite(v):
                continue
            pad = (y1 - y0) * 0.05 if (y1 > y0) else 0.0
            new_y0 = min(new_y0, v - pad)
            new_y1 = max(new_y1, v + pad)
        if (new_y0, new_y1) != (y0, y1):
            ax.set_ylim(new_y0, new_y1)
        for v in refs:
            if not np.isfinite(v):
                continue
            ax.axhline(v, color="#4A4A4A", linestyle="--", linewidth=0.8, alpha=0.65, zorder=1)
            if label:
                ax.text(
                    0.995,
                    v,
                    f" {v:g}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#333333",
                )

    @staticmethod
    def _draw_static_markers(axes, markers, *, label_y=1.01, x_clip=True):
        if not markers:
            return
        try:
            ax_iter = list(axes)
        except TypeError:
            ax_iter = [axes]
        for ax in ax_iter:
            xl, xr = ax.get_xlim() if x_clip else (None, None)
            for m in markers:
                if m.condition is not None:
                    continue
                if x_clip and not (xl <= m.x <= xr):
                    continue
                color = m.color or "#5E5E5E"
                ax.axvline(m.x, color=color, linestyle=m.linestyle, linewidth=1.2, alpha=0.7, zorder=2)
                if m.label and m.show_label:
                    ax.text(
                        m.x,
                        label_y,
                        m.label,
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9
                        ),
                        zorder=12,
                    )

    def _get_required_source_columns(self, source_type):
        required_channels = set()

        def _extract_channels(spec_item):
            if isinstance(spec_item, str):
                required_channels.add(spec_item)
                return
            if isinstance(spec_item, (list, tuple)):
                for part in spec_item:
                    _extract_channels(part)

        if self.PLOT_DEFINITIONS:
            for plot_group in self.PLOT_DEFINITIONS:
                if not plot_group:
                    continue
                for plot_def in plot_group:
                    kind = getattr(plot_def, "kind", None)
                    if kind == "waveform":
                        _extract_channels(plot_def.channels)
                        if plot_def.x_channel:
                            required_channels.add(plot_def.x_channel)
                        for m in getattr(plot_def, "markers", None) or []:
                            if getattr(m, "condition", None) is not None:
                                required_channels.update(datafunctions.collect_gate_channels(m.condition))
                    elif kind == "scatter":
                        required_channels.add(plot_def.x_channel)
                        required_channels.add(plot_def.y_channel)
                        if isinstance(plot_def.best_fit, (list, tuple)):
                            required_channels.update(
                                datafunctions.collect_multi_fit_condition_channels(plot_def.best_fit)
                            )
                        if plot_def.gate is not None:
                            required_channels.update(datafunctions.collect_gate_channels(plot_def.gate))
                        if isinstance(plot_def.color_gate, (list, tuple)) and len(plot_def.color_gate) >= 3:
                            required_channels.update(
                                datafunctions.collect_gate_channels(tuple(plot_def.color_gate[:3]))
                            )
                    elif kind == "psd":
                        _extract_channels(plot_def.channel)
                        if getattr(plot_def, "gate", None) is not None:
                            required_channels.update(datafunctions.collect_gate_channels(plot_def.gate))
                    elif kind == "histogram":
                        _extract_channels(plot_def.channel)
                    elif kind == "bar":
                        for ch, _agg in datafunctions.normalize_bar_metric_specs(plot_def.metrics):
                            _extract_channels(ch)
                    elif kind == "box":
                        _extract_channels(plot_def.channels)
                        if plot_def.gate is not None:
                            required_channels.update(datafunctions.collect_gate_channels(plot_def.gate))
                    elif kind == "box_grid":
                        _extract_channels(plot_def.channels)
                        for _gate in plot_def.rows.values():
                            if _gate is not None:
                                required_channels.update(datafunctions.collect_gate_channels(_gate))
                        for _gate in plot_def.cols.values():
                            if _gate is not None:
                                required_channels.update(datafunctions.collect_gate_channels(_gate))
                    elif kind == "heatmap":
                        required_channels.add(plot_def.x_channel)
                        required_channels.add(plot_def.y_channel)
                        if plot_def.z_channel:
                            required_channels.add(plot_def.z_channel)
                        if plot_def.gate is not None:
                            required_channels.update(datafunctions.collect_gate_channels(plot_def.gate))
        for plot_def in getattr(self, "debug_scatter3d_plots", None) or []:
            required_channels.add(plot_def.x_channel)
            required_channels.add(plot_def.y_channel)
            required_channels.add(plot_def.z_channel)
            if getattr(plot_def, "gate", None) is not None:
                required_channels.update(datafunctions.collect_gate_channels(plot_def.gate))
        for support in ("sLap", "tLap", "vCar", "TimeIntoExport"):
            required_channels.add(support)
        resolved_channels = set()
        to_process = deque(required_channels)
        processed = set()
        while to_process:
            channel = to_process.popleft()
            if channel in processed:
                continue
            processed.add(channel)
            calc_set = self.CALCULATED_CHANNELS
            if isinstance(calc_set, dict):
                calc_set = calc_set.get(source_type) or calc_set
            if isinstance(calc_set, dict) and channel in calc_set:
                deps = _extract_calculated_dependencies(calc_set[channel])
                for dep in deps:
                    if dep not in processed:
                        to_process.append(dep)
            else:
                resolved_channels.add(channel)
        source_columns = set()
        mappings = self._reverse_mappings.get(source_type, {})
        for ch in resolved_channels:
            source_columns.add(ch)
            source_columns.add(mappings.get(ch, ch))
        return source_columns

    def _available_parquet_engines(self):
        engines = []
        if importlib.util.find_spec("pyarrow") is not None:
            engines.append("pyarrow")
        if importlib.util.find_spec("fastparquet") is not None:
            engines.append("fastparquet")
        return engines

    def _get_parquet_schema_columns(self, file_path, engine):
        try:
            if engine == "pyarrow":
                import pyarrow.parquet as pq

                return [str(c).strip() for c in pq.read_schema(file_path).names]
            elif engine == "fastparquet":
                import fastparquet

                return [str(c).strip() for c in fastparquet.ParquetFile(str(file_path)).columns]
        except Exception:
            pass
        return None

    def _normalize_parquet_column_aliases(self, df):
        raw_columns = [str(c).strip() for c in df.columns]
        rename_map = {}
        existing = set(raw_columns)
        for col in raw_columns:
            if col.startswith("_") and len(col) > 1 and col[1].isalpha():
                canonical = col[1].upper() + col[2:]
                if canonical not in existing:
                    rename_map[col] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _find_parquet_column(self, df, logical_name):
        columns = [str(c).strip() for c in df.columns]
        column_set = set(columns)
        lower_target = logical_name.lower()
        for candidate in [
            logical_name,
            logical_name.lower(),
            logical_name.upper(),
            f"_{logical_name}",
            f"_{logical_name.lower()}",
            logical_name[0].upper() + logical_name[1:],
        ]:
            if candidate in column_set:
                return candidate
        insensitive = [c for c in columns if c.lower() == lower_target]
        if insensitive:
            if len(insensitive) > 1:
                log.warning(
                    "Multiple %s-like columns found: %s. Using '%s'.",
                    logical_name,
                    ", ".join(insensitive),
                    insensitive[0],
                )
            return insensitive[0]
        return None

    def _resolve_required_parquet_columns(self, schema_cols, columns_to_load, nrun=None, nlap=None):
        raw_set = set(schema_cols)
        raw_lower = {c.lower(): c for c in schema_cols}
        schema_key = tuple(schema_cols)
        canonical_to_raw = self._parquet_alias_cache.get(schema_key)
        if canonical_to_raw is None:
            canonical_to_raw = {}
            for raw in schema_cols:
                if raw.startswith("_") and len(raw) > 1 and raw[1].isalpha():
                    canonical = raw[1].upper() + raw[2:]
                    # Also expose the camelCase form so users can write
                    # 'nRun' when the parquet column is '_nRun'.
                    camelcase = raw[1].lower() + raw[2:]
                    canonical_to_raw.setdefault(camelcase, raw)
                else:
                    canonical = raw
                canonical_to_raw.setdefault(canonical, raw)
                canonical_to_raw.setdefault(raw, raw)
            self._parquet_alias_cache[schema_key] = canonical_to_raw
        needed = set()
        for candidates, flag in [
            (["nRun", "nrun", "_nRun", "_nrun", "NRun"], nrun),
            (["nLap", "nlap", "_nLap", "_nlap", "NLap"], nlap),
        ]:
            if flag is not None:
                for c in candidates:
                    if c in raw_set:
                        needed.add(c)
                        break
                else:
                    target_lower = candidates[0].lower()
                    if target_lower in raw_lower:
                        needed.add(raw_lower[target_lower])
        if columns_to_load:
            for logical in columns_to_load:
                if logical in canonical_to_raw:
                    needed.add(canonical_to_raw[logical])
                elif logical in raw_set:
                    needed.add(logical)
                else:
                    lower = logical.lower()
                    if lower in raw_lower:
                        needed.add(raw_lower[lower])
        return sorted(needed) if needed else None

    def _apply_parquet_rank_value_filter(
        self,
        df,
        filter_spec,
        column_logical_name,
        file_path,
        run_name,
        is_rank=False,
        raise_on_missing_column=True,
        raise_on_empty_result=True,
    ):
        if filter_spec is None:
            return df
        is_multi = isinstance(filter_spec, (range, list, tuple, set, frozenset))
        if is_multi:
            specs = []
            seen = set()
            for v in filter_spec:
                if v not in seen:
                    seen.add(v)
                    specs.append(v)
            if not specs:
                return df
            if len(specs) == 1:
                filter_spec = specs[0]
                is_multi = False
        else:
            specs = [filter_spec]
        run_label = run_name.upper() if run_name else file_path.name
        run_col = self._find_parquet_column(df, column_logical_name)
        if run_col is None:
            msg = (
                f"Run '{run_label}' requested {column_logical_name.lower()}={filter_spec}, "
                f"but parquet has no '{column_logical_name}' column"
            )
            if raise_on_missing_column:
                raise KeyError(msg + ".")
            else:
                log.warning("%s. Skipping filter.", msg)
                return df
        series = df[run_col]
        numeric = pd.to_numeric(series, errors="coerce")
        if is_multi:
            if is_rank:
                if numeric.notna().any():
                    unique_vals = sorted(numeric.dropna().unique().tolist())
                else:
                    unique_vals = sorted(
                        [v for v in series.astype(str).str.strip().unique() if v and v.lower() != "nan"]
                    )
                target_values = []
                for s in specs:
                    rank = int(pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0])
                    if rank < 1 or rank > len(unique_vals):
                        raise ValueError(
                            f"Run '{run_label}' requested {column_logical_name.lower()}={rank}, "
                            f"but only {len(unique_vals)} unique values exist."
                        )
                    target_values.append(unique_vals[rank - 1])
                if numeric.notna().any():
                    mask = numeric.isin(target_values)
                else:
                    mask = series.astype(str).str.strip().isin([str(v) for v in target_values])
            else:
                target_numerics = pd.to_numeric(pd.Series(list(specs)), errors="coerce")
                if target_numerics.notna().all():
                    mask = numeric.isin(target_numerics.astype(float).tolist())
                    target_values = target_numerics.astype(float).tolist()
                else:
                    mask = series.astype(str).str.strip().isin([str(s).strip() for s in specs])
                    target_values = list(specs)
            filtered = df.loc[mask].copy()
            if filtered.empty:
                msg = (
                    f"Run '{run_label}' {column_logical_name.lower()}={list(specs)} "
                    f"produced 0 rows from column '{run_col}'."
                )
                if raise_on_empty_result:
                    raise ValueError(msg)
                log.warning("%s", msg)
                return df
            log.info(
                "Run '%s' filtered by %s: %s in %s -> %d/%d rows kept.",
                run_label,
                column_logical_name,
                run_col,
                target_values,
                len(filtered),
                len(df),
            )
            return filtered
        if is_rank:
            rank = int(pd.to_numeric(pd.Series([filter_spec]), errors="coerce").iloc[0])
            if rank < 1:
                raise ValueError(f"Run '{run_label}' {column_logical_name.lower()} must be >= 1.")
            if numeric.notna().any():
                unique_vals = sorted(numeric.dropna().unique().tolist())
            else:
                unique_vals = sorted([v for v in series.astype(str).str.strip().unique() if v and v.lower() != "nan"])
            if rank > len(unique_vals):
                raise ValueError(
                    f"Run '{run_label}' requested {column_logical_name.lower()}={rank}, "
                    f"but only {len(unique_vals)} unique values exist. "
                    f"Available: {unique_vals[:12]}" + (" ..." if len(unique_vals) > 12 else "")
                )
            target_value = unique_vals[rank - 1]
            mask = (
                (numeric == target_value)
                if numeric.notna().any()
                else (series.astype(str).str.strip() == str(target_value))
            )
        else:
            target_numeric = pd.to_numeric(pd.Series([filter_spec]), errors="coerce").iloc[0]
            if pd.notna(target_numeric):
                mask = numeric == float(target_numeric)
            else:
                mask = series.astype(str).str.strip() == str(filter_spec).strip()
            target_value = filter_spec
        filtered = df.loc[mask].copy()
        if filtered.empty:
            msg = (
                f"Run '{run_label}' {column_logical_name.lower()}={filter_spec} "
                f"produced 0 rows from column '{run_col}'."
            )
            if raise_on_empty_result:
                raise ValueError(msg)
            else:
                log.warning("%s", msg)
                return df
        log.info(
            "Run '%s' filtered by %s: %s=%s -> %s=%s (%d/%d rows kept).",
            run_label,
            column_logical_name,
            column_logical_name.lower(),
            filter_spec,
            run_col,
            target_value,
            len(filtered),
            len(df),
        )
        return filtered

    def _load_parquet_with_fallback(
        self, file_path, columns_to_load=None, parquet_nrun=None, parquet_nlap=None, run_name=""
    ):
        available_engines = self._available_parquet_engines()
        if not available_engines:
            raise ImportError("Parquet input requires 'pyarrow' or 'fastparquet', but neither is installed.")
        errors = []
        for engine in available_engines:
            try:
                schema_cols = self._get_parquet_schema_columns(file_path, engine)
                if schema_cols is not None and columns_to_load:
                    col_subset = self._resolve_required_parquet_columns(
                        schema_cols,
                        columns_to_load,
                        nrun=parquet_nrun,
                        nlap=parquet_nlap,
                    )
                else:
                    col_subset = None
                read_kwargs = {"engine": engine}
                if col_subset:
                    read_kwargs["columns"] = col_subset
                if parquet_nrun is not None and parquet_nlap is not None:
                    log.info(
                        "Run '%s' provided both nrun and nlap; applying nrun filter and ignoring nlap.",
                        run_name.upper() if run_name else file_path.name,
                    )
                df = pd.read_parquet(file_path, **read_kwargs)
                df.columns = [str(c).strip() for c in df.columns]
                df = self._normalize_parquet_column_aliases(df)
                if parquet_nrun is not None:
                    df = self._apply_parquet_rank_value_filter(
                        df,
                        filter_spec=parquet_nrun,
                        column_logical_name="nRun",
                        file_path=file_path,
                        run_name=run_name,
                        is_rank=True,
                        raise_on_missing_column=True,
                        raise_on_empty_result=True,
                    )
                elif parquet_nlap is not None:
                    df = self._apply_parquet_rank_value_filter(
                        df,
                        filter_spec=parquet_nlap,
                        column_logical_name="nLap",
                        file_path=file_path,
                        run_name=run_name,
                        is_rank=False,
                        raise_on_missing_column=False,
                        raise_on_empty_result=False,
                    )
                if columns_to_load:
                    requested = sorted(set(columns_to_load))
                    df_cols = list(df.columns)
                    df_cols_set = set(df_cols)
                    df_cols_lower = {str(c).lower(): c for c in df_cols}
                    available = []
                    missing = []
                    seen_avail = set()
                    for c in requested:
                        hit = None
                        if c in df_cols_set:
                            hit = c
                        elif (
                            c.startswith("_")
                            and len(c) > 1
                            and c[1].isalpha()
                            and (c[1].upper() + c[2:]) in df_cols_set
                        ):
                            hit = c[1].upper() + c[2:]
                        elif len(c) > 1 and c[0].isalpha() and c[0].islower() and (c[0].upper() + c[1:]) in df_cols_set:
                            # e.g. user asked for 'nRun'; parquet column
                            # normalised from '_nRun' is 'NRun'.
                            hit = c[0].upper() + c[1:]
                        elif c.lower() in df_cols_lower:
                            # Case-insensitive fallback (e.g. user 'nRun',
                            # parquet column 'nrun' with no underscore).
                            hit = df_cols_lower[c.lower()]
                        if hit is None:
                            missing.append(c)
                            continue
                        if hit not in seen_avail:
                            available.append(hit)
                            seen_avail.add(hit)
                    if missing and self.verbose:
                        log.debug(
                            "Parquet '%s' missing %d channel(s): %s%s",
                            file_path.name,
                            len(missing),
                            ", ".join(missing[:10]),
                            " ..." if len(missing) > 10 else "",
                        )
                    if available:
                        df = df[available]
                    else:
                        raise KeyError(f"No requested channels found in parquet. Requested: {requested[:10]}")
                return df
            except Exception as exc:
                errors.append(f"{engine}: {exc}")
        raise RuntimeError(
            f"Unable to load parquet '{file_path}' via engines {available_engines}. Errors: {' | '.join(errors)}"
        )

    def _load_run_data(
        self,
        file_path,
        use_python_engine=False,
        columns_to_load=None,
        parquet_nrun=None,
        parquet_nlap=None,
        run_name="",
    ):
        try:
            if file_path.suffix.lower() == ".parquet":
                df = self._load_parquet_with_fallback(
                    file_path,
                    columns_to_load=columns_to_load,
                    parquet_nrun=parquet_nrun,
                    parquet_nlap=parquet_nlap,
                    run_name=run_name,
                )
                df.columns = make_unique([str(c).strip() for c in df.columns])
                units = {c: "" for c in df.columns}
                return df, df.columns, units
            with open(file_path) as f:
                lines = f.readlines()
            header = make_unique(lines[1].strip().split(","))
            units_row = lines[2].strip().split(",")
            units = dict(zip(header, units_row))
            if columns_to_load:
                cols_to_read = [c for c in header if c in set(columns_to_load)]
            else:
                cols_to_read = None
            kwargs = dict(
                sep=",",
                skiprows=3,
                header=None,
                names=header,
                on_bad_lines="skip",
                usecols=cols_to_read,
            )
            if use_python_engine:
                kwargs["engine"] = "python"
            else:
                kwargs["low_memory"] = False
            df = pd.read_csv(file_path, **kwargs)
            units = {c: units.get(c, "") for c in df.columns}
            return df, df.columns, units
        except Exception as e:
            log.error("Failed to load '%s': %s", file_path, e)
            raise

    def _clean_data(self):
        for run_name in list(self.run_data.keys()):
            df = datafunctions.convert_yes_no_to_binary(self.run_data[run_name])
            # Per-run interp limit (= 1 s of samples) so high-rate sources don't
            # get bridged with stale 100 Hz spacing. Detection here is independent
            # of the post-resample detection a few steps later in _preprocess_data.
            detected_rate, _ = datafunctions.detect_sample_rate(
                df,
                default=self.FILTER_SAMPLE_RATE,
            )
            interp_limit = max(1, int(detected_rate))
            drop_cols = []
            for col in list(df.columns):
                if col == "TimeIntoExport" and not pd.api.types.is_numeric_dtype(df[col]):
                    td = pd.to_timedelta(df[col].astype(str).str.strip(), errors="coerce")
                    if td.notna().any():
                        df[col] = td.dt.total_seconds()
                        continue
                if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                    non_nan = df[col].dropna()
                    if any(isinstance(x, str) for x in non_nan):
                        drop_cols.append(col)
                        continue
                df[col] = datafunctions.sanitize_numeric_series(df[col])
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                if self.verbose:
                    for col in drop_cols:
                        log.debug("Dropped '%s' from run '%s' (string column)", col, run_name)
            if df.empty:
                self.run_data[run_name] = df
                continue
            required = self.run_required_cols.get(run_name)
            if required:
                cols_to_interp = [c for c in df.columns if c in required]
            else:
                cols_to_interp = list(df.columns)
            if cols_to_interp:
                df[cols_to_interp] = df[cols_to_interp].interpolate(
                    method="linear",
                    limit=interp_limit,
                    axis=0,
                )
            self.run_data[run_name] = df

    def _ensure_preprocessed(self):
        if not self._loaded:
            raise RuntimeError("Data has not been loaded.")
        if not self._preprocessed:
            raise RuntimeError("Data has not been preprocessed.")

    def _cached_psd_with_segments(self, run_name, channel, nperseg, gate_spec=None):
        gate_key = repr(gate_spec) if gate_spec is not None else None
        key = (run_name, channel, nperseg, gate_key)
        cached = self._psd_cache.get(key)
        if cached is not None:
            return cached
        df = self.run_data.get(run_name)
        if df is None or channel not in df.columns:
            return None, None, 0
        signal = np.asarray(df[channel], dtype=float)
        rate = self.run_sample_rates.get(run_name, (self.FILTER_SAMPLE_RATE, "default"))[0]
        if gate_spec is None:
            finite_n = int(np.isfinite(signal).sum())
            if finite_n < nperseg and finite_n >= 8:
                effective = min(nperseg, finite_n)
                if effective < max(64, nperseg // 4):
                    log.warning(
                        "PSD '%s'/'%s': only %d finite samples — nperseg capped from %d to %d "
                        "(coarse frequency resolution, low averaging).",
                        run_name,
                        channel,
                        finite_n,
                        nperseg,
                        effective,
                    )
            freq, power = datafunctions.calculate_psd(signal, rate, nperseg=nperseg)
            if freq is not None:
                eff_n = min(nperseg, int(np.isfinite(signal).sum()))
                step = max(1, eff_n // 2)
                n_segs = max(1, 1 + (int(np.isfinite(signal).sum()) - eff_n) // step)
            else:
                n_segs = 0
        else:
            try:
                mask = datafunctions.compute_gate_mask(df, gate_spec).to_numpy()
            except Exception as exc:
                log.warning(
                    "PSD '%s'/'%s': gate evaluation failed (%s). Skipping.",
                    run_name,
                    channel,
                    exc,
                )
                self._psd_cache[key] = (None, None, 0)
                return None, None, 0
            freq, power, n_segs = datafunctions.calculate_segmented_psd(
                signal,
                mask,
                rate,
                nperseg=nperseg,
            )
            if freq is None:
                log.warning(
                    "PSD '%s'/'%s': no gated segment >= nperseg (%d). Skipping.",
                    run_name,
                    channel,
                    nperseg,
                )
        self._psd_cache[key] = (freq, power, n_segs)
        return freq, power, n_segs

    def _suggest_similar_channels(self, run_name, missing_channel, max_suggestions=5):
        df = self.run_data.get(run_name)
        if df is None:
            return []
        return datafunctions.suggest_similar_channels(missing_channel, list(df.columns), max_results=max_suggestions)

    def _format_missing_channel_hint(self, run_name, missing_channel):
        suggestions = self._suggest_similar_channels(run_name, missing_channel)
        if suggestions:
            return f"  Similar available: {', '.join(suggestions)}"
        return ""

    def _get_plot_group(self, index):
        if not self.PLOT_DEFINITIONS or len(self.PLOT_DEFINITIONS) <= index:
            return []
        return self.PLOT_DEFINITIONS[index] or []

    def _sanitize_plot_filename(self, prefix, plot_name, suffix=""):
        safe = (
            plot_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        )
        subdir = self.plots_dir / prefix
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return f"{prefix}/{prefix}_{safe}{suffix}.png"

    def _resolve_plot_figsize(self, filename, default_size, *, min_height=None):
        w0, h0 = default_size
        target_aspect = self.plot_aspect_ratios.get(filename)
        if isinstance(target_aspect, (list, tuple)):
            target_aspect = sum(target_aspect) / len(target_aspect)
        if target_aspect is None:
            w, h = w0, h0
        else:
            h = h0
            w = h * target_aspect
        if min_height:
            h = max(h, min_height)
            if target_aspect:
                w = h * target_aspect
            elif h > h0:
                # Height was forced above the default; scale width to preserve
                # the default aspect ratio so all waveforms look consistent.
                w = h * (w0 / h0)
        return (w, h)

    def _add_axis_edge_padding(self, ax, x_pad_ratio=0.02, y_pad_ratio=0.03):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmax > xmin:
            pad = (xmax - xmin) * x_pad_ratio
            ax.set_xlim(xmin - pad, xmax + pad)
        if ymax > ymin:
            pad = (ymax - ymin) * y_pad_ratio
            ax.set_ylim(ymin - pad, ymax + pad)

    def _get_filtered_run_dataframe(self, run_name, gate_spec=None):
        if gate_spec is None:
            return self.run_data.get(run_name)
        cache_key = (run_name, repr(gate_spec))
        cached = self._gated_data_cache.get(cache_key)
        if cached is not None:
            return cached
        df = self.run_data.get(run_name)
        if df is None:
            return None
        filtered = datafunctions.apply_gate_to_dataframe(df, gate_spec)
        self._gated_data_cache[cache_key] = filtered
        return filtered

    def load_data(self, root_folder: str | Path) -> dict[str, pd.DataFrame]:
        root_folder = Path(root_folder)
        self._root_folder = root_folder
        self._loaded = False
        self._preprocessed = False
        self.run_filepaths = {}
        self.run_data = {}
        self.run_units = {}
        self.run_required_cols = {}
        self._gated_data_cache = {}
        loaded_runs = []
        for run in self.runs:
            run_name = run["name"].lower()
            if "_consolidate_sources" in run:
                # Synthetic consolidated run — data is built post-preprocess in
                # _build_consolidated_runs(). Skip file load; keep it in
                # self.runs so downstream steps know it exists.
                loaded_runs.append(run)
                continue
            file_path = root_folder / run["file"]
            if not file_path.exists():
                log.warning(
                    "Missing data file for run '%s': %s. Skipping run.",
                    run_name,
                    file_path,
                )
                continue
            try:
                use_python_engine = run.get("use_python_engine", False)
                self.run_required_cols[run_name] = self._get_required_source_columns(run.get("type", run_name))
                # Ensure the split_by column survives parquet column
                # projection — otherwise the loader would strip it.
                split_cols = _split_by_required_columns(run.get("split_by"))
                if split_cols is None:
                    columns_to_load = None  # load all (callable spec)
                else:
                    if self.run_required_cols[run_name] is None:
                        columns_to_load = None
                    else:
                        columns_to_load = set(self.run_required_cols[run_name]) | set(split_cols)
                        self.run_required_cols[run_name] = columns_to_load
                data, _, units = self._load_run_data(
                    file_path,
                    use_python_engine=use_python_engine,
                    columns_to_load=columns_to_load,
                    parquet_nrun=run.get("nrun"),
                    parquet_nlap=run.get("nlap"),
                    run_name=run_name,
                )
                nlap_spec = run.get("nlap")
                if (
                    nlap_spec is not None
                    and isinstance(nlap_spec, (list, tuple, range, set, frozenset))
                    and "nLap" in data.columns
                ):
                    wanted = list(nlap_spec)
                    if wanted:
                        before = len(data)
                        mask = pd.to_numeric(data["nLap"], errors="coerce").isin(wanted)
                        if mask.any():
                            data = data[mask].reset_index(drop=True)
                            log.info(
                                "Run '%s': nlap filter %s -> kept %d/%d rows.",
                                run_name,
                                wanted,
                                len(data),
                                before,
                            )
                        else:
                            log.warning(
                                "Run '%s': nlap filter %s matched no rows; keeping all data.",
                                run_name,
                                wanted,
                            )
                best_n = run.get("best_n")
                if best_n is not None and "nLap" in data.columns:
                    try:
                        n_keep = int(best_n)
                    except (TypeError, ValueError):
                        n_keep = 0
                    if n_keep > 0:
                        time_col = next(
                            (c for c in ("tLap_Calc", "tLap", "Time", "time") if c in data.columns),
                            None,
                        )
                        if time_col is None:
                            log.warning(
                                "Run '%s': best_n=%d requested but no time column "
                                "(tLap_Calc/tLap/Time) found; ignoring.",
                                run_name,
                                n_keep,
                            )
                        else:
                            durations = data.groupby("nLap")[time_col].agg(lambda s: s.max() - s.min()).sort_values()
                            durations = durations[durations > 0]
                            keep_laps = list(durations.head(n_keep).index)
                            if keep_laps:
                                before = len(data)
                                data = data[data["nLap"].isin(keep_laps)].reset_index(drop=True)
                                log.info(
                                    "Run '%s' best_n=%d -> kept laps %s (%d/%d rows).",
                                    run_name,
                                    n_keep,
                                    keep_laps,
                                    len(data),
                                    before,
                                )
            except Exception as exc:
                log.warning(
                    "Failed to load run '%s' from %s: %s. Skipping run.",
                    run_name,
                    file_path,
                    exc,
                )
                self.run_required_cols.pop(run_name, None)
                continue
            self.run_filepaths[run_name] = file_path
            self.run_data[run_name] = data
            self.run_units[run_name] = units
            loaded_runs.append(run)
        self.runs = loaded_runs if loaded_runs else []
        self._loaded = True
        self._expand_split_runs()
        return self.run_data

    def preprocess_data(self) -> None:
        if not self._loaded:
            raise RuntimeError("Data must be loaded before preprocessing.")
        self._gated_data_cache.clear()
        missing_transform_warned: set = set()
        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            source_type = run.get("type", name)
            self.run_data[name] = datafunctions.apply_channel_mappings(
                self.run_data[name], self.CHANNEL_MAPPINGS, source_type
            )
            self.run_data[name] = datafunctions.apply_transformations(
                self.run_data[name],
                source_type,
                self.CHANNEL_TRANSFORMS,
                missing_warned=missing_transform_warned,
            )
        self._clean_data()
        if self.RESAMPLE_RATE and self.RESAMPLE_RATE > 0:
            for run in self.runs:
                name = run["name"].lower()
                if name not in self.run_data:
                    continue
                self.run_data[name] = datafunctions.resample_to_uniform_rate(
                    self.run_data[name],
                    self.RESAMPLE_RATE,
                    run_name=name,
                )
        self._align_slap()
        detected_rates = []
        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            rate, source = datafunctions.detect_sample_rate(self.run_data[name], default=self.FILTER_SAMPLE_RATE)
            self.run_sample_rates[name] = (rate, source)
            detected_rates.append(rate)
            if self.verbose:
                log.debug("[%s] sample rate: %.1f Hz (source: %s)", name, rate, source)
        if detected_rates:
            rmin, rmax = min(detected_rates), max(detected_rates)
            # Keep the global FILTER_SAMPLE_RATE in sync with reality so any
            # consumer that still reads it directly (or has no per-run name in
            # scope) gets a sensible value. When resampling is active every run
            # is already at RESAMPLE_RATE, so we don't overwrite that.
            if not (self.RESAMPLE_RATE and self.RESAMPLE_RATE > 0):
                representative = float(np.median(detected_rates))
                if representative > 0 and representative != self.FILTER_SAMPLE_RATE:
                    log.debug(
                        "FILTER_SAMPLE_RATE updated from %.1f Hz to %.1f Hz (per-run detection).",
                        self.FILTER_SAMPLE_RATE,
                        representative,
                    )
                    self.FILTER_SAMPLE_RATE = representative
            if rmin > 0 and (rmax / rmin) > 1.05:
                log.warning(
                    "Per-run sample rates vary by %.2fx (%.1f–%.1f Hz). "
                    "Filters and PSDs use each run's own rate; cross-run peak "
                    "comparisons may still show different frequency resolution.",
                    rmax / rmin,
                    rmin,
                    rmax,
                )
        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            required = self.run_required_cols.get(name)
            required_set = set(required) if required else None
            self.run_data[name] = datafunctions.apply_calculated_channels(
                self.run_data[name],
                name,
                self.CALCULATED_CHANNELS,
                required_channels=required_set,
            )
        self._compute_tdiff_channel()
        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            required = self.run_required_cols.get(name)
            required_set = set(required) if required else None
            self.run_data[name] = datafunctions.apply_filters(
                self.run_data[name],
                self.FILTERS,
                self._run_fs(name),
                name,
                required_channels=required_set,
            )
        self._build_consolidated_runs()
        self._run_modal_fits()
        self._preprocessed = True
        return self.run_data

    def _expand_split_runs(self) -> None:
        """Partition entries with ``split_by`` set into one run per group value.

        Operates on already-loaded data: each entry whose original spec carries
        ``split_by`` is replaced in-place inside ``self.runs`` by N children
        (one per unique non-null group value in first-occurrence order). The
        parent DataFrame is removed from ``self.run_data``; each child receives
        a row-filtered ``reset_index`` copy. Per-run metadata (units, sample
        rates, source path, required columns) is duplicated to each child.

        Spec forms accepted:
          - ``"colName"`` – partition by every unique non-null value
          - ``{"column": "colName", "values": [v1, v2, ...]}`` – keep only the
            listed values, one run per value
          - ``callable(df) -> Series`` – custom group keys (same length as df)

        Column names are resolved case-insensitively and tolerate the parquet
        ``_nRun`` / ``nRun`` underscore alias. ``split_by`` cannot be combined
        with ``consolidate`` / ``consolidate_by`` on the same entry.
        """
        if not getattr(self, "runs", None):
            return
        if not any(r.get("split_by") for r in self.runs):
            return
        try:
            from .plot_runtime import (
                _FOLDER_RUN_COLOR_PALETTE,
                _TYPE_COLORMAPS,
                _interpolate_two_colors,
                _shades_from_cmap,
            )
        except Exception:
            _shades_from_cmap = None
            _interpolate_two_colors = None
            _FOLDER_RUN_COLOR_PALETTE = (
                "#FF8000",
                "#2000BF",
                "#D70000",
                "#008CFF",
                "#00CC88",
                "#CC0066",
                "#FFD700",
                "#4C00BF",
            )
            _TYPE_COLORMAPS = {}
        new_runs: list = []
        for run in self.runs:
            spec = run.get("split_by")
            if not spec:
                new_runs.append(run)
                continue
            base_name = run["name"]
            if run.get("consolidate") or run.get("consolidate_by") or "_consolidate_sources" in run:
                raise ValueError(
                    f"Run '{base_name}': split_by cannot be combined with "
                    f"consolidate / consolidate_by on the same entry."
                )
            base_lower = base_name.lower()
            df = self.run_data.get(base_lower)
            if df is None or df.empty:
                log.warning(
                    "Run '%s': split_by requested but no loaded data; leaving run intact.",
                    base_name,
                )
                new_runs.append(run)
                continue
            try:
                partitions = self._partition_for_split_by(spec, df, base_name)
            except (KeyError, ValueError, TypeError) as exc:
                log.error(
                    "Run '%s': split_by failed (%s). Leaving run intact.",
                    base_name,
                    exc,
                )
                new_runs.append(run)
                continue
            if not partitions:
                log.warning(
                    "Run '%s': split_by produced no groups; leaving run intact.",
                    base_name,
                )
                new_runs.append(run)
                continue
            if len(partitions) == 1:
                log.info(
                    "Run '%s': split_by produced a single group %r; leaving run intact (no split).",
                    base_name,
                    partitions[0][0],
                )
                new_runs.append(run)
                continue
            n = len(partitions)
            colors_list = run.get("colors")
            color_range = run.get("color_range")
            run_type = run.get("type")
            if isinstance(colors_list, (list, tuple)) and colors_list:
                child_colors = [colors_list[i % len(colors_list)] for i in range(n)]
            elif (
                _interpolate_two_colors is not None and isinstance(color_range, (list, tuple)) and len(color_range) == 2
            ):
                # User-supplied gradient endpoints, HSV-interpolated.
                child_colors = _interpolate_two_colors(
                    color_range[0],
                    color_range[1],
                    n,
                )
            elif _shades_from_cmap is not None and run_type in _TYPE_COLORMAPS:
                # Use the saturated half of the type colormap (low=0.55
                # avoids near-white tints).
                child_colors = _shades_from_cmap(
                    _TYPE_COLORMAPS[run_type],
                    n,
                    low=0.55,
                    high=1.0,
                )
            else:
                child_colors = [_FOLDER_RUN_COLOR_PALETTE[i % len(_FOLDER_RUN_COLOR_PALETTE)] for i in range(n)]
            units = self.run_units.get(base_lower)
            fp = self.run_filepaths.get(base_lower)
            req = self.run_required_cols.get(base_lower)
            child_names: list[str] = []
            for i, (key, sub_df) in enumerate(partitions):
                suffix = _format_split_key(key)
                child_name = f"{base_name}_{suffix}"
                child = {k: v for k, v in run.items() if k != "split_by"}
                child["name"] = child_name
                child["color"] = child_colors[i]
                child["_split_parent"] = base_name
                child["_split_key"] = key
                new_runs.append(child)
                child_lower = child_name.lower()
                self.run_data[child_lower] = sub_df.reset_index(drop=True)
                if units is not None:
                    self.run_units[child_lower] = dict(units) if isinstance(units, dict) else units
                if fp is not None:
                    self.run_filepaths[child_lower] = fp
                if req is not None:
                    self.run_required_cols[child_lower] = (
                        set(req) if isinstance(req, (set, frozenset, list, tuple)) else req
                    )
                child_names.append(child_name)
            self.run_data.pop(base_lower, None)
            self.run_units.pop(base_lower, None)
            self.run_filepaths.pop(base_lower, None)
            self.run_required_cols.pop(base_lower, None)
            log.info(
                "Run '%s': split_by produced %d sub-runs (%d rows total): %s",
                base_name,
                n,
                sum(len(sub) for _, sub in partitions),
                ", ".join(child_names),
            )
        self.runs = new_runs

    def _partition_for_split_by(self, spec, df: pd.DataFrame, run_label: str) -> list[tuple]:
        """Compute (group_key, sub_df) partitions for a split_by spec.

        Preserves first-occurrence order of group keys, skips NaN keys, and
        raises KeyError if a referenced column is missing.
        """
        if callable(spec):
            keys = spec(df)
            if not isinstance(keys, pd.Series):
                keys = pd.Series(keys)
            if len(keys) != len(df):
                raise ValueError(f"split_by callable returned a series of length {len(keys)}, expected {len(df)}.")
            keys = keys.reset_index(drop=True)
        else:
            column: str | None
            filter_values = None
            if isinstance(spec, str):
                column = spec
            elif isinstance(spec, dict):
                column = spec.get("column")
                filter_values = spec.get("values")
                if column is None:
                    raise ValueError("split_by dict must include a 'column' key.")
            else:
                raise ValueError(
                    f"unsupported split_by={spec!r}; expected column name (str), dict with 'column', or callable."
                )
            resolved = _find_split_column(df, column)
            if resolved is None:
                raise KeyError(
                    f"split_by column {column!r} not found in DataFrame "
                    f"(have: {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''})."
                )
            keys = df[resolved].reset_index(drop=True)
            if filter_values is not None:
                try:
                    wanted = set(filter_values)
                except TypeError:
                    wanted = set(list(filter_values))
                keys = keys.where(keys.isin(wanted))
        df_indexed = df.reset_index(drop=True)
        order: list = []
        seen: set = set()
        for v in keys.tolist():
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            # Use a stable hashable proxy for the seen-set when possible.
            try:
                key_hash = v
                if key_hash in seen:
                    continue
                seen.add(key_hash)
            except TypeError:
                key_hash = repr(v)
                if key_hash in seen:
                    continue
                seen.add(key_hash)
            order.append(v)
        partitions: list[tuple] = []
        for v in order:
            mask = keys == v
            if hasattr(mask, "fillna"):
                mask = mask.fillna(False)
            sub = df_indexed[mask.to_numpy()]
            if not sub.empty:
                partitions.append((v, sub))
        return partitions

    def _build_consolidated_runs(self) -> None:
        """Construct synthetic 'consolidated' runs by concatenating sources.

        Sources are appended end-to-end with a one-second NaN gap between
        segments to prevent Welch from seeing false continuity across joins.
        Time-like columns (tLap, sLap) are reset to start at zero at the head
        of each segment and offset so the consolidated run has a monotonic
        time axis; this keeps PSD/waveform plots usable.
        """
        consolidated = [r for r in self.runs if "_consolidate_sources" in r]
        if not consolidated:
            return
        keep_runs = []
        drop_source_names: set[str] = set()
        for run in consolidated:
            cname = run["name"].lower()
            src_names = run.get("_consolidate_sources") or []
            src_dfs = [self.run_data.get(s) for s in src_names if s in self.run_data]
            src_dfs = [df for df in src_dfs if df is not None and not df.empty]
            if not src_dfs:
                log.warning(
                    "Consolidated run '%s': no source data available; skipping.",
                    cname,
                )
                continue
            # detect sample rate from first source (they should match after resample)
            first_src = next(s for s in src_names if s in self.run_data)
            sr_pair = self.run_sample_rates.get(first_src)
            fs = sr_pair[0] if sr_pair else self.FILTER_SAMPLE_RATE
            gap_samples = max(1, int(round(fs)))
            # NaN-row template: upcast integer columns so they accept NaN
            gap_template = src_dfs[0].iloc[0:1].copy()
            for col in gap_template.columns:
                if pd.api.types.is_integer_dtype(gap_template[col]):
                    gap_template[col] = gap_template[col].astype("float64")
            gap_template.loc[:, :] = np.nan
            gap_rows = pd.concat([gap_template] * gap_samples, ignore_index=True)
            # rebase time columns per segment, accumulate offsets
            segments = []
            t_offset = 0.0
            s_offset = 0.0
            for i, df in enumerate(src_dfs):
                seg = df.copy().reset_index(drop=True)
                if "tLap" in seg.columns:
                    t_col = pd.to_numeric(seg["tLap"], errors="coerce")
                    t_min = float(t_col.min()) if t_col.notna().any() else 0.0
                    seg["tLap"] = t_col - t_min + t_offset
                    t_offset = float(seg["tLap"].max()) + gap_samples / fs
                if "sLap" in seg.columns:
                    s_col = pd.to_numeric(seg["sLap"], errors="coerce")
                    s_min = float(s_col.min()) if s_col.notna().any() else 0.0
                    seg["sLap"] = s_col - s_min + s_offset
                    s_offset = float(seg["sLap"].max()) + 1.0
                segments.append(seg)
                if i < len(src_dfs) - 1:
                    segments.append(gap_rows.copy())
            merged = pd.concat(segments, ignore_index=True)
            self.run_data[cname] = merged
            self.run_units[cname] = self.run_units.get(first_src, {})
            self.run_sample_rates[cname] = (fs, f"inherited from {first_src}")
            log.info(
                "Consolidated '%s' built from %d source(s) -> %d rows (%.1f s @ %.0f Hz).",
                cname,
                len(src_dfs),
                len(merged),
                len(merged) / max(fs, 1.0),
                fs,
            )
            if run.get("_consolidate_drop_sources"):
                drop_source_names.update(s for s in src_names)
        if drop_source_names:
            for s in drop_source_names:
                self.run_data.pop(s, None)
                self.run_units.pop(s, None)
                self.run_sample_rates.pop(s, None)
                self.run_filepaths.pop(s, None)
            keep_runs = [r for r in self.runs if r["name"].lower() not in drop_source_names]
            self.runs = keep_runs
            log.info(
                "Consolidation 'only' mode: removed %d source run(s) from active set.",
                len(drop_source_names),
            )

    def _run_modal_fits(self) -> None:
        """Run vibrations Lorentz/body4dof fit per run and inject constants.

        Skips silently when self.VIBRATIONS_FIT is None. For each run that
        has FPushrod* (or xDamperPot*) channels available the fit is executed
        and the resulting modal parameters are broadcast as constant columns
        named `modal_<mode>_f0`, `modal_<mode>_zeta`, `modal_<mode>_f0_sigma`,
        `modal_<mode>_zeta_sigma` so existing BarPlot/ScatterPlot generators
        can consume them.
        """
        cfg = self.VIBRATIONS_FIT
        if not cfg:
            return
        try:
            from . import vibrations_io as _vib
        except Exception as exc:
            log.warning("Modal fit skipped — failed to import engine.vibrations_io: %s", exc)
            return
        displacement_mode = bool(cfg.get("displacement_mode", False))
        fmin = float(cfg.get("fmin", 2.0))
        fmax = float(cfg.get("fmax", 13.0))
        nperseg = cfg.get("nperseg", "auto")
        method = cfg.get("method", "lorentzian_combined")
        expected_freqs = cfg.get("expected_freqs")
        bootstrap_ci = bool(cfg.get("bootstrap_ci", False))
        bootstrap_n = int(cfg.get("bootstrap_n", 400))
        bootstrap_seed = int(cfg.get("bootstrap_seed", 0))
        primary_channels = _vib.DISPLACEMENT_CHANNELS if displacement_mode else _vib.FORCE_CHANNELS
        for run in list(self.runs):
            name = run["name"].lower()
            df = self.run_data.get(name)
            if df is None or df.empty:
                continue
            missing = [c for c in primary_channels if c not in df.columns]
            if missing:
                if self.verbose:
                    log.debug(
                        "Modal fit skipped for '%s' — missing channel(s): %s",
                        name,
                        missing,
                    )
                continue
            sr_pair = self.run_sample_rates.get(name)
            fs = sr_pair[0] if sr_pair else self.FILTER_SAMPLE_RATE
            try:
                arr = np.stack(
                    [
                        pd.to_numeric(df[c], errors="coerce")
                        .interpolate(limit=int(fs))
                        .ffill()
                        .bfill()
                        .to_numpy(dtype=float)
                        for c in primary_channels
                    ]
                )
            except Exception as exc:
                log.warning("Modal fit '%s': channel extraction failed: %s", name, exc)
                continue
            source_type = run.get("type")
            try:
                result = _vib.run_fit_from_arrays(
                    arr,
                    fs=fs,
                    source_type=source_type,
                    fmin=fmin,
                    fmax=fmax,
                    nperseg=nperseg,
                    method=method,
                    expected_freqs=expected_freqs,
                    displacement_mode=displacement_mode,
                    label=run["name"],
                    output_dir=self.plots_dir.parent,
                    event=cfg.get("event", ""),
                    show_plots=bool(cfg.get("show_plots", False)),
                    output_dpi=self.output_dpi,
                    bootstrap_ci=bootstrap_ci,
                    bootstrap_n=bootstrap_n,
                    bootstrap_seed=bootstrap_seed,
                    min_averages_target=self.PSD_MIN_AVERAGES_TARGET,
                )
            except Exception as exc:
                log.warning("Modal fit failed for run '%s': %s", name, exc)
                continue
            self.modal_results[name] = result

            def _as_list(v):
                if v is None:
                    return []
                try:
                    return list(v)
                except TypeError:
                    return [v]

            mode_labels = _as_list(result.get("mode_labels"))
            fn = _as_list(result.get("fn"))
            zeta = _as_list(result.get("zeta"))
            sigma_fn_raw = result.get("sigma_fn")
            sigma_zeta_raw = result.get("sigma_zeta")
            sigma_fn = _as_list(sigma_fn_raw) if sigma_fn_raw is not None else None
            sigma_zeta = _as_list(sigma_zeta_raw) if sigma_zeta_raw is not None else None
            amp_front_raw = result.get("amp_front")
            amp_rear_raw = result.get("amp_rear")
            amp_front = _as_list(amp_front_raw) if amp_front_raw is not None else None
            amp_rear = _as_list(amp_rear_raw) if amp_rear_raw is not None else None
            sigma_amp_front_raw = result.get("sigma_amp_front")
            sigma_amp_rear_raw = result.get("sigma_amp_rear")
            sigma_amp_front = _as_list(sigma_amp_front_raw) if sigma_amp_front_raw is not None else None
            sigma_amp_rear = _as_list(sigma_amp_rear_raw) if sigma_amp_rear_raw is not None else None

            # Bootstrap CIs (Option B). When present, they replace the
            # symmetric Jacobian sigmas: the per-mode `_sigma` channels are
            # overwritten with half the CI width so downstream bar plots
            # still work, and new `_lo` / `_hi` channels are emitted so the
            # modal-evolution figure can draw asymmetric bands.
            def _ci_pair(name: str):
                pair = result.get(name)
                if pair is None:
                    return None, None
                try:
                    lo_arr, hi_arr = pair
                except Exception:  # noqa: BLE001
                    return None, None
                return _as_list(lo_arr), _as_list(hi_arr)

            fn_lo, fn_hi = _ci_pair("fn_ci")
            zeta_lo, zeta_hi = _ci_pair("zeta_ci")
            af_lo, af_hi = _ci_pair("amp_front_ci")
            ar_lo, ar_hi = _ci_pair("amp_rear_ci")

            def _from(lst, i, default=float("nan")):
                if lst is None or i >= len(lst):
                    return default
                return float(lst[i])

            n_rows = len(df)
            new_cols: dict[str, np.ndarray] = {}
            for i, mlabel in enumerate(mode_labels):
                key = str(mlabel).lower()
                f0 = float(fn[i]) if i < len(fn) else float("nan")
                z = float(zeta[i]) if i < len(zeta) else float("nan")
                sf = float(sigma_fn[i]) if sigma_fn is not None and i < len(sigma_fn) else float("nan")
                sz = float(sigma_zeta[i]) if sigma_zeta is not None and i < len(sigma_zeta) else float("nan")

                # Bootstrap CI values (NaN when disabled).
                f0_lo = _from(fn_lo, i)
                f0_hi = _from(fn_hi, i)
                z_lo = _from(zeta_lo, i)
                z_hi = _from(zeta_hi, i)
                af_lo_i = _from(af_lo, i)
                af_hi_i = _from(af_hi, i)
                ar_lo_i = _from(ar_lo, i)
                ar_hi_i = _from(ar_hi, i)

                # When CIs exist, override the symmetric sigmas with
                # half-width so bar plots reflect the honest uncertainty.
                if np.isfinite(f0_lo) and np.isfinite(f0_hi):
                    sf = 0.5 * (f0_hi - f0_lo)
                if np.isfinite(z_lo) and np.isfinite(z_hi):
                    sz = 0.5 * (z_hi - z_lo)

                af = float(amp_front[i]) if amp_front is not None and i < len(amp_front) else float("nan")
                ar = float(amp_rear[i]) if amp_rear is not None and i < len(amp_rear) else float("nan")
                saf = (
                    float(sigma_amp_front[i])
                    if sigma_amp_front is not None and i < len(sigma_amp_front)
                    else float("nan")
                )
                sar = (
                    float(sigma_amp_rear[i]) if sigma_amp_rear is not None and i < len(sigma_amp_rear) else float("nan")
                )
                if np.isfinite(af_lo_i) and np.isfinite(af_hi_i):
                    saf = 0.5 * (af_hi_i - af_lo_i)
                if np.isfinite(ar_lo_i) and np.isfinite(ar_hi_i):
                    sar = 0.5 * (ar_hi_i - ar_lo_i)

                new_cols[f"modal_{key}_f0"] = np.full(n_rows, f0, dtype=float)
                new_cols[f"modal_{key}_zeta"] = np.full(n_rows, z, dtype=float)
                new_cols[f"modal_{key}_f0_sigma"] = np.full(n_rows, sf, dtype=float)
                new_cols[f"modal_{key}_zeta_sigma"] = np.full(n_rows, sz, dtype=float)
                new_cols[f"modal_{key}_f0_lo"] = np.full(n_rows, f0_lo, dtype=float)
                new_cols[f"modal_{key}_f0_hi"] = np.full(n_rows, f0_hi, dtype=float)
                new_cols[f"modal_{key}_zeta_lo"] = np.full(n_rows, z_lo, dtype=float)
                new_cols[f"modal_{key}_zeta_hi"] = np.full(n_rows, z_hi, dtype=float)
                new_cols[f"modal_{key}_amp_front"] = np.full(n_rows, af, dtype=float)
                new_cols[f"modal_{key}_amp_rear"] = np.full(n_rows, ar, dtype=float)
                new_cols[f"modal_{key}_amp_front_sigma"] = np.full(n_rows, saf, dtype=float)
                new_cols[f"modal_{key}_amp_rear_sigma"] = np.full(n_rows, sar, dtype=float)
                new_cols[f"modal_{key}_amp_front_lo"] = np.full(n_rows, af_lo_i, dtype=float)
                new_cols[f"modal_{key}_amp_front_hi"] = np.full(n_rows, af_hi_i, dtype=float)
                new_cols[f"modal_{key}_amp_rear_lo"] = np.full(n_rows, ar_lo_i, dtype=float)
                new_cols[f"modal_{key}_amp_rear_hi"] = np.full(n_rows, ar_hi_i, dtype=float)
            df = pd.concat(
                [df, pd.DataFrame(new_cols, index=df.index)],
                axis=1,
                copy=False,
            )
            self.run_data[name] = df
            log.info(
                "Modal fit '%s' (%s): %s",
                name,
                method,
                ", ".join(f"{m}={float(fn[i]):.2f}Hz/{float(zeta[i]):.3f}" for i, m in enumerate(mode_labels)),
            )

    def _align_slap(self) -> None:
        if not self.runs:
            return
        track_length = self._detect_track_length()
        if track_length is not None:
            log.info(
                "sLap alignment: using official track length %.1f m",
                track_length,
            )
            alignment = compute_slap_alignment(self.runs, self.run_data, target_length=track_length)
        else:
            if len(self.runs) < 2:
                return
            log.info("sLap alignment: no track detected, using baseline-relative mode")
            alignment = compute_slap_alignment(self.runs, self.run_data)
        if not alignment:
            return
        for rn, scale in alignment.items():
            df = self.run_data.get(rn)
            if df is None or "sLap" not in df.columns:
                continue
            df["sLap"] = df["sLap"] * scale
            drift_est = (scale - 1.0) * float(df["sLap"].max() - df["sLap"].min())
            log.info(
                "sLap aligned '%s': scale=%.6f (drift correction ~%.1f m)",
                rn,
                scale,
                drift_est,
            )

    def _detect_track_length(self) -> float | None:
        try:
            from channel_config import TRACK_LENGTHS
        except ImportError:
            return None
        track_code = self._extract_track_code(getattr(self, "_root_folder", None))
        if track_code and track_code in TRACK_LENGTHS:
            return TRACK_LENGTHS[track_code]
        for run in self.runs:
            filename = run.get("file", "")
            code = self._extract_track_code_from_filename(filename)
            if code and code in TRACK_LENGTHS:
                return TRACK_LENGTHS[code]
        return None

    @staticmethod
    def _extract_track_code(path) -> str | None:
        if path is None:
            return None
        name = Path(path).name
        if len(name) >= 3:
            code = name[-3:].upper()
            if code.isalpha():
                return code
        return None

    @staticmethod
    def _extract_track_code_from_filename(filename: str) -> str | None:
        if not filename:
            return None
        event = filename.split("_")[0] if "_" in filename else filename
        if len(event) >= 3:
            code = event[-3:].upper()
            if code.isalpha():
                return code
        return None

    def reference_run_name(self) -> str | None:
        loaded = [r["name"].lower() for r in self.runs if r["name"].lower() in self.run_data]
        if not loaded:
            return None
        for run in self.runs:
            if run.get("reference") and run["name"].lower() in self.run_data:
                return run["name"].lower()
        return loaded[0]

    def _compute_tdiff_channel(self) -> None:
        loaded = [r["name"].lower() for r in self.runs if r["name"].lower() in self.run_data]
        if len(loaded) < 2:
            return

        def _time_series(df):
            for col in ("tLap", "tLap_Calc", "Time", "time"):
                if col in df.columns:
                    return df[col].to_numpy(dtype=float)
            return None

        ref_name = self.reference_run_name()
        if ref_name is None:
            return
        ref_df = self.run_data[ref_name]
        if "sLap" not in ref_df.columns:
            log.debug("tDiff: reference run '%s' has no sLap; skipping.", ref_name)
            return
        ref_t = _time_series(ref_df)
        if ref_t is None:
            log.debug("tDiff: reference run '%s' has no time channel; skipping.", ref_name)
            return
        ref_s = ref_df["sLap"].to_numpy(dtype=float)
        finite_ref = np.isfinite(ref_s) & np.isfinite(ref_t)
        if finite_ref.sum() < 2:
            return
        ref_s_f, ref_t_f = ref_s[finite_ref], ref_t[finite_ref]
        order = np.argsort(ref_s_f)
        ref_s_sorted, ref_t_sorted = ref_s_f[order], ref_t_f[order]
        for name in loaded:
            df = self.run_data[name]
            if name == ref_name:
                df["tDiff"] = 0.0
                continue
            if "sLap" not in df.columns:
                continue
            t = _time_series(df)
            if t is None:
                continue
            s = df["sLap"].to_numpy(dtype=float)
            ref_t_on_s = np.interp(s, ref_s_sorted, ref_t_sorted, left=np.nan, right=np.nan)
            df["tDiff"] = t - ref_t_on_s
            log.debug("tDiff computed for '%s' (vs '%s').", name, ref_name)

    def _colorize_legend_labels(self, legend):
        if legend is None:
            return
        for text, handle in zip(legend.get_texts(), legend.legend_handles):
            color = None
            if hasattr(handle, "get_color") and not isinstance(handle, Patch):
                color = handle.get_color()
                if isinstance(color, (list, tuple, np.ndarray)):
                    color = color[0] if len(color) and isinstance(color[0], (list, tuple, np.ndarray)) else color
            elif isinstance(handle, Patch):
                fc = handle.get_facecolor()
                if isinstance(fc, (list, tuple, np.ndarray)) and len(fc) >= 3:
                    color = fc[:3]
            if color is None and hasattr(handle, "get_facecolor"):
                fc = handle.get_facecolor()
                if isinstance(fc, np.ndarray) and fc.size > 0:
                    color = fc[0]
                elif isinstance(fc, (list, tuple)) and len(fc) > 0:
                    color = fc[0] if isinstance(fc[0], (list, tuple, np.ndarray)) else fc
            if color is not None:
                text.set_color(color)

    def _legend_corner(self, legend):
        if legend is None:
            return None
        loc_map = {
            "upper right": ("right", "top"),
            "upper left": ("left", "top"),
            "lower right": ("right", "bottom"),
            "lower left": ("left", "bottom"),
            "upper center": ("center", "top"),
            "lower center": ("center", "bottom"),
        }
        try:
            loc_code = legend._loc
            for name, code in legend.codes.items():
                if code == loc_code and name in loc_map:
                    return loc_map[name]
        except Exception:
            pass
        return self._legend_corner_from_bbox(legend)

    def _legend_corner_from_bbox(self, legend):
        bbox = self._legend_axes_bbox(legend)
        if bbox is None:
            return None
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        halign = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        valign = "bottom" if cy < 1 / 3 else ("top" if cy > 2 / 3 else "center")
        corner = (halign, valign)
        if corner in self._INFO_CORNER_XY:
            return corner
        return min(
            self._INFO_CORNER_XY.keys(),
            key=lambda c: (self._INFO_CORNER_XY[c][0] - cx) ** 2 + (self._INFO_CORNER_XY[c][1] - cy) ** 2,
        )

    def _legend_axes_bbox(self, legend):
        if legend is None:
            return None
        ax = getattr(legend, "axes", None)
        if ax is None:
            return None
        try:
            fig = ax.figure
            renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else None
            if renderer is None:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
            win_bbox = legend.get_window_extent(renderer)
            inv = ax.transAxes.inverted()
            (x0, y0) = inv.transform((win_bbox.x0, win_bbox.y0))
            (x1, y1) = inv.transform((win_bbox.x1, win_bbox.y1))
            return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        except Exception:
            return None

    _INFO_CORNER_XY = {
        ("left", "top"): (0.02, 0.98),
        ("right", "top"): (0.98, 0.98),
        ("left", "bottom"): (0.02, 0.02),
        ("right", "bottom"): (0.98, 0.02),
        ("center", "top"): (0.50, 0.98),
        ("center", "bottom"): (0.50, 0.02),
        ("left", "center"): (0.02, 0.50),
        ("right", "center"): (0.98, 0.50),
    }
    _CORNER_TO_LOC = {
        ("left", "top"): "upper left",
        ("right", "top"): "upper right",
        ("left", "bottom"): "lower left",
        ("right", "bottom"): "lower right",
        ("center", "top"): "upper center",
        ("center", "bottom"): "lower center",
        ("left", "center"): "center left",
        ("right", "center"): "center right",
    }
    _LOC_TO_CORNER = {
        "upper right": ("right", "top"),
        "upper left": ("left", "top"),
        "lower right": ("right", "bottom"),
        "lower left": ("left", "bottom"),
        "upper center": ("center", "top"),
        "lower center": ("center", "bottom"),
        "center left": ("left", "center"),
        "center right": ("right", "center"),
    }

    def _sample_ax_data(self, ax):
        xs, ys = [], []
        for coll in ax.collections:
            try:
                offsets = coll.get_offsets()
                step = max(1, len(offsets) // 500)
                xs.extend(offsets[::step, 0])
                ys.extend(offsets[::step, 1])
            except Exception:
                pass
        for line in ax.lines:
            xd, yd = line.get_xdata(), line.get_ydata()
            step = max(1, len(xd) // 500)
            xs.extend(xd[::step])
            ys.extend(yd[::step])
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def _count_points_in_region(self, xs, ys, x0, x1, y0, y1, halign, valign, w_frac=0.18, h_frac=0.20):
        if xs.size == 0:
            return 0
        w = (x1 - x0) * w_frac
        h = (y1 - y0) * h_frac
        xa = 0.97 if halign == "right" else (0.50 if halign == "center" else 0.03)
        ya = 0.97 if valign == "top" else (0.50 if valign == "center" else 0.03)
        x_abs = x0 + xa * (x1 - x0)
        x_min = x_abs if halign == "left" else (x_abs - w if halign == "right" else x_abs - w / 2)
        x_max = x_min + w
        y_abs = y0 + ya * (y1 - y0)
        if valign == "top":
            y_min, y_max = y_abs - h, y_abs
        elif valign == "bottom":
            y_min, y_max = y_abs, y_abs + h
        else:
            y_min, y_max = y_abs - h / 2, y_abs + h / 2
        return int(((xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)).sum())

    def _rank_info_corners(self, ax, w_frac=0.22, h_frac=0.28):
        xs, ys = self._sample_ax_data(ax)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        corners = list(self._INFO_CORNER_XY.keys())
        if xs.size == 0:
            return corners
        return sorted(
            corners,
            key=lambda c: self._count_points_in_region(xs, ys, x0, x1, y0, y1, c[0], c[1], w_frac, h_frac),
        )

    def _add_standard_legend(
        self, ax, handles=None, labels=None, loc="best", bbox_to_anchor=None, ncol=1, avoid_corner=None
    ):
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None
        if avoid_corner is not None and bbox_to_anchor is None:
            ranked = [c for c in self._rank_info_corners(ax) if c != avoid_corner]
            corner = ranked[0] if ranked else None
            loc = self._CORNER_TO_LOC.get(corner, "best") if corner else "best"
        legend_kwargs = dict(
            fancybox=True,
            framealpha=0.92,
            edgecolor="#3C3C3C",
            borderpad=0.55,
            handlelength=1.8,
            ncol=ncol,
            prop={"family": self.PLOT_FONT["family"], "weight": "bold", "size": self.PLOT_FONT["legend_size"]},
        )
        corner = self._LOC_TO_CORNER.get(loc) if isinstance(loc, str) else None
        if corner is not None and bbox_to_anchor is None:
            legend_kwargs["bbox_to_anchor"] = self._INFO_CORNER_XY[corner]
            legend_kwargs["bbox_transform"] = ax.transAxes
            legend_kwargs["borderaxespad"] = 0
        else:
            legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
        legend = ax.legend(handles, labels, loc=loc, **legend_kwargs)
        legend.get_frame().set_linewidth(1.4)
        legend.set_zorder(10)
        self._colorize_legend_labels(legend)
        return legend

    def _add_waveform_figure_legend(self, fig, handles, labels, position="top"):
        if not handles:
            return None
        if position == "right":
            legend = fig.legend(
                handles,
                labels,
                loc="center right",
                bbox_to_anchor=(1.0, 0.5),
                ncol=1,
                fancybox=True,
                framealpha=0.92,
                edgecolor="#3C3C3C",
                borderpad=0.4,
                handlelength=1.8,
                prop={"family": self.PLOT_FONT["family"], "weight": "bold", "size": self.PLOT_FONT["legend_size"]},
            )
        else:
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=max(1, min(len(handles), 5)),
                fancybox=True,
                framealpha=0.92,
                edgecolor="#3C3C3C",
                borderpad=0.3,
                handlelength=1.8,
                prop={"family": self.PLOT_FONT["family"], "weight": "bold", "size": self.PLOT_FONT["legend_size"]},
            )
        legend.get_frame().set_linewidth(1.4)
        legend.set_zorder(10)
        self._colorize_legend_labels(legend)
        return legend

    def _display_gate_info(self, ax, text, legend=None, trend_anchor=None):
        occupied = set()
        if trend_anchor is not None:
            _, trend_halign, trend_valign, _ = trend_anchor
            occupied.add((trend_halign, trend_valign))
        legend_corner = self._legend_corner(legend)
        if legend_corner is not None:
            occupied.add(legend_corner)
        ranked = self._rank_info_corners(ax)
        free = [c for c in ranked if c not in occupied]
        halign, valign = free[0] if free else ranked[0]
        x_anchor, y_anchor = self._INFO_CORNER_XY[(halign, valign)]
        ax.text(
            x_anchor,
            y_anchor,
            text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment=valign,
            horizontalalignment=halign,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white",
                alpha=0.92,
                edgecolor="#3C3C3C",
                linewidth=1.4,
            ),
            color="#1A1A1A",
            fontweight="bold",
            family=self.PLOT_FONT["family"],
        )

    def plot_data(self, plot_types: list[str] | None = None, plot_names: list[str] | None = None) -> None:
        self._ensure_preprocessed()
        self._psd_cache = {}
        self._gated_data_cache = {}
        self._outlier_log = []
        from collections import Counter as _Counter

        class _AggregatingHandler(logging.Handler):
            def __init__(self):
                super().__init__(level=logging.WARNING)
                self.counts = _Counter()

            def emit(self, record):
                try:
                    key = record.getMessage()
                except Exception:
                    key = record.msg
                self.counts[key] += 1

        agg = _AggregatingHandler()
        log.addHandler(agg)
        all_generators = [
            ("waveform", self.generate_waveform_plots),
            ("scatter", self.generate_scatter_plots),
            ("psd", self.generate_psd_plots),
            ("histogram", self.generate_histogram_plots),
            ("bar", self.generate_bar_plots),
            ("box", self.generate_box_plots),
            ("heatmap", self.generate_heatmap_plots),
            ("scatter3d", self.generate_scatter3d_plots),
        ]
        if plot_types is not None:
            requested = {t.lower() for t in plot_types}
            generators = [(name, fn) for name, fn in all_generators if name in requested]
            if not generators:
                log.warning("No plot types matched from: %r", plot_types)
                return
        else:
            generators = all_generators
        if plot_names is not None:
            names_lower = {n.lower() for n in plot_names}
            original_defs = self.PLOT_DEFINITIONS
            self.PLOT_DEFINITIONS = tuple(
                [pd for pd in group if getattr(pd, "name", "").lower() in names_lower]
                for group in self.PLOT_DEFINITIONS
            )
        try:
            for _, fn in generators:
                fn()
        finally:
            if plot_names is not None:
                self.PLOT_DEFINITIONS = original_defs
            log.removeHandler(agg)
            repeated = [(msg, n) for msg, n in agg.counts.most_common() if n > 1]
            if repeated:
                top = repeated[:10]
                log.info(
                    "Plot-time warning summary: %d unique repeated message(s); showing top %d.",
                    len(repeated),
                    len(top),
                )
                for msg, n in top:
                    log.info("  [x%d] %s", n, msg)
        log.info("All plots saved to: %s", self.plots_dir)
