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
from .plotting import helpers as _helpers
from .plotting import loaders as _loaders
from .plotting.context import PlotContext
from .plotting.corner_layout import (  # noqa: F401  (back-compat re-export)
    _CORNER_TO_LOC,
    _INFO_CORNER_XY,
    _LOC_TO_CORNER,
    _legend_corner_from_bbox,
)
from .plotting.slap_alignment import (  # noqa: F401  (back-compat re-export)
    _prepare_slap_vcar_series,
    _score_slap_alignment,
)


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
        # Runtime state (per-run data + caches) lives on a single PlotContext
        # so it can be threaded through extracted helpers without a full
        # DataPlotter instance. The eight `@property` shims below let existing
        # `self.run_data` / `self._psd_cache` reads and writes flow through
        # unchanged.
        self.ctx = PlotContext()
        self.modal_results = {}
        self.VIBRATIONS_FIT = vibrations_fit
        self.PSD_MIN_AVERAGES_TARGET = int(psd_min_averages_target)
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

    # ------------------------------------------------------------------
    # PlotContext shim properties
    # ------------------------------------------------------------------
    # These forward reads/writes of the eight per-run state fields to
    # ``self.ctx``. Kept so mixin/plugin code that still says
    # ``self.run_data[...]`` or ``self._psd_cache.clear()`` keeps working
    # verbatim during the incremental extraction. Do NOT remove — the goal
    # is that ``plotter.run_data`` and ``plotter.ctx.run_data`` are both
    # valid entry points.
    @property
    def run_data(self):
        return self.ctx.run_data

    @run_data.setter
    def run_data(self, value):
        self.ctx.run_data = value

    @property
    def run_units(self):
        return self.ctx.run_units

    @run_units.setter
    def run_units(self, value):
        self.ctx.run_units = value

    @property
    def run_filepaths(self):
        return self.ctx.run_filepaths

    @run_filepaths.setter
    def run_filepaths(self, value):
        self.ctx.run_filepaths = value

    @property
    def run_required_cols(self):
        return self.ctx.run_required_cols

    @run_required_cols.setter
    def run_required_cols(self, value):
        self.ctx.run_required_cols = value

    @property
    def run_sample_rates(self):
        return self.ctx.run_sample_rates

    @run_sample_rates.setter
    def run_sample_rates(self, value):
        self.ctx.run_sample_rates = value

    @property
    def _psd_cache(self):
        return self.ctx.psd_cache

    @_psd_cache.setter
    def _psd_cache(self, value):
        self.ctx.psd_cache = value

    @property
    def _gated_data_cache(self):
        return self.ctx.gated_data_cache

    @_gated_data_cache.setter
    def _gated_data_cache(self, value):
        self.ctx.gated_data_cache = value

    @property
    def _outlier_log(self):
        return self.ctx.outlier_log

    @_outlier_log.setter
    def _outlier_log(self, value):
        self.ctx.outlier_log = value

    @property
    def _lorentz_fit_records(self):
        return self.ctx.lorentz_fit_records

    @_lorentz_fit_records.setter
    def _lorentz_fit_records(self, value):
        self.ctx.lorentz_fit_records = value

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
        return _helpers._apply_grid(self, ax, which=which, axis=axis)

    def _run_fs(self, run_name: str) -> float:
        return _helpers._run_fs(self, run_name)

    @staticmethod
    def _apply_2d_axis_limits(ax, axis_limits, *, log_scale_y=False, y_floor=1e-4):
        return _helpers._apply_2d_axis_limits(ax, axis_limits, log_scale_y=log_scale_y, y_floor=y_floor)

    @staticmethod
    def _draw_horizontal_reference_lines(ax, refs, *, label=True):
        return _helpers._draw_horizontal_reference_lines(ax, refs, label=label)

    @staticmethod
    def _draw_static_markers(axes, markers, *, label_y=1.01, x_clip=True):
        return _helpers._draw_static_markers(axes, markers, label_y=label_y, x_clip=x_clip)

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
                # Self-referential guard pattern (e.g. hRideF calc uses
                # df["hRideF"] if present, else falls back to corner-avg):
                # the channel name appearing in its own deps means the calc
                # can consume the native source column, so it must be kept
                # in the projected source-column set.
                if channel in deps:
                    resolved_channels.add(channel)
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
        return _loaders._normalize_parquet_column_aliases(df)

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
        return _loaders._resolve_required_parquet_columns(
            schema_cols,
            columns_to_load,
            nrun=nrun,
            nlap=nlap,
            alias_cache=self._parquet_alias_cache,
        )

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
        return _loaders._load_parquet_with_fallback(
            file_path,
            columns_to_load=columns_to_load,
            parquet_nrun=parquet_nrun,
            parquet_nlap=parquet_nlap,
            run_name=run_name,
            available_engines=self._available_parquet_engines(),
            get_schema_columns=self._get_parquet_schema_columns,
            apply_rank_value_filter=self._apply_parquet_rank_value_filter,
            alias_cache=self._parquet_alias_cache,
            verbose=self.verbose,
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
        return _loaders._load_run_data(
            file_path,
            use_python_engine=use_python_engine,
            columns_to_load=columns_to_load,
            parquet_nrun=parquet_nrun,
            parquet_nlap=parquet_nlap,
            run_name=run_name,
            load_parquet=self._load_parquet_with_fallback,
            make_unique=make_unique,
        )

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
        return _helpers._ensure_preprocessed(self)

    def _cached_psd_with_segments(self, run_name, channel, nperseg, gate_spec=None):
        return _helpers._cached_psd_with_segments(self, run_name, channel, nperseg, gate_spec=gate_spec)

    def _suggest_similar_channels(self, run_name, missing_channel, max_suggestions=5):
        df = self.run_data.get(run_name)
        if df is None:
            return []
        return datafunctions.suggest_similar_channels(missing_channel, list(df.columns), max_results=max_suggestions)

    def _format_missing_channel_hint(self, run_name, missing_channel):
        return _helpers._format_missing_channel_hint(self, run_name, missing_channel)

    def _get_plot_group(self, index):
        return _helpers._get_plot_group(self, index)

    def _sanitize_plot_filename(self, prefix, plot_name, suffix=""):
        return _helpers._sanitize_plot_filename(self, prefix, plot_name, suffix=suffix)

    def _resolve_plot_figsize(self, filename, default_size, *, min_height=None):
        return _helpers._resolve_plot_figsize(self, filename, default_size, min_height=min_height)

    def _add_axis_edge_padding(self, ax, x_pad_ratio=0.02, y_pad_ratio=0.03):
        return _helpers._add_axis_edge_padding(ax, x_pad_ratio=x_pad_ratio, y_pad_ratio=y_pad_ratio)

    def _get_filtered_run_dataframe(self, run_name, gate_spec=None):
        return _helpers._get_filtered_run_dataframe(self, run_name, gate_spec=gate_spec)

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
        return _legend_corner_from_bbox(self._legend_axes_bbox(legend))

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

    # Class-attribute rebindings so `self._INFO_CORNER_XY` etc. still resolve
    # from the plot_generators_* mixins after the constants moved to
    # engine.plotting.corner_layout. RHS resolves in enclosing module scope.
    _INFO_CORNER_XY = _INFO_CORNER_XY
    _CORNER_TO_LOC = _CORNER_TO_LOC
    _LOC_TO_CORNER = _LOC_TO_CORNER

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
        return _helpers._count_points_in_region(xs, ys, x0, x1, y0, y1, halign, valign, w_frac=w_frac, h_frac=h_frac)

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
        return _helpers._add_standard_legend(
            self, ax, handles=handles, labels=labels, loc=loc,
            bbox_to_anchor=bbox_to_anchor, ncol=ncol, avoid_corner=avoid_corner,
        )

    def _add_waveform_figure_legend(self, fig, handles, labels, position="top"):
        return _helpers._add_waveform_figure_legend(self, fig, handles, labels, position=position)

    def _display_gate_info(self, ax, text, legend=None, trend_anchor=None):
        return _helpers._display_gate_info(self, ax, text, legend=legend, trend_anchor=trend_anchor)

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
