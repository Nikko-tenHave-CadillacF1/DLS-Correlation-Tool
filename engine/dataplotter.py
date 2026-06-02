"""Data loading, preprocessing, and plotting pipeline for correlation reports.

The DataPlotter class handles:
  - Loading CAR (.txt), DLS/OC/DIL (.parquet) files
  - Channel mapping, transforms, calculated channels, and filtering
  - Resampling to uniform rate and sLap alignment
  - Dispatching to plot generators by type
  - Data quality report generation

Split into focused generator modules (mixed in via multiple inheritance):
  plot_generators_waveform.py  — WaveformMixin
  plot_generators_scatter.py   — ScatterMixin
  plot_generators_misc.py      — PsdHistMixin, HeatmapMixin
  plot_generators_bar_box.py   — BarBoxMixin
"""

from __future__ import annotations

import matplotlib
# Force the non-interactive Agg backend so the tool can run from terminals,
# remote shells, and CI agents that lack a display. Must come before any
# pyplot import. (#16)
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
import importlib.util
from typing import Optional
from . import datafunctions
from collections import Counter, deque
from matplotlib.patches import Patch

from .plot_definitions import (
    PLOT_TYPE_ORDER,
    Marker,
    WaveformPlot,
    ScatterPlot,
    PsdPlot,
    HistogramPlot,
    BarPlot,
    BoxPlot,
    HeatmapPlot,
)
from .plot_generators_waveform import WaveformMixin
from .plot_generators_scatter import ScatterMixin
from .plot_generators_misc import PsdHistMixin, HeatmapMixin
from .plot_generators_bar_box import BarBoxMixin
from .logger import log
import logging




# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def make_unique(names):
    """Make column names unique by appending suffixes to duplicates."""
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


_CALC_DEP_CACHE = {}


def _extract_calculated_dependencies(func):
    """Return source column names referenced by a calculated-channel lambda.

    Results are memoized by ``id(func)`` at module scope so repeated lookups
    across DataPlotter instances are O(1).
    """
    key = id(func)
    cached = _CALC_DEP_CACHE.get(key)
    if cached is not None:
        return cached

    # Explicit declarations via ``calc_channel`` decorator take priority (#5).
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
        _CALC_DEP_CACHE[key] = set()
        return _CALC_DEP_CACHE[key]

    matches = re.findall(r"df\['([^']+)'\]|df\[\"([^\"]+)\"\]", source)
    deps = {m[0] or m[1] for m in matches}
    _CALC_DEP_CACHE[key] = deps
    return deps


# ================================================================
# DATA QUALITY CHECKS
# ================================================================

def collect_referenced_channels(plot_definitions):
    """Collect channels referenced by configured plot definitions.

    Works on the new typed plot dataclasses (WaveformPlot, ScatterPlot, ...).
    """
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
                # Condition markers reference gate channels.
                for m in getattr(plot_def, "markers", None) or []:
                    if getattr(m, "condition", None) is not None:
                        referenced.update(
                            datafunctions.collect_gate_channels(m.condition)
                        )
            elif kind == "scatter":
                _add(plot_def.x_channel)
                _add(plot_def.y_channel)
                if isinstance(plot_def.best_fit, (list, tuple)):
                    referenced.update(
                        datafunctions.collect_multi_fit_condition_channels(plot_def.best_fit)
                    )
                if plot_def.gate is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
                if isinstance(plot_def.color_gate, (list, tuple)) and len(plot_def.color_gate) >= 3:
                    referenced.update(
                        datafunctions.collect_gate_channels(tuple(plot_def.color_gate[:3]))
                    )
            elif kind == "psd":
                _add(plot_def.channel)
                if getattr(plot_def, "gate", None) is not None:
                    referenced.update(datafunctions.collect_gate_channels(plot_def.gate))
            elif kind == "histogram":
                _add(plot_def.channel)
            elif kind == "bar":
                for ch, _agg in datafunctions.normalize_bar_metric_specs(plot_def.metrics):
                    _add(ch)
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
    return sorted(referenced)


def _prepare_slap_vcar_series(df):
    """Return cleaned (sLap, vCar) arrays for alignment diagnostics."""
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
    """Score one linear mapping transformed_s = oth_s*scale + offset."""
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
    """
    Estimate linear sLap alignment between baseline run and other runs.
    Uses transformed_sLap = sLap * scale + offset and vCar similarity scoring.
    """
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

        # Coarse seed grid (~25 evals) followed by Nelder-Mead refinement (#24).
        # Replaces the previous 1718-eval brute-force grid; converges to the
        # same optimum to ~1e-4 in scale and ~0.05 m in offset.
        best = None
        for scale in np.linspace(scale_guess - 0.01, scale_guess + 0.01, 5):
            for offset in np.linspace(offset_guess - 40, offset_guess + 40, 5):
                score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset)
                if score is None:
                    continue
                corr, mae, n = score
                key = (corr, -mae, n)
                if best is None or key > best["key"]:
                    best = {"scale": float(scale), "offset": float(offset),
                            "corr": corr, "mae": mae, "n": int(n), "key": key}

        if best is not None:
            try:
                from scipy.optimize import minimize

                def _obj(x):
                    score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, x[0], x[1])
                    if score is None:
                        return 1e6
                    corr, mae, _n = score
                    # Maximise corr (penalty), with mild MAE tie-break.
                    return -corr + 1e-3 * mae

                result = minimize(
                    _obj, x0=[best["scale"], best["offset"]],
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 200},
                )
                if result.success or result.nit > 0:
                    score = _score_slap_alignment(
                        ref_s, ref_v, oth_s, oth_v, float(result.x[0]), float(result.x[1])
                    )
                    if score is not None:
                        corr, mae, n = score
                        key = (corr, -mae, n)
                        if key > best["key"]:
                            best = {"scale": float(result.x[0]),
                                    "offset": float(result.x[1]),
                                    "corr": corr, "mae": mae,
                                    "n": int(n), "key": key}
            except Exception:
                pass

        if best is None:
            lines.append(f"{rn.upper()}: could not estimate sLap mapping")
            continue

        drift_end = (best["scale"] - 1.0) * ref_range
        lines.append(
            (
                f"{rn.upper()} vs {base_name.upper()}: "
                f"scale={best['scale']:.6f}, offset={best['offset']:+.2f} m, "
                f"end_drift_est={drift_end:+.2f} m, "
                f"vCar_corr={best['corr']:.4f}, vCar_mae={best['mae']:.2f} kph, "
                f"samples={best['n']}"
            )
        )

    return lines


# (Old top-level wrappers removed; callers should import directly from
# `engine.data_quality_report`. The wrappers had a broken non-relative import.)


# ---------------------------------------------------------------------------
# DataPlotter
# ---------------------------------------------------------------------------

class DataPlotter(WaveformMixin, ScatterMixin, PsdHistMixin, HeatmapMixin, BarBoxMixin):
    """Main class for loading, processing, and plotting multi-run data."""

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
    ):
        """Build a plotter instance and run the preprocessing pipeline."""
        if fig_size is None:
            fig_size = {"waveform": (15.5, 6.4), "scatter": (10, 8), "psd": (10, 8),
                        "histogram": (10, 8), "bar": (10, 6), "box": (10, 6)}

        root_folder = Path(root_folder)
        output_dir = Path(output_dir) if output_dir is not None else root_folder
        # Normalise runs in-place: ensure each has 'run_id' (lowercase name)
        # and an auto-assigned 'color' if missing (#15, #17).
        _AUTO_COLORS = (
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
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
        # When resample_rate is given, use it as the canonical filter design
        # rate too so that resampling output and filter cutoffs are aligned.
        if resample_rate is not None:
            try:
                rr = float(resample_rate)
            except (TypeError, ValueError):
                rr = 0.0
            self.RESAMPLE_RATE = rr
            if rr > 0:
                self.FILTER_SAMPLE_RATE = rr
        else:
            self.RESAMPLE_RATE = float(sample_rate) if sample_rate else 0.0

        self.SCATTER_DOT_SIZE = scatter_dot_size
        self.SCATTER_TRANSPARENCY = scatter_transparency
        self.SCATTER_MAX_POINTS = scatter_max_points

        # Accept both dict and legacy list format
        if isinstance(fig_size, dict):
            self.waveform_figsize = fig_size.get("waveform", (15.5, 6.4))
            self.scatter_FIGSIZE = fig_size.get("scatter", (10, 8))
            self.psd_FIGSIZE = fig_size.get("psd", (10, 8))
            self.histogram_FIGSIZE = fig_size.get("histogram", (10, 8))
            self.bar_FIGSIZE = fig_size.get("bar", (10, 6))
            self.boxplot_FIGSIZE = fig_size.get("box", fig_size.get("bar", (10, 6)))
        else:
            self.waveform_figsize = fig_size[0]
            self.scatter_FIGSIZE = fig_size[1]
            self.psd_FIGSIZE = fig_size[2]
            self.histogram_FIGSIZE = fig_size[3]
            self.bar_FIGSIZE = fig_size[4] if len(fig_size) > 4 else (10, 6)
            self.boxplot_FIGSIZE = fig_size[5] if len(fig_size) > 5 else self.bar_FIGSIZE
        self.plot_aspect_ratios = plot_aspect_ratios or {}
        self.BOX_PLOT_SETTINGS = box_plot_settings or {}

        # Internal state
        self.run_filepaths = {}
        self.run_data = {}
        self.run_units = {}
        self.run_required_cols = {}
        self.run_sample_rates = {}   # {run_name: (rate_hz, source_label)} (#17)
        self._gated_data_cache = {}
        self._psd_cache = {}         # {(run_name, channel, nperseg): (freq, power)} (#15)
        self._outlier_log = []       # populated by scatter robust fits (#18)
        self._reverse_mappings = {}
        self._parquet_alias_cache = {}
        self._loaded = False
        self._preprocessed = False

        if self.CHANNEL_MAPPINGS:
            for source_type, mapping in self.CHANNEL_MAPPINGS.items():
                if mapping:
                    self._reverse_mappings[source_type] = {
                        mapped: raw for raw, mapped in mapping.items()
                    }

        # Create plots directory
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        self.load_data(root_folder)
        self.preprocess_data()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    # Centralized style constants — used by all generators for consistency.
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
        """Apply a consistent font and baseline styling to all plots."""
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        preferred_font = (
            self.PLOT_FONT["family"]
            if self.PLOT_FONT["family"] in available_fonts
            else self.PLOT_FONT["fallback"][0]
        )

        # Soft dark grey replaces pure black for less harsh contrast in print/PDF.
        ink = "#1A1A1A"
        plt.rcParams.update({
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
        })

    def _apply_grid(self, ax, which="both", axis="both"):
        """Apply consistent grid styling to an axis.

        which: 'major', 'minor', or 'both' (default: both for report-quality plots).
        axis: 'both', 'x', or 'y'.
        """
        if which in ("major", "both"):
            ax.grid(True, which="major", axis=axis, **self.GRID_STYLE["major"])
        if which in ("minor", "both"):
            ax.grid(True, which="minor", axis=axis, **self.GRID_STYLE["minor"])
        ax.set_axisbelow(True)

    @staticmethod
    def _apply_2d_axis_limits(ax, axis_limits, *, log_scale_y=False, y_floor=1e-4):
        """Apply ((xmin,xmax),(ymin,ymax)) limits to ``ax``. Returns (has_x, has_y)."""
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
        """Draw flat-list horizontal reference lines on a 2-D plot.

        Used by scatter, PSD, histogram, bar, box, and any 2-D plot type with
        a ``reference_lines: list[float]`` field. WaveformPlot has its own
        per-row schema and does NOT use this helper.
        """
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
                    0.995, v, f" {v:g}",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom",
                    fontsize=8, fontweight="bold", color="#333333",
                )

    @staticmethod
    def _draw_static_markers(axes, markers, *, label_y=1.01, x_clip=True):
        """Draw static (x-valued) markers on one or more axes.

        ``axes`` may be a single Axes or an iterable. Condition-based markers
        (``Marker.condition is not None``) are skipped — those are waveform-only.
        """
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
                ax.axvline(m.x, color=color, linestyle=m.linestyle,
                           linewidth=1.2, alpha=0.7, zorder=2)
                if m.label and m.show_label:
                    ax.text(
                        m.x, label_y, m.label,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=color, linewidth=0.8, alpha=0.9),
                        zorder=12,
                    )

    # ------------------------------------------------------------------
    # Required columns resolution
    # ------------------------------------------------------------------

    def _get_required_source_columns(self, source_type):
        """Determine which raw source columns to load for a given run type."""
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
                                required_channels.update(
                                    datafunctions.collect_gate_channels(m.condition)
                                )
                    elif kind == "scatter":
                        required_channels.add(plot_def.x_channel)
                        required_channels.add(plot_def.y_channel)
                        if isinstance(plot_def.best_fit, (list, tuple)):
                            required_channels.update(
                                datafunctions.collect_multi_fit_condition_channels(plot_def.best_fit)
                            )
                        if plot_def.gate is not None:
                            required_channels.update(
                                datafunctions.collect_gate_channels(plot_def.gate)
                            )
                        if isinstance(plot_def.color_gate, (list, tuple)) and len(plot_def.color_gate) >= 3:
                            required_channels.update(
                                datafunctions.collect_gate_channels(tuple(plot_def.color_gate[:3]))
                            )
                    elif kind == "psd":
                        _extract_channels(plot_def.channel)
                        if getattr(plot_def, "gate", None) is not None:
                            required_channels.update(
                                datafunctions.collect_gate_channels(plot_def.gate)
                            )
                    elif kind == "histogram":
                        _extract_channels(plot_def.channel)
                    elif kind == "bar":
                        for ch, _agg in datafunctions.normalize_bar_metric_specs(plot_def.metrics):
                            _extract_channels(ch)
                    elif kind == "box":
                        _extract_channels(plot_def.channels)
                        if plot_def.gate is not None:
                            required_channels.update(
                                datafunctions.collect_gate_channels(plot_def.gate)
                            )
                    elif kind == "box_grid":
                        _extract_channels(plot_def.channels)
                        for _gate in plot_def.rows.values():
                            if _gate is not None:
                                required_channels.update(
                                    datafunctions.collect_gate_channels(_gate)
                                )
                        for _gate in plot_def.cols.values():
                            if _gate is not None:
                                required_channels.update(
                                    datafunctions.collect_gate_channels(_gate)
                                )
                    elif kind == "heatmap":
                        required_channels.add(plot_def.x_channel)
                        required_channels.add(plot_def.y_channel)
                        if plot_def.z_channel:
                            required_channels.add(plot_def.z_channel)
                        if plot_def.gate is not None:
                            required_channels.update(
                                datafunctions.collect_gate_channels(plot_def.gate)
                            )

        # Always pull sLap and tLap if the dataset has them — needed for axis
        # rendering and sample-rate detection.  TimeIntoExport is the
        # monotonic wall-clock column on CAR exports (HH:MM:SS.mmm strings);
        # it's the only reliable rate source when tLap is missing and sLap is
        # zero-order-held.
        for support in ("sLap", "tLap", "vCar", "TimeIntoExport"):
            required_channels.add(support)

        # Resolve calculated channel dependencies
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

        # Map canonical names back to raw source column names
        source_columns = set()
        mappings = self._reverse_mappings.get(source_type, {})
        for ch in resolved_channels:
            source_columns.add(mappings.get(ch, ch))

        return source_columns

    # ------------------------------------------------------------------
    # Parquet loading (with column projection)
    # ------------------------------------------------------------------

    def _available_parquet_engines(self):
        """Return parquet engines available in the current Python environment."""
        engines = []
        if importlib.util.find_spec("pyarrow") is not None:
            engines.append("pyarrow")
        if importlib.util.find_spec("fastparquet") is not None:
            engines.append("fastparquet")
        return engines

    def _get_parquet_schema_columns(self, file_path, engine):
        """Read column names from a parquet file without loading any row data."""
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
        """
        Normalize parquet column aliases: a leading underscore indicates
        an upper-case first character (e.g. '_fzTyreFL' → 'FzTyreFL').
        """
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
        """Find a parquet column by canonical logical name (supports underscore aliases)."""
        columns = [str(c).strip() for c in df.columns]
        column_set = set(columns)
        lower_target = logical_name.lower()

        for candidate in [
            logical_name, logical_name.lower(), logical_name.upper(),
            f"_{logical_name}", f"_{logical_name.lower()}",
            logical_name[0].upper() + logical_name[1:],
        ]:
            if candidate in column_set:
                return candidate

        insensitive = [c for c in columns if c.lower() == lower_target]
        if insensitive:
            if len(insensitive) > 1:
                log.warning(
                    "Multiple %s-like columns found: %s. Using '%s'.",
                    logical_name, ', '.join(insensitive), insensitive[0],
                )
            return insensitive[0]
        return None

    def _resolve_required_parquet_columns(self, schema_cols, columns_to_load, nrun=None, nlap=None):
        """
        Map canonical column names to raw parquet column names for column projection.
        Includes filter columns (nRun/nLap) if row filtering will be needed.
        """
        raw_set = set(schema_cols)
        raw_lower = {c.lower(): c for c in schema_cols}

        # Build canonical → raw mapping (cached per unique schema)
        schema_key = tuple(schema_cols)
        canonical_to_raw = self._parquet_alias_cache.get(schema_key)
        if canonical_to_raw is None:
            canonical_to_raw = {}
            for raw in schema_cols:
                if raw.startswith("_") and len(raw) > 1 and raw[1].isalpha():
                    canonical = raw[1].upper() + raw[2:]
                else:
                    canonical = raw
                canonical_to_raw.setdefault(canonical, raw)
                canonical_to_raw.setdefault(raw, raw)
            self._parquet_alias_cache[schema_key] = canonical_to_raw

        needed = set()

        # Add filter columns
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

        # Add data columns
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
        self, df, filter_spec, column_logical_name, file_path, run_name,
        is_rank=False, raise_on_missing_column=True, raise_on_empty_result=True,
    ):
        """Unified parquet filtering for rank-based (nRun) or value-based (nLap) selection.

        ``filter_spec`` may be a scalar (single value), or a ``range`` /
        ``list`` / ``tuple`` of values to keep all matching rows (#40).
        """
        if filter_spec is None:
            return df

        # Normalise iterable specs to a list of scalars (preserve order, dedupe).
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
                    unique_vals = sorted([
                        v for v in series.astype(str).str.strip().unique()
                        if v and v.lower() != "nan"
                    ])
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
                run_label, column_logical_name, run_col, target_values,
                len(filtered), len(df),
            )
            return filtered

        if is_rank:
            rank = int(pd.to_numeric(pd.Series([filter_spec]), errors="coerce").iloc[0])
            if rank < 1:
                raise ValueError(f"Run '{run_label}' {column_logical_name.lower()} must be >= 1.")

            if numeric.notna().any():
                unique_vals = sorted(numeric.dropna().unique().tolist())
            else:
                unique_vals = sorted([
                    v for v in series.astype(str).str.strip().unique()
                    if v and v.lower() != "nan"
                ])

            if rank > len(unique_vals):
                raise ValueError(
                    f"Run '{run_label}' requested {column_logical_name.lower()}={rank}, "
                    f"but only {len(unique_vals)} unique values exist. "
                    f"Available: {unique_vals[:12]}"
                    + (" ..." if len(unique_vals) > 12 else "")
                )
            target_value = unique_vals[rank - 1]
            mask = (numeric == target_value) if numeric.notna().any() else (
                series.astype(str).str.strip() == str(target_value)
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
            run_label, column_logical_name, column_logical_name.lower(),
            filter_spec, run_col, target_value, len(filtered), len(df),
        )
        return filtered

    def _load_parquet_with_fallback(
        self, file_path, columns_to_load=None, parquet_nrun=None, parquet_nlap=None, run_name=""
    ):
        """Load parquet with column projection, row filtering, and engine fallback."""
        available_engines = self._available_parquet_engines()
        if not available_engines:
            raise ImportError(
                "Parquet input requires 'pyarrow' or 'fastparquet', but neither is installed."
            )

        errors = []
        for engine in available_engines:
            try:
                # Step 1: Read schema column names (metadata only — no data)
                schema_cols = self._get_parquet_schema_columns(file_path, engine)

                if schema_cols is not None and columns_to_load:
                    # Step 2: Compute minimal column subset
                    col_subset = self._resolve_required_parquet_columns(
                        schema_cols, columns_to_load,
                        nrun=parquet_nrun, nlap=parquet_nlap,
                    )
                else:
                    col_subset = None

                # Step 3: Read parquet with optional column projection
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

                # Step 4: Apply row filters
                if parquet_nrun is not None:
                    df = self._apply_parquet_rank_value_filter(
                        df, filter_spec=parquet_nrun, column_logical_name="nRun",
                        file_path=file_path, run_name=run_name,
                        is_rank=True, raise_on_missing_column=True, raise_on_empty_result=True,
                    )
                elif parquet_nlap is not None:
                    df = self._apply_parquet_rank_value_filter(
                        df, filter_spec=parquet_nlap, column_logical_name="nLap",
                        file_path=file_path, run_name=run_name,
                        is_rank=False, raise_on_missing_column=False, raise_on_empty_result=False,
                    )

                # Step 5: Final column selection (handles any stragglers like nRun/nLap cols)
                if columns_to_load:
                    requested = sorted(set(columns_to_load))
                    available = [c for c in requested if c in df.columns]
                    missing = [c for c in requested if c not in df.columns]
                    if missing and self.verbose:
                        log.debug(
                            "Parquet '%s' missing %d channel(s): %s%s",
                            file_path.name, len(missing), ', '.join(missing[:10]),
                            " ..." if len(missing) > 10 else "",
                        )
                    if available:
                        df = df[available]
                    else:
                        raise KeyError(
                            f"No requested channels found in parquet. Requested: {requested[:10]}"
                        )

                return df
            except Exception as exc:
                errors.append(f"{engine}: {exc}")

        raise RuntimeError(
            f"Unable to load parquet '{file_path}' via engines {available_engines}. "
            f"Errors: {' | '.join(errors)}"
        )

    # ------------------------------------------------------------------
    # CSV/TXT loading (with usecols projection)
    # ------------------------------------------------------------------

    def _load_run_data(
        self, file_path, use_python_engine=False, columns_to_load=None,
        parquet_nrun=None, parquet_nlap=None, run_name="",
    ):
        """Load CSV/TXT or Parquet with column filtering applied at parse time."""
        try:
            if file_path.suffix.lower() == ".parquet":
                df = self._load_parquet_with_fallback(
                    file_path, columns_to_load=columns_to_load,
                    parquet_nrun=parquet_nrun, parquet_nlap=parquet_nlap, run_name=run_name,
                )
                df.columns = make_unique([str(c).strip() for c in df.columns])
                units = {c: "" for c in df.columns}
                return df, df.columns, units

            # Legacy CSV format: row 0 = metadata, row 1 = headers, row 2 = units, data from row 3
            with open(file_path, "r") as f:
                lines = f.readlines()

            header = make_unique(lines[1].strip().split(","))
            units_row = lines[2].strip().split(",")
            units = dict(zip(header, units_row))

            # Build usecols filter for parse-time column projection
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

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def _clean_data(self):
        """Remove non-numeric columns, patch YES/NO values, and interpolate gaps.

        Interpolation is restricted to the columns this run actually needs
        (per :pyattr:`run_required_cols`) when that information is available
        — typically a 5–10× speedup on wide DLS exports (#22).
        """
        interp_limit = max(1, int(self.FILTER_SAMPLE_RATE))

        for run_name in list(self.run_data.keys()):
            df = datafunctions.convert_yes_no_to_binary(self.run_data[run_name])

            # Drop string columns and sanitize numeric ones up-front.
            # Special-case ``TimeIntoExport`` (CAR HH:MM:SS.mmm strings) —
            # convert to seconds so the sample-rate detector can use it.
            drop_cols = []
            for col in list(df.columns):
                # Special-case ``TimeIntoExport`` regardless of whether the
                # column came in as object/str/string dtype.
                if col == "TimeIntoExport" and not pd.api.types.is_numeric_dtype(df[col]):
                    td = pd.to_timedelta(df[col].astype(str).str.strip(),
                                         errors="coerce")
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
                    method="linear", limit=interp_limit, axis=0,
                )
            self.run_data[run_name] = df

    def _ensure_preprocessed(self):
        """Guard against plotting before preprocessing has completed."""
        if not self._loaded:
            raise RuntimeError("Data has not been loaded.")
        if not self._preprocessed:
            raise RuntimeError("Data has not been preprocessed.")

    # ------------------------------------------------------------------
    # Cached PSD computation (#15)
    # ------------------------------------------------------------------

    def _cached_psd(self, run_name, channel, nperseg, gate_spec=None):
        """Compute (or fetch cached) PSD for a (run, channel, nperseg[, gate]) tuple.

        Returns ``(freq, power)`` for backward compatibility. The number of
        Welch sub-segments used is stored internally via
        :py:meth:`_cached_psd_with_segments` and reused for PSD-domain
        averaging across runs in a group.
        """
        freq, power, _n = self._cached_psd_with_segments(run_name, channel, nperseg, gate_spec)
        return freq, power

    def _cached_psd_with_segments(self, run_name, channel, nperseg, gate_spec=None):
        """Like :py:meth:`_cached_psd` but also returns the Welch segment count.

        The segment count is used as a weight when averaging PSDs across
        runs in the same ``group``. Returns ``(None, None, 0)`` on failure.
        """
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
            # Warn when the available (finite) sample count forces Welch to use a
            # much shorter segment than requested — frequency resolution and
            # averaging suffer, but the plot will still be drawn.
            finite_n = int(np.isfinite(signal).sum())
            if finite_n < nperseg and finite_n >= 8:
                effective = min(nperseg, finite_n)
                if effective < max(64, nperseg // 4):
                    log.warning(
                        "PSD '%s'/'%s': only %d finite samples — nperseg capped from %d to %d "
                        "(coarse frequency resolution, low averaging).",
                        run_name, channel, finite_n, nperseg, effective,
                    )
            freq, power = datafunctions.calculate_psd(signal, rate, nperseg=nperseg)
            # Estimate Welch sub-segments for weighting (50% overlap default).
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
                    run_name, channel, exc,
                )
                self._psd_cache[key] = (None, None, 0)
                return None, None, 0
            freq, power, n_segs = datafunctions.calculate_segmented_psd(
                signal, mask, rate, nperseg=nperseg,
            )
            if freq is None:
                log.warning(
                    "PSD '%s'/'%s': no gated segment >= nperseg (%d). Skipping.",
                    run_name, channel, nperseg,
                )

        self._psd_cache[key] = (freq, power, n_segs)
        return freq, power, n_segs

    # ------------------------------------------------------------------
    # Plot utilities
    # ------------------------------------------------------------------

    def _suggest_similar_channels(self, run_name, missing_channel, max_suggestions=5):
        """Return a list of available channel names similar to `missing_channel`.

        Uses ``datafunctions.suggest_similar_channels`` which combines
        substring/prefix matching with ``difflib.get_close_matches`` for
        typo-tolerant suggestions (#10).
        """
        df = self.run_data.get(run_name)
        if df is None:
            return []
        return datafunctions.suggest_similar_channels(
            missing_channel, list(df.columns), max_results=max_suggestions
        )

    def _format_missing_channel_hint(self, run_name, missing_channel):
        """Build a hint string showing similar available channels for a missing one."""
        suggestions = self._suggest_similar_channels(run_name, missing_channel)
        if suggestions:
            return f"  Similar available: {', '.join(suggestions)}"
        return ""

    def _get_plot_group(self, index):
        """Return one plot-definition group by index or an empty list."""
        if not self.PLOT_DEFINITIONS or len(self.PLOT_DEFINITIONS) <= index:
            return []
        return self.PLOT_DEFINITIONS[index] or []

    def _sanitize_plot_filename(self, prefix, plot_name, suffix=""):
        """Create a filesystem-safe, lowercase PNG path under a per-type subfolder (#35).

        Returns a relative path like ``"scatter/scatter_gear_ratios.png"`` so
        outputs land in ``plots/<type>/`` instead of a single flat directory.
        """
        safe = (
            plot_name.lower()
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_").replace("\\", "_")
        )
        subdir = self.plots_dir / prefix
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # Use forward slash consistently — ``Path / str`` joins correctly on all OSes.
        return f"{prefix}/{prefix}_{safe}{suffix}.png"

    def _resolve_plot_figsize(self, filename, default_size, *, min_height=None):
        """Resolve figure size using defaults and optional PPT template aspect ratio."""
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

        return (w, h)

    def _add_axis_edge_padding(self, ax, x_pad_ratio=0.02, y_pad_ratio=0.03):
        """Add proportional padding to current axis limits."""
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmax > xmin:
            pad = (xmax - xmin) * x_pad_ratio
            ax.set_xlim(xmin - pad, xmax + pad)
        if ymax > ymin:
            pad = (ymax - ymin) * y_pad_ratio
            ax.set_ylim(ymin - pad, ymax + pad)

    def _get_filtered_run_dataframe(self, run_name, gate_spec=None):
        """Return a cached gated dataframe for a run."""
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

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def load_data(self, root_folder: str | Path) -> dict[str, pd.DataFrame]:
        """Load raw run files into memory."""
        root_folder = Path(root_folder)
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
            file_path = root_folder / run["file"]

            if not file_path.exists():
                log.warning(
                    "Missing data file for run '%s': %s. Skipping run.",
                    run_name, file_path,
                )
                continue

            try:
                # use_python_engine can be set explicitly in the run dict, or falls back to False
                use_python_engine = run.get("use_python_engine", False)
                self.run_required_cols[run_name] = self._get_required_source_columns(
                    run.get("type", run_name)
                )

                data, _, units = self._load_run_data(
                    file_path,
                    use_python_engine=use_python_engine,
                    columns_to_load=self.run_required_cols[run_name],
                    parquet_nrun=run.get("nrun"),
                    parquet_nlap=run.get("nlap"),
                    run_name=run_name,
                )

                # Post-load nLap filter — handles list/tuple/range values for
                # any file type (parquet's internal filter already covers the
                # parquet path; this is idempotent there and adds support for
                # CSV/TXT). Scalar values pass through unchanged.
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
                                run_name, wanted, len(data), before,
                            )
                        else:
                            log.warning(
                                "Run '%s': nlap filter %s matched no rows; keeping all data.",
                                run_name, wanted,
                            )

                # #40 best_n: keep only the fastest N laps (auto-select).
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
                                run_name, n_keep,
                            )
                        else:
                            durations = (
                                data.groupby("nLap")[time_col]
                                .agg(lambda s: s.max() - s.min())
                                .sort_values()
                            )
                            durations = durations[durations > 0]
                            keep_laps = list(durations.head(n_keep).index)
                            if keep_laps:
                                before = len(data)
                                data = data[data["nLap"].isin(keep_laps)].reset_index(drop=True)
                                log.info(
                                    "Run '%s' best_n=%d -> kept laps %s (%d/%d rows).",
                                    run_name, n_keep, keep_laps, len(data), before,
                                )
            except Exception as exc:
                log.warning(
                    "Failed to load run '%s' from %s: %s. Skipping run.",
                    run_name, file_path, exc,
                )
                self.run_required_cols.pop(run_name, None)
                continue

            self.run_filepaths[run_name] = file_path
            self.run_data[run_name] = data
            self.run_units[run_name] = units
            loaded_runs.append(run)

        self.runs = loaded_runs if loaded_runs else []
        self._loaded = True
        return self.run_data

    def preprocess_data(self) -> None:
        """Apply mappings, transforms, calculated channels, and filters."""
        if not self._loaded:
            raise RuntimeError("Data must be loaded before preprocessing.")

        self._gated_data_cache.clear()
        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            source_type = run.get("type", name)
            self.run_data[name] = datafunctions.apply_channel_mappings(
                self.run_data[name], self.CHANNEL_MAPPINGS, source_type
            )
            self.run_data[name] = datafunctions.apply_transformations(
                self.run_data[name], source_type, self.CHANNEL_TRANSFORMS
            )

        self._clean_data()

        # Resample to a uniform rate (channel_config.RESAMPLE_RATE) so that
        # filter cutoffs designed at self.FILTER_SAMPLE_RATE are consistent
        # channel-to-channel and run-to-run. No-op if the resample rate is
        # 0/None or matches the source rate within 0.5%.
        if self.RESAMPLE_RATE and self.RESAMPLE_RATE > 0:
            for run in self.runs:
                name = run["name"].lower()
                if name not in self.run_data:
                    continue
                self.run_data[name] = datafunctions.resample_to_uniform_rate(
                    self.run_data[name], self.RESAMPLE_RATE, run_name=name,
                )

        # Detect per-run sample rates (#17). The global self.FILTER_SAMPLE_RATE
        # is preserved for filter design (which expects a single value); per-run
        # rates are stored separately so PSD and the data-quality report can use
        # the actual rate of each dataset rather than a single assumed value.
        detected_rates = []
        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            rate, source = datafunctions.detect_sample_rate(
                self.run_data[name], default=self.FILTER_SAMPLE_RATE
            )
            self.run_sample_rates[name] = (rate, source)
            detected_rates.append(rate)
            if self.verbose:
                log.debug("[%s] sample rate: %.1f Hz (source: %s)", name, rate, source)
        if detected_rates:
            rmin, rmax = min(detected_rates), max(detected_rates)
            if rmin > 0 and (rmax / rmin) > 1.05:
                log.warning(
                    "Per-run sample rates vary by %.2fx (%.1f–%.1f Hz). "
                    "This may affect PSD comparisons.",
                    rmax / rmin, rmin, rmax,
                )

        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            required = self.run_required_cols.get(name)
            required_set = set(required) if required else None
            datafunctions.apply_calculated_channels(
                self.run_data[name], name, self.CALCULATED_CHANNELS,
                required_channels=required_set,
            )

        # Cross-run derived channels (require >=2 runs; reference = first loaded).
        # Computed BEFORE filtering so they appear in the channel list normally.
        self._compute_tdiff_channel()

        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            required = self.run_required_cols.get(name)
            required_set = set(required) if required else None
            self.run_data[name] = datafunctions.apply_filters(
                self.run_data[name], self.FILTERS, self.FILTER_SAMPLE_RATE, name,
                required_channels=required_set,
            )

        self._preprocessed = True
        return self.run_data

    # ------------------------------------------------------------------
    # Cross-run derived channels
    # ------------------------------------------------------------------

    def reference_run_name(self) -> Optional[str]:
        """Return the lowercased name of the reference run.

        The reference run is selected by:
          1. The first loaded run with ``"reference": True`` in its config.
          2. Otherwise, the first loaded run.

        Returns ``None`` if no runs are loaded.
        """
        loaded = [r["name"].lower() for r in self.runs if r["name"].lower() in self.run_data]
        if not loaded:
            return None
        for run in self.runs:
            if run.get("reference") and run["name"].lower() in self.run_data:
                return run["name"].lower()
        return loaded[0]

    def _compute_tdiff_channel(self) -> None:
        """Compute a ``tDiff`` column for every loaded run.

        ``tDiff`` is the lap-time difference vs the reference run (see
        :meth:`reference_run_name`) at each ``sLap`` point:
        ``tDiff = tLap_this − interp(sLap_this, sLap_ref, tLap_ref)``.

        Reference run gets ``tDiff = 0``. Skipped silently if any run lacks
        ``sLap`` or a usable time channel (``tLap`` / ``tLap_Calc`` / ``Time``).
        """
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
            ref_t_on_s = np.interp(s, ref_s_sorted, ref_t_sorted,
                                   left=np.nan, right=np.nan)
            df["tDiff"] = t - ref_t_on_s
            log.debug("tDiff computed for '%s' (vs '%s').", name, ref_name)

    # ------------------------------------------------------------------
    # Shared rendering helpers
    # ------------------------------------------------------------------

    def _colorize_legend_labels(self, legend):
        """Match legend text color to the corresponding series color."""
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
        """Return the (halign, valign) corner string of a placed legend, or None.

        Falls back to inferring the corner from the legend's rendered bounding
        box (in axes-fraction coords) when the location is ``"best"`` or cannot
        otherwise be parsed.
        """
        if legend is None:
            return None
        loc_map = {
            "upper right":  ("right",  "top"),
            "upper left":   ("left",   "top"),
            "lower right":  ("right",  "bottom"),
            "lower left":   ("left",   "bottom"),
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

        # Fallback: derive from rendered bbox center (works for loc="best").
        return self._legend_corner_from_bbox(legend)

    def _legend_corner_from_bbox(self, legend):
        """Infer (halign, valign) from a legend's rendered axes-fraction bbox."""
        bbox = self._legend_axes_bbox(legend)
        if bbox is None:
            return None
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        halign = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        valign = "bottom" if cy < 1 / 3 else ("top" if cy > 2 / 3 else "center")
        return (halign, valign)

    def _legend_axes_bbox(self, legend):
        """Return the legend's bbox in axes-fraction coordinates, or None.

        Forces a canvas draw if needed so ``get_window_extent`` is valid.
        """
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

    # Ordered preference lists for legend corner when another corner is taken.
    # ------------------------------------------------------------------
    # Info-box placement: 4 corners with shared anchor coordinates.
    # All text boxes (fit info, gate info) use these XY positions, and the
    # legend is anchored to the same coordinates via bbox_to_anchor +
    # borderaxespad=0 — so two boxes sharing a row/column line up on the
    # same axes-fraction edge automatically.
    # ------------------------------------------------------------------
    _INFO_CORNER_XY = {
        ("left",   "top"):    (0.02, 0.98),
        ("right",  "top"):    (0.98, 0.98),
        ("left",   "bottom"): (0.02, 0.02),
        ("right",  "bottom"): (0.98, 0.02),
        ("center", "top"):    (0.50, 0.98),
        ("center", "bottom"): (0.50, 0.02),
        ("left",   "center"): (0.02, 0.50),
        ("right",  "center"): (0.98, 0.50),
    }

    _CORNER_TO_LOC = {
        ("left",   "top"):    "upper left",
        ("right",  "top"):    "upper right",
        ("left",   "bottom"): "lower left",
        ("right",  "bottom"): "lower right",
        ("center", "top"):    "upper center",
        ("center", "bottom"): "lower center",
        ("left",   "center"): "center left",
        ("right",  "center"): "center right",
    }

    # Map matplotlib loc strings to (halign, valign).
    _LOC_TO_CORNER = {
        "upper right":  ("right",  "top"),
        "upper left":   ("left",   "top"),
        "lower right":  ("right",  "bottom"),
        "lower left":   ("left",   "bottom"),
        "upper center": ("center", "top"),
        "lower center": ("center", "bottom"),
        "center left":  ("left",   "center"),
        "center right": ("right",  "center"),
    }

    def _sample_ax_data(self, ax):
        """Collect sampled (xs, ys) arrays from an axis's lines and collections."""
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
        """Count sampled data points in a corner region of the axes."""
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
        """Rank the 4 info-box corners from least to most data-dense.

        Returns a list of (halign, valign) tuples ordered by ascending point
        count in each corner region. Used by fit-info, legend, and gate-info
        placement so they all share the same density model and corner set.
        """
        xs, ys = self._sample_ax_data(ax)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        corners = list(self._INFO_CORNER_XY.keys())
        if xs.size == 0:
            return corners
        return sorted(
            corners,
            key=lambda c: self._count_points_in_region(
                xs, ys, x0, x1, y0, y1, c[0], c[1], w_frac, h_frac
            ),
        )

    def _add_standard_legend(self, ax, handles=None, labels=None, loc="best",
                              bbox_to_anchor=None, ncol=1, avoid_corner=None):
        """Add a consistently styled axis legend and colorize labels.

        avoid_corner: optional (halign, valign) tuple from the fit-info anchor;
        the legend is placed at the least data-dense non-conflicting corner.

        When the resolved location is one of the 4 standard corners, the legend
        is anchored at the shared corner coordinate (``_INFO_CORNER_XY``) with
        ``borderaxespad=0`` so its frame edges line up with the fit-info and
        gate-info text boxes (which use the same anchor points).
        """
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None

        if avoid_corner is not None and bbox_to_anchor is None:
            ranked = [c for c in self._rank_info_corners(ax) if c != avoid_corner]
            corner = ranked[0] if ranked else None
            loc = self._CORNER_TO_LOC.get(corner, "best") if corner else "best"

        # If loc names a standard corner, anchor it at the shared corner XY so
        # the legend frame aligns with text-box info panels.
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
        """Place waveform legend above (default) or to the right of subplots.

        position: 'top' (legend across the top, multi-column) or 'right'
            (vertical legend to the right of the plot area).
        """
        if not handles:
            return None
        if position == "right":
            legend = fig.legend(
                handles, labels,
                loc="center right", bbox_to_anchor=(1.0, 0.5),
                ncol=1,
                fancybox=True, framealpha=0.92, edgecolor="#3C3C3C",
                borderpad=0.4, handlelength=1.8,
                prop={"family": self.PLOT_FONT["family"], "weight": "bold", "size": self.PLOT_FONT["legend_size"]},
            )
        else:
            legend = fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, 1.0),
                ncol=max(1, min(len(handles), 5)),
                fancybox=True, framealpha=0.92, edgecolor="#3C3C3C",
                borderpad=0.3, handlelength=1.8,
                prop={"family": self.PLOT_FONT["family"], "weight": "bold", "size": self.PLOT_FONT["legend_size"]},
            )
        legend.get_frame().set_linewidth(1.4)
        legend.set_zorder(10)
        self._colorize_legend_labels(legend)
        return legend

    def _display_gate_info(self, ax, text, legend=None, trend_anchor=None):
        """Place gate-info callout at a free corner using shared corner anchors.

        Picks the data-clearest of the 4 standard corners that doesn't collide
        with the fit-info box or the legend. Because all info boxes share the
        same XY anchor points (``_INFO_CORNER_XY``), two boxes in the same row
        or column line up automatically.
        """
        occupied = set()
        if trend_anchor is not None:
            _, trend_halign, trend_valign, _ = trend_anchor
            occupied.add((trend_halign, trend_valign))

        # Determine the legend's corner — by its loc, falling back to its
        # rendered bbox center for loc="best".
        legend_corner = self._legend_corner(legend)
        if legend_corner is not None:
            occupied.add(legend_corner)

        ranked = self._rank_info_corners(ax)
        free = [c for c in ranked if c not in occupied]
        halign, valign = (free[0] if free else ranked[0])
        x_anchor, y_anchor = self._INFO_CORNER_XY[(halign, valign)]
        ax.text(
            x_anchor, y_anchor, text,
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment=valign, horizontalalignment=halign,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white", alpha=0.92,
                edgecolor="#3C3C3C", linewidth=1.4,
            ),
            color="#1A1A1A", fontweight="bold", family=self.PLOT_FONT["family"],
        )

    # ------------------------------------------------------------------
    # Plot dispatcher
    # ------------------------------------------------------------------

    def plot_data(self, plot_types: list[str] | None = None, plot_names: list[str] | None = None) -> None:
        """Run all (or filtered) plot generators.

        plot_types: list of 'waveform','scatter','psd','histogram','bar','box','heatmap' or None.
        plot_names: list of plot name strings (case-insensitive) or None.
        """
        self._ensure_preprocessed()

        # Clear per-run state that should not persist across calls (#4).
        self._psd_cache = {}
        self._gated_data_cache = {}
        self._outlier_log = []

        # Aggregate repeated WARNING messages emitted during plot generation
        # so the user gets a single summary at the end instead of dozens of
        # near-identical lines (#11).
        from collections import Counter as _Counter

        class _AggregatingHandler(logging.Handler):
            def __init__(self):
                super().__init__(level=logging.WARNING)
                self.counts = _Counter()

            def emit(self, record):
                # Use the *unformatted* message + args as the dedup key so
                # that e.g. "Cannot filter '%s'" with different channel names
                # collapses to one bucket.
                try:
                    key = record.getMessage()
                except Exception:
                    key = record.msg
                self.counts[key] += 1

        agg = _AggregatingHandler()
        log.addHandler(agg)

        all_generators = [
            ("waveform",   self.generate_waveform_plots),
            ("scatter",    self.generate_scatter_plots),
            ("psd",        self.generate_psd_plots),
            ("histogram",  self.generate_histogram_plots),
            ("bar",        self.generate_bar_plots),
            ("box",        self.generate_box_plots),
            ("heatmap",    self.generate_heatmap_plots),
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
            # Print a deduped summary of repeated warnings (#11).
            repeated = [(msg, n) for msg, n in agg.counts.most_common() if n > 1]
            if repeated:
                top = repeated[:10]
                log.info(
                    "Plot-time warning summary: %d unique repeated message(s); "
                    "showing top %d.", len(repeated), len(top),
                )
                for msg, n in top:
                    log.info("  [x%d] %s", n, msg)

        log.info("All plots saved to: %s", self.plots_dir)

