"""Data loading, preprocessing, and plotting pipeline for correlation reports.

Split into focused modules:
  plot_generators_waveform.py  — WaveformMixin
  plot_generators_scatter.py   — ScatterMixin
  plot_generators_misc.py      — PsdHistMixin, HeatmapMixin
  plot_generators_bar_box.py   — BarBoxMixin
"""

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
import datafunctions
from collections import Counter, deque
from matplotlib.patches import Patch

from plot_definitions import (
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
from plot_generators_waveform import WaveformMixin
from plot_generators_scatter import ScatterMixin
from plot_generators_misc import PsdHistMixin, HeatmapMixin
from plot_generators_bar_box import BarBoxMixin




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
            else:
                # Defensive fallback for unknown / legacy tuples.
                if isinstance(plot_def, (list, tuple)) and len(plot_def) >= 2:
                    _add(plot_def[1])
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
        best = None

        for scale in np.linspace(scale_guess - 0.01, scale_guess + 0.01, 25):
            for offset in np.linspace(offset_guess - 40, offset_guess + 40, 41):
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
            s0 = best["scale"]
            o0 = best["offset"]
            for scale in np.linspace(s0 - 0.002, s0 + 0.002, 21):
                for offset in np.linspace(o0 - 8, o0 + 8, 33):
                    score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset)
                    if score is None:
                        continue
                    corr, mae, n = score
                    key = (corr, -mae, n)
                    if key > best["key"]:
                        best = {
                            "scale": float(scale),
                            "offset": float(offset),
                            "corr": corr,
                            "mae": mae,
                            "n": int(n),
                            "key": key,
                        }

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


def build_quality_sections(runs, run_data, plot_definitions, run_sample_rates=None, outlier_log=None):
    """Backward-compatible thin wrapper around the data_quality_report module."""
    from data_quality_report import build_quality_sections as _impl
    return _impl(runs, run_data, plot_definitions, run_sample_rates, outlier_log)


def write_data_quality_report(plots_dir, sections):
    """Backward-compatible wrapper — emits Markdown via data_quality_report."""
    from data_quality_report import write_data_quality_report as _impl
    return _impl(plots_dir, sections)


# ---------------------------------------------------------------------------
# DataPlotter
# ---------------------------------------------------------------------------

class DataPlotter(WaveformMixin, ScatterMixin, PsdHistMixin, HeatmapMixin, BarBoxMixin):
    """Main class for loading, processing, and plotting multi-run data."""

    def __init__(
        self,
        root_folder,
        runs,
        plot_definitions=None,
        channel_mappings=None,
        channel_transforms=None,
        calculated_channels=None,
        filters=None,
        fig_size=None,
        units_map=None,
        plot_aspect_ratios=None,
        sample_rate=100,
        scatter_dot_size=5,
        scatter_transparency=0.7,
        scatter_max_points=45000,
        bar_secondary_axis_ratio=20.0,
        box_plot_settings=None,
        output_dir=None,
        verbose=False,
        output_dpi=300,
    ):
        """Build a plotter instance and run the preprocessing pipeline."""
        if fig_size is None:
            fig_size = {"waveform": (15.5, 6.4), "scatter": (10, 8), "psd": (10, 8), "histogram": (10, 8), "bar": (10, 6)}

        root_folder = Path(root_folder)
        output_dir = Path(output_dir) if output_dir is not None else root_folder
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

        self.SCATTER_DOT_SIZE = scatter_dot_size
        self.SCATTER_TRANSPARENCY = scatter_transparency
        self.SCATTER_MAX_POINTS = scatter_max_points

        # Accept both dict and legacy list format
        if isinstance(fig_size, dict):
            self.waveform_figsize = fig_size.get("waveform", (15.5, 6.4))
            self.scatter_FIGSIZE = fig_size.get("scatter", (10, 8))
            self.psd_FIGSIZE = fig_size.get("psd", (10, 8))
            self.histogram_FIGSIZE = fig_size.get("histogram", (10, 8))
            self.boxplot_FIGSIZE = fig_size.get("bar", (10, 6))
        else:
            self.waveform_figsize = fig_size[0]
            self.scatter_FIGSIZE = fig_size[1]
            self.psd_FIGSIZE = fig_size[2]
            self.histogram_FIGSIZE = fig_size[3]
            self.boxplot_FIGSIZE = fig_size[4] if len(fig_size) > 4 else (10, 6)
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
        "minor": {"alpha": 0.15, "linewidth": 0.3},
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
        # rendering and sample-rate detection.
        for support in ("sLap", "tLap", "vCar"):
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
                print(
                    f"[WARNING][DataPlotter] Multiple {logical_name}-like columns found: "
                    f"{', '.join(insensitive)}. Using '{insensitive[0]}'."
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
        """Unified parquet filtering for rank-based (nRun) or value-based (nLap) selection."""
        if filter_spec is None:
            return df

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
                print(f"[WARNING][DataPlotter] {msg}. Skipping filter.")
                return df

        series = df[run_col]
        numeric = pd.to_numeric(series, errors="coerce")

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
                print(f"[WARNING][DataPlotter] {msg}")
                return df

        print(
            f"[INFO][DataPlotter] Run '{run_label}' filtered by {column_logical_name}: "
            f"{column_logical_name.lower()}={filter_spec} → {run_col}={target_value} "
            f"({len(filtered)}/{len(df)} rows kept)."
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
                    print(
                        f"[INFO][DataPlotter] Run '{run_name.upper() if run_name else file_path.name}' "
                        "provided both nrun and nlap; applying nrun filter and ignoring nlap."
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
                        print(
                            f"[WARNING][DataPlotter] Parquet '{file_path.name}' missing "
                            f"{len(missing)} channel(s): {', '.join(missing[:10])}"
                            + (" ..." if len(missing) > 10 else "")
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
            print(f"[ERROR][DataPlotter] Failed to load '{file_path}': {e}")
            raise

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def _clean_data(self):
        """Remove non-numeric columns, patch YES/NO values, and interpolate gaps."""
        interp_limit = max(1, int(self.FILTER_SAMPLE_RATE))

        for run_name in list(self.run_data.keys()):
            df = datafunctions.convert_yes_no_to_binary(self.run_data[run_name])

            # Drop string columns and sanitize numeric ones up-front.
            drop_cols = []
            for col in list(df.columns):
                if df[col].dtype == "object":
                    non_nan = df[col].dropna()
                    if any(isinstance(x, str) for x in non_nan):
                        drop_cols.append(col)
                        continue
                df[col] = datafunctions.sanitize_numeric_series(df[col])
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                if self.verbose:
                    for col in drop_cols:
                        print(f"  Dropped '{col}' from run '{run_name}' (string column)")

            # Batched DataFrame-wide linear interpolation; ~10× faster than per-column.
            if not df.empty:
                df.interpolate(method="linear", limit=interp_limit, axis=0, inplace=True)
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

    def _cached_psd(self, run_name, channel, nperseg):
        """Compute (or fetch cached) PSD for a (run, channel, nperseg) triple."""
        key = (run_name, channel, nperseg)
        cached = self._psd_cache.get(key)
        if cached is not None:
            return cached
        df = self.run_data.get(run_name)
        if df is None or channel not in df.columns:
            return None, None
        signal = np.asarray(df[channel], dtype=float)
        rate = self.run_sample_rates.get(run_name, (self.FILTER_SAMPLE_RATE, "default"))[0]
        freq, power = datafunctions.calculate_psd(signal, rate, nperseg=nperseg)
        self._psd_cache[key] = (freq, power)
        return freq, power

    # ------------------------------------------------------------------
    # Data export (#7)
    # ------------------------------------------------------------------

    def export_run_data(self, export_format="csv"):
        """Dump preprocessed per-run dataframes to disk.

        Drops to ``<plots_dir>/exported_data/<run>.{csv|parquet}``.
        """
        self._ensure_preprocessed()
        fmt = export_format.lower()
        if fmt not in {"csv", "parquet"}:
            raise ValueError(f"export_format must be 'csv' or 'parquet'; got {fmt!r}.")
        out_dir = self.plots_dir / "exported_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for run_name, df in self.run_data.items():
            safe = run_name.replace("/", "_").replace("\\", "_")
            path = out_dir / f"{safe}.{fmt}"
            if fmt == "csv":
                df.to_csv(path, index=False)
            else:
                try:
                    df.to_parquet(path, index=False)
                except Exception as exc:
                    print(f"[WARNING][DataPlotter] Parquet export failed for '{run_name}': {exc}. Falling back to CSV.")
                    path = out_dir / f"{safe}.csv"
                    df.to_csv(path, index=False)
            written.append(path)
            if self.verbose:
                print(f"  Exported: {path}")
        print(f"Exported {len(written)} runs to {out_dir}")
        return written

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
        """Create a filesystem-safe, lowercase PNG name from a plot title."""
        safe = (
            plot_name.lower()
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_").replace("\\", "_")
        )
        return f"{prefix}_{safe}{suffix}.png"

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

    def load_data(self, root_folder):
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
                print(
                    f"[WARNING][DataPlotter] Missing data file for run '{run_name}': "
                    f"{file_path}. Skipping run."
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
            except Exception as exc:
                print(
                    f"[WARNING][DataPlotter] Failed to load run '{run_name}' from {file_path}: "
                    f"{exc}. Skipping run."
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

    def preprocess_data(self):
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
                print(f"  [{name}] sample rate: {rate:.1f} Hz (source: {source})")
        if detected_rates:
            rmin, rmax = min(detected_rates), max(detected_rates)
            if rmin > 0 and (rmax / rmin) > 1.05:
                print(
                    f"[WARNING][DataPlotter] Per-run sample rates vary by "
                    f"{rmax/rmin:.2f}x ({rmin:.1f}–{rmax:.1f} Hz). "
                    f"This may affect PSD comparisons."
                )

        for run in self.runs:
            name = run["name"].lower()
            if name not in self.run_data:
                continue
            datafunctions.apply_calculated_channels(
                self.run_data[name], name, self.CALCULATED_CHANNELS
            )
            self.run_data[name] = datafunctions.apply_filters(
                self.run_data[name], self.FILTERS, self.FILTER_SAMPLE_RATE, name,
            )

        self._preprocessed = True
        return self.run_data

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
        """Return the (halign, valign) corner string of a placed legend, or None."""
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
            loc_str = legend._loc_real if hasattr(legend, "_loc_real") else None
            # matplotlib stores the numeric code; map it via _loc
            if loc_str is None:
                for name, code in legend.codes.items():
                    if code == legend._loc:
                        loc_str = name
                        break
        except Exception:
            return None
        return loc_map.get(loc_str)

    # Ordered preference lists for legend corner when another corner is taken.
    _LEGEND_CORNER_PREFS = {
        ("left",   "top"):    ["upper right", "lower right", "lower center", "lower left",  "upper center", "upper left"],
        ("right",  "top"):    ["upper left",  "lower left",  "lower center", "lower right", "upper center", "upper right"],
        ("left",   "bottom"): ["lower right", "upper right", "upper center", "upper left",  "lower center", "lower left"],
        ("right",  "bottom"): ["lower left",  "upper left",  "upper center", "upper right", "lower center", "lower right"],
        ("center", "top"):    ["upper right", "upper left",  "lower right",  "lower left",  "lower center", "upper center"],
        ("center", "bottom"): ["lower right",  "lower left", "upper right",  "upper left",  "upper center", "lower center"],
        ("left",   "center"): ["upper right", "lower right", "upper left",   "lower left",  "upper center", "lower center"],
        ("right",  "center"): ["upper left",  "lower left",  "upper right",  "lower right", "upper center", "lower center"],
    }

    # Map matplotlib loc strings to (halign, valign) for density scoring
    _LOC_TO_CORNER = {
        "upper right":  ("right",  "top"),
        "upper left":   ("left",   "top"),
        "lower right":  ("right",  "bottom"),
        "lower left":   ("left",   "bottom"),
        "upper center": ("center", "top"),
        "lower center": ("center", "bottom"),
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

    def _score_legend_position(self, ax, loc_str):
        """Score a legend position by data density — lower is better."""
        corner = self._LOC_TO_CORNER.get(loc_str)
        if corner is None:
            return 0
        halign, valign = corner
        xs, ys = self._sample_ax_data(ax)
        if xs.size == 0:
            return 0
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return self._count_points_in_region(xs, ys, x0, x1, y0, y1, halign, valign)

    def _add_standard_legend(self, ax, handles=None, labels=None, loc="best",
                              bbox_to_anchor=None, ncol=1, avoid_corner=None):
        """Add a consistently styled axis legend and colorize labels.

        avoid_corner: optional (halign, valign) tuple from the fit-info anchor;
        the legend is placed in the least data-dense non-conflicting corner.
        """
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None

        if avoid_corner is not None and bbox_to_anchor is None:
            prefs = self._LEGEND_CORNER_PREFS.get(avoid_corner)
            if prefs:
                # Pick the least data-dense position from the preference list
                loc = min(prefs, key=lambda p: self._score_legend_position(ax, p))
            else:
                loc = "best"

        legend = ax.legend(
            handles, labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            fancybox=True,
            framealpha=0.92,
            edgecolor="#3C3C3C",
            borderpad=0.55,
            handlelength=1.8,
            ncol=ncol,
            prop={"family": self.PLOT_FONT["family"], "weight": "bold", "size": self.PLOT_FONT["legend_size"]},
        )
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
        """Place gate-info callout in a free corner, avoiding fit box and legend.

        Bottom positions use y=0.07 to provide clearance from tick labels and
        the x-axis label. Mid-edge positions are included for additional options.
        """
        all_positions = [
            (0.03, 0.97, "left",   "top"),
            (0.97, 0.97, "right",  "top"),
            (0.03, 0.07, "left",   "bottom"),
            (0.97, 0.07, "right",  "bottom"),
            (0.50, 0.97, "center", "top"),
            (0.50, 0.07, "center", "bottom"),
            (0.03, 0.50, "left",   "center"),
            (0.97, 0.50, "right",  "center"),
        ]

        # --- Step 1: exclude known-occupied positions deterministically --------
        occupied = set()

        # Fit-info box corner
        if trend_anchor is not None:
            _, trend_halign, trend_valign, _ = trend_anchor
            occupied.add((trend_halign, trend_valign))

        # Legend corner
        legend_corner = self._legend_corner(legend)
        if legend_corner is not None:
            occupied.add(legend_corner)

        free = [c for c in all_positions if (c[2], c[3]) not in occupied]
        candidates = free if free else all_positions

        # --- Step 2: pick the least data-dense corner (sampled) ----------------
        xs, ys = self._sample_ax_data(ax)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        def _data_density(c):
            _, _, hal, val = c
            return self._count_points_in_region(xs, ys, x0, x1, y0, y1, hal, val, 0.22, 0.28)

        chosen = min(candidates, key=_data_density)

        x_anchor, y_anchor, halign, valign = chosen
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

    def plot_data(self, plot_types=None, plot_names=None):
        """Run all (or filtered) plot generators.

        plot_types: list of 'waveform','scatter','psd','histogram','bar','box','heatmap' or None.
        plot_names: list of plot name strings (case-insensitive) or None.
        """
        self._ensure_preprocessed()

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
                print(f"[WARNING][DataPlotter] No plot types matched from: {plot_types!r}")
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

        print(f"\nAll plots saved to: {self.plots_dir}")

