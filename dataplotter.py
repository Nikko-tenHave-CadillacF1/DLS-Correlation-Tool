"""Data loading, preprocessing, and plotting pipeline for correlation reports."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import font_manager
from pathlib import Path
import importlib.util
import datafunctions
import data_quality_report
from collections import Counter, deque
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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


def _get_run_label(run_name: str, file_path=None) -> str:
    """Generate consistent run label for logging messages."""
    if run_name:
        return run_name.upper()
    if file_path:
        return file_path.name if hasattr(file_path, 'name') else str(file_path)
    return "unknown"


def _log_info(run_name: str, msg: str, file_path=None):
    """Log an info-level message with consistent formatting."""
    label = _get_run_label(run_name, file_path)
    print(f"[INFO][DataPlotter] Run '{label}' {msg}")


def _log_warning(run_name: str, msg: str, file_path=None):
    """Log a warning-level message with consistent formatting."""
    label = _get_run_label(run_name, file_path)
    print(f"[WARNING][DataPlotter] Run '{label}' {msg}")


def _log_error(run_name: str, msg: str, file_path=None):
    """Log an error-level message with consistent formatting."""
    label = _get_run_label(run_name, file_path)
    print(f"[ERROR][DataPlotter] Run '{label}' {msg}")


def _extract_calculated_dependencies(func):
    """Return source column names referenced by a calculated-channel lambda."""
    import inspect
    import re

    try:
        source = inspect.getsource(func)
    except Exception:
        return set()

    matches = re.findall(r"df\['([^']+)'\]|df\[\"([^\"]+)\"\]", source)
    return {m[0] or m[1] for m in matches}


def _resolve_scatter_style(point_count, base_size, base_alpha):
    """Tune scatter styling slightly for dense plots."""
    if point_count <= 5000:
        return base_size, base_alpha
    if point_count <= 20000:
        return max(3.5, base_size * 0.9), min(0.75, base_alpha + 0.05)
    if point_count <= 60000:
        return max(3.0, base_size * 0.8), max(0.35, base_alpha * 0.8)
    return max(2.5, base_size * 0.7), max(0.22, base_alpha * 0.65)


class DataPlotter:
    """Main class for loading, processing, and plotting multi-run data."""

    def __init__(
        self,
        root_folder,
        runs,
        plot_definitions=None,
        channel_mappings=None,
        channel_transforms=None,
        calculated_channels=None,
        low_pass_filters=None,
        fig_size=[(15.5, 6.4), (10, 8), (10, 8), (10, 8), (10, 6)],
        units_map=None,
        plot_aspect_ratios=None,
        sample_rate=100,
        scatter_dot_size=5,
        scatter_transparency=0.7,
        scatter_render_mode="auto",
        scatter_density_threshold=25000,
        scatter_max_points=45000,
        scatter_hexbin_gridsize=70,
        bar_secondary_axis_ratio=20.0,
        box_plot_settings=None,
        output_dir=None,
    ):
        """Build a plotter instance and run the preprocessing pipeline."""
        root_folder = Path(root_folder)
        output_dir = Path(output_dir) if output_dir is not None else root_folder
        self.runs = runs
        self._configure_plot_style()
        self.BAR_SECONDARY_AXIS_RATIO = float(bar_secondary_axis_ratio)

        # Store config
        self.PLOT_DEFINITIONS = plot_definitions
        self.CHANNEL_MAPPINGS = channel_mappings
        self.CALCULATED_CHANNELS = calculated_channels

        self.run_filepaths = {}
        self.run_data = {}
        self.run_units = {}
        self.run_required_cols = {}
        self._calculated_dependency_cache = {}
        self._gated_data_cache = {}
        self._reverse_mappings = {}
        self._loaded = False
        self._preprocessed = False

        if self.CHANNEL_MAPPINGS:
            for source_type, mapping in self.CHANNEL_MAPPINGS.items():
                self._reverse_mappings[source_type] = {
                    mapped: raw for raw, mapped in mapping.items()
                }

        self.CHANNEL_TRANSFORMS = channel_transforms
        self.units_map = units_map
        self.FILTER_SAMPLE_RATE = sample_rate
        self.LOW_PASS_FILTERS = low_pass_filters

        self.SCATTER_DOT_SIZE = scatter_dot_size
        self.SCATTER_TRANSPARENCY = scatter_transparency
        self.SCATTER_RENDER_MODE = scatter_render_mode
        self.SCATTER_DENSITY_THRESHOLD = scatter_density_threshold
        self.SCATTER_MAX_POINTS = scatter_max_points
        self.SCATTER_HEXBIN_GRIDSIZE = scatter_hexbin_gridsize

        self.waveform_figsize = fig_size[0]
        self.scatter_FIGSIZE = fig_size[1]
        self.psd_FIGSIZE = fig_size[2]
        self.histogram_FIGSIZE = fig_size[3]
        self.boxplot_FIGSIZE = fig_size[4] if len(fig_size) > 4 else (10, 6)
        self.plot_aspect_ratios = plot_aspect_ratios or {}
        self.BOX_PLOT_SETTINGS = box_plot_settings or {}

        # Create plots directory
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline
        self.load_data(root_folder)
        self.preprocess_data()

    # ------------------------------------------------------------
    # STYLE
    # ------------------------------------------------------------

    def _configure_plot_style(self):
        """Apply a consistent font and baseline styling to all plots."""
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        preferred_font = "Montserrat" if "Montserrat" in available_fonts else "DejaVu Sans"

        plt.rcParams.update(
            {
                "font.family": preferred_font,
                "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial", "sans-serif"],
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.labelsize": 11,
                "axes.labelweight": "bold",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
                "figure.titleweight": "bold",
            }
        )

    # ------------------------------------------------------------
    # LOAD REQUIRED COLUMNS
    # ------------------------------------------------------------

    def _get_required_source_columns(self, source_type):
        """Determine which columns to load from files."""
        required_channels = set()

        def _extract_channels(spec_item):
            """Recursively collect channel names from strings/tuples/lists."""
            if isinstance(spec_item, str):
                required_channels.add(spec_item)
                return
            if isinstance(spec_item, (list, tuple)):
                for part in spec_item:
                    _extract_channels(part)

        # Always include sLap for waveform alignment
        if (
            self.PLOT_DEFINITIONS
            and len(self.PLOT_DEFINITIONS) > 0
            and self.PLOT_DEFINITIONS[0]
        ):
            required_channels.add("sLap")

        # Scan all plot definitions
        if self.PLOT_DEFINITIONS:
            for plot_group_index, plot_group in enumerate(self.PLOT_DEFINITIONS):
                if plot_group is None:
                    continue
                for plot_def in plot_group:
                    if plot_group_index == 4:
                        if len(plot_def) >= 2:
                            metric_specs = datafunctions.normalize_bar_metric_specs(plot_def[1])
                            for channel, _aggregation in metric_specs:
                                _extract_channels(channel)
                    else:
                        if len(plot_def) >= 2:
                            _extract_channels(plot_def[1])

                    if plot_group_index == 1:
                        # Scatter-specific auxiliary channels used by fits/gates:
                        # plot_def[3] => best_fit, plot_def[4]/[5] => gate spec.
                        if len(plot_def) >= 4:
                            best_fit = plot_def[3]
                            fit_channels = datafunctions.collect_multi_fit_condition_channels(best_fit)
                            required_channels.update(fit_channels)

                        gate_spec = None
                        if len(plot_def) == 5:
                            gate_spec = plot_def[4]
                        elif len(plot_def) >= 6:
                            gate_spec = plot_def[5]
                        if datafunctions.is_gate_spec(gate_spec):
                            gate_channels = datafunctions.collect_gate_channels(gate_spec)
                            required_channels.update(gate_channels)

                    elif plot_group_index == 5:
                        # Box plots use [name, channels, mode, limits, gate_spec, options]
                        gate_spec = plot_def[4] if len(plot_def) >= 5 else None
                        if datafunctions.is_gate_spec(gate_spec):
                            gate_channels = datafunctions.collect_gate_channels(gate_spec)
                            required_channels.update(gate_channels)

        # Resolve calculated dependencies
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
                cache_key = id(calc_set[channel])
                deps = self._calculated_dependency_cache.get(cache_key)
                if deps is None:
                    deps = _extract_calculated_dependencies(calc_set[channel])
                    self._calculated_dependency_cache[cache_key] = deps
                for dep in deps:
                    if dep not in processed:
                        to_process.append(dep)
            else:
                resolved_channels.add(channel)

        # Apply channel mappings: convert mapped names to original raw names
        source_columns = set()
        mappings = self._reverse_mappings.get(source_type, {})
        for ch in resolved_channels:
            source_columns.add(mappings.get(ch, ch))

        return source_columns

    # ------------------------------------------------------------
    # LOAD RUN DATA
    # ------------------------------------------------------------

    def _available_parquet_engines(self):
        """Return parquet engines available in the current Python environment."""
        engines = []
        if importlib.util.find_spec("pyarrow") is not None:
            engines.append("pyarrow")
        if importlib.util.find_spec("fastparquet") is not None:
            engines.append("fastparquet")
        return engines

    def _normalize_parquet_column_aliases(self, df):
        """
        Normalize parquet column aliases where a leading underscore indicates
        an upper-case first character (e.g. '_fzTyreFL' -> 'FzTyreFL').
        """
        raw_columns = [str(c).strip() for c in df.columns]
        rename_map = {}
        existing = set(raw_columns)

        for col in raw_columns:
            if col.startswith("_") and len(col) > 1 and col[1].isalpha():
                canonical = col[1].upper() + col[2:]
                # Do not overwrite an already present canonical channel.
                if canonical not in existing:
                    rename_map[col] = canonical

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _find_parquet_column(self, df, logical_name):
        """Find a parquet column by canonical logical name (supports underscore aliases)."""
        columns = [str(c).strip() for c in df.columns]
        column_set = set(columns)  # For O(1) lookup instead of O(n)
        lower_target = logical_name.lower()

        # Exact/canonical-first candidates
        candidates = [
            logical_name,
            logical_name.lower(),
            logical_name.upper(),
            f"_{logical_name}",
            f"_{logical_name.lower()}",
            logical_name[0].upper() + logical_name[1:],  # e.g. NRun / NLap
        ]
        for candidate in candidates:
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

    def _apply_parquet_nrun_filter(self, df, nrun, file_path, run_name=""):
        """
        Filter parquet rows by nRun rank:
        nrun=1 -> lowest nRun value, nrun=2 -> next lowest, etc.
        
        Unified with _apply_parquet_lap_filter via _apply_parquet_rank_value_filter.
        """
        return self._apply_parquet_rank_value_filter(
            df=df,
            filter_spec=nrun,
            column_logical_name="nRun",
            file_path=file_path,
            run_name=run_name,
            is_rank=True,
            raise_on_missing_column=True,
            raise_on_empty_result=True,
        )

    def _apply_parquet_lap_filter(self, df, nlap, file_path, run_name=""):
        """
        Filter parquet rows to a specific lap index using nLap.
        
        Unified with _apply_parquet_nrun_filter via _apply_parquet_rank_value_filter.
        """
        return self._apply_parquet_rank_value_filter(
            df=df,
            filter_spec=nlap,
            column_logical_name="nLap",
            file_path=file_path,
            run_name=run_name,
            is_rank=False,
            raise_on_missing_column=False,
            raise_on_empty_result=False,
        )

    def _apply_parquet_rank_value_filter(
        self,
        df,
        filter_spec,
        column_logical_name: str,
        file_path,
        run_name: str,
        is_rank: bool = False,
        raise_on_missing_column: bool = True,
        raise_on_empty_result: bool = True,
    ):
        """
        Unified parquet filtering for rank-based (nRun) or value-based (nLap) selection.
        
        Args:
            df: DataFrame to filter
            filter_spec: rank (int 1,2,3...) or value to match
            column_logical_name: "nRun" or "nLap"
            file_path: Path object for error messages
            run_name: Run name string for error messages
            is_rank: True for rank-based selection, False for direct value matching
            raise_on_missing_column: Raise KeyError or print warning if column missing
            raise_on_empty_result: Raise ValueError or print warning if result is empty
        """
        if filter_spec is None:
            return df

        run_label = run_name.upper() if run_name else file_path.name
        run_col = self._find_parquet_column(df, column_logical_name)

        # Handle missing column
        if run_col is None:
            msg = f"Run '{run_label}' requested {column_logical_name.lower()}={filter_spec}, but parquet has no '{column_logical_name}' column"
            if raise_on_missing_column:
                raise KeyError(
                    msg + f" (accepted aliases: {column_logical_name}, {column_logical_name.lower()}, _{column_logical_name}, _{column_logical_name.lower()})."
                )
            else:
                print(f"[WARNING][DataPlotter] {msg}. Skipping filter.")
                return df

        series = df[run_col]
        numeric = pd.to_numeric(series, errors="coerce")
        
        # Determine target value(s) and create mask
        if is_rank:
            # Rank-based selection (nRun)
            rank_numeric = pd.to_numeric(pd.Series([filter_spec]), errors="coerce").iloc[0]
            if pd.isna(rank_numeric):
                raise ValueError(f"Run '{run_label}' {column_logical_name.lower()} must be an integer rank (1-based). Got: {filter_spec!r}")
            
            rank = int(rank_numeric)
            if rank < 1:
                raise ValueError(f"Run '{run_label}' {column_logical_name.lower()} must be >= 1. Got: {filter_spec!r}")

            if numeric.notna().any():
                unique_vals = sorted(numeric.dropna().unique().tolist())
                if rank > len(unique_vals):
                    raise ValueError(
                        f"Run '{run_label}' requested {column_logical_name.lower()}={rank}, but only {len(unique_vals)} unique {column_logical_name} values "
                        f"exist in '{run_col}'. Available: {unique_vals[:12]}"
                        + (" ..." if len(unique_vals) > 12 else "")
                    )
                target_value = unique_vals[rank - 1]
                mask = numeric == target_value
            else:
                str_vals = series.astype(str).str.strip()
                unique_vals = sorted([v for v in str_vals.unique().tolist() if v and v.lower() != "nan"])
                if rank > len(unique_vals):
                    raise ValueError(
                        f"Run '{run_label}' requested {column_logical_name.lower()}={rank}, but only {len(unique_vals)} unique {column_logical_name} values "
                        f"exist in '{run_col}'. Available: {unique_vals[:12]}"
                        + (" ..." if len(unique_vals) > 12 else "")
                    )
                target_value = unique_vals[rank - 1]
                mask = str_vals == target_value
        else:
            # Direct value matching (nLap)
            target_series = pd.to_numeric(pd.Series([filter_spec]), errors="coerce")
            target_numeric = target_series.iloc[0]

            if pd.notna(target_numeric):
                mask = numeric == float(target_numeric)
            else:
                mask = series.astype(str).str.strip() == str(filter_spec).strip()
            
            target_value = filter_spec

        filtered = df.loc[mask].copy()

        # Handle empty result
        if filtered.empty:
            msg = f"Run '{run_label}' {column_logical_name.lower()}={filter_spec} produced 0 rows from parquet column '{run_col}'."
            if raise_on_empty_result:
                raise ValueError(msg)
            else:
                print(f"[WARNING][DataPlotter] {msg}")
                return df

        print(
            f"[INFO][DataPlotter] Run '{run_label}' filtered parquet by {column_logical_name}: "
            f"{column_logical_name.lower()}={filter_spec} -> {run_col}={target_value} ({len(filtered)}/{len(df)} rows kept)."
        )
        return filtered

    def _load_parquet_with_fallback(
        self,
        file_path,
        columns_to_load=None,
        parquet_nrun=None,
        parquet_nlap=None,
        run_name="",
    ):
        """Load parquet with automatic engine fallback and optional column filtering."""
        available_engines = self._available_parquet_engines()
        if not available_engines:
            raise ImportError(
                "Parquet input requires 'pyarrow' or 'fastparquet', but neither is installed. "
                "Install one parquet backend and rerun."
            )

        errors = []
        for engine in available_engines:
            try:
                # Read full parquet first so we can normalize underscore aliases
                # before applying required-column filtering.
                df = pd.read_parquet(file_path, engine=engine)
                df.columns = [str(c).strip() for c in df.columns]
                df = self._normalize_parquet_column_aliases(df)
                if parquet_nrun is not None and parquet_nlap is not None:
                    print(
                        f"[INFO][DataPlotter] Run '{run_name.upper() if run_name else file_path.name}' "
                        f"provided both nrun and nlap; applying nrun filter and ignoring nlap."
                    )

                if parquet_nrun is not None:
                    df = self._apply_parquet_nrun_filter(
                        df,
                        nrun=parquet_nrun,
                        file_path=file_path,
                        run_name=run_name,
                    )
                elif parquet_nlap is not None:
                    df = self._apply_parquet_lap_filter(
                        df,
                        nlap=parquet_nlap,
                        file_path=file_path,
                        run_name=run_name,
                    )

                if columns_to_load:
                    requested = sorted(set(columns_to_load))
                    available = [c for c in requested if c in df.columns]
                    missing = [c for c in requested if c not in df.columns]
                    if missing:
                        print(
                            f"[WARNING][DataPlotter] Parquet file '{file_path.name}' is missing "
                            f"{len(missing)} requested channel(s): {', '.join(missing[:10])}"
                            + (" ..." if len(missing) > 10 else "")
                        )
                    if available:
                        df = df[available]
                    else:
                        raise KeyError(
                            f"No requested channels found after parquet alias normalization. "
                            f"Requested: {requested[:10]}"
                            + (" ..." if len(requested) > 10 else "")
                        )

                return df
            except Exception as exc:
                errors.append(f"{engine}: {exc}")

        details = " | ".join(errors)
        raise RuntimeError(
            f"Unable to load parquet file '{file_path}' using available engines {available_engines}. "
            f"Errors: {details}"
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
        """Load CSV/TXT or Parquet, applying column filtering."""
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

            # Legacy CSV format
            with open(file_path, "r") as f:
                lines = f.readlines()

            header = make_unique(lines[1].strip().split(","))
            units_row = lines[2].strip().split(",")

            kwargs = dict(
                sep=",",
                skiprows=3,
                header=None,
                names=header,
                on_bad_lines="skip",
            )
            if use_python_engine:
                kwargs["engine"] = "python"
            else:
                kwargs["low_memory"] = False

            df = pd.read_csv(file_path, **kwargs)
            units = dict(zip(header, units_row))

            # Filter columns
            if columns_to_load:
                cols = [c for c in header if c in columns_to_load]
                df = df[cols]
                units = {c: units.get(c, "") for c in cols}

            return df, df.columns, units

        except Exception as e:
            print(f"[ERROR][DataPlotter] Failed to load data file '{file_path}': {e}")
            raise

    # ------------------------------------------------------------
    # CLEAN DATA
    # ------------------------------------------------------------

    def clean_data(self):
        """Remove non-numeric columns and patch YES/NO."""
        for run_name in list(self.run_data.keys()):
            df = datafunctions.convert_yes_no_to_binary(self.run_data[run_name])
            for col in list(df.columns):
                if df[col].dtype == "object":
                    non_nan = df[col].dropna()
                    if any(isinstance(x, str) for x in non_nan):
                        df.drop(col, axis=1, inplace=True)
                        print(f"Dropped {col} from run {run_name} (string column)")
                        continue

                df[col] = datafunctions.sanitize_numeric_series(df[col])
                df[col] = df[col].interpolate(method="linear")
            self.run_data[run_name] = df

    # ------------------------------------------------------------
    # MAPPINGS / TRANSFORMS / CALCULATED / FILTERS
    # ------------------------------------------------------------

    def apply_channel_mappings(self):
        """Apply source-specific channel renaming for every loaded run."""
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_channel_mappings(
                    self.run_data[name], self.CHANNEL_MAPPINGS, run.get("type", name)
                )

    def apply_transformations(self):
        """Apply configured per-source numeric transforms to each run."""
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_transformations(
                    self.run_data[name], run.get("type", name), self.CHANNEL_TRANSFORMS
                )

    def apply_calculated_channels(self):
        """Create configured derived channels for each run."""
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                datafunctions.apply_calculated_channels(
                    self.run_data[name], name, self.CALCULATED_CHANNELS
                )

    def apply_lowpass_filters(self):
        """Apply low-pass filtering to each run using shared settings."""
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_lowpass_filters(
                    self.run_data[name],
                    self.LOW_PASS_FILTERS,
                    self.FILTER_SAMPLE_RATE,
                    name,
                )

    def _ensure_preprocessed(self):
        """Guard against plotting before preprocessing has completed."""
        if not self._loaded:
            raise RuntimeError("Data has not been loaded.")
        if not self._preprocessed:
            raise RuntimeError("Data has not been preprocessed.")

    # ------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------

    def _get_plot_group(self, index):
        """Return one plot-definition group by index or an empty list."""
        if not self.PLOT_DEFINITIONS or len(self.PLOT_DEFINITIONS) <= index:
            return []
        return self.PLOT_DEFINITIONS[index] or []

    def _sanitize_plot_filename(self, prefix, plot_name, suffix=""):
        """Create a filesystem-safe PNG name from a plot title."""
        safe = (
            plot_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("\\", "_")
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

    def _prepare_scatter_xy(self, df, x_var, y_var):
        """Build aligned numeric x/y arrays from a dataframe."""
        if df is None:
            return None, None, None
        if x_var not in df.columns or y_var not in df.columns:
            return None, None, None

        xy = pd.concat(
            [
                pd.to_numeric(df[x_var], errors="coerce").rename(x_var),
                pd.to_numeric(df[y_var], errors="coerce").rename(y_var),
            ],
            axis=1,
        ).dropna()
        if xy.empty:
            return None, None, None
        return xy.index, xy[x_var].to_numpy(dtype=float), xy[y_var].to_numpy(dtype=float)

    def _resolve_scatter_plot_style(self, point_count):
        """Adjust scatter style slightly for dense plots."""
        return _resolve_scatter_style(point_count, self.SCATTER_DOT_SIZE, self.SCATTER_TRANSPARENCY)

    def _build_gradient_segment_labels(self, fit_defs, x_var=None, y_var=None):
        """Create descriptive labels for segmented gradient error reporting."""
        if not isinstance(fit_defs, (list, tuple)):
            return None

        labels = []
        for idx, fit_def in enumerate(fit_defs, start=1):
            if not isinstance(fit_def, (list, tuple)) or len(fit_def) != 3:
                labels.append(f"Segment {idx}")
                continue

            axis, min_val, max_val = fit_def
            axis_name = x_var if axis == "x" else y_var if axis == "y" else str(axis)

            if min_val is None and max_val is None:
                labels.append(f"{axis_name}: full range")
            elif min_val is None:
                labels.append(f"{axis_name} < {max_val:g}")
            elif max_val is None:
                labels.append(f"{axis_name} >= {min_val:g}")
            else:
                labels.append(f"{min_val:g} <= {axis_name} < {max_val:g}")

        return labels if labels else None

    def run_data_quality_checks(self):
        """Run lightweight checks and write a report before plotting."""
        sections = data_quality_report.build_quality_sections(
            runs=self.runs,
            run_data=self.run_data,
            plot_definitions=self.PLOT_DEFINITIONS,
        )

        total_items = sum(len(v) for _, v in sections)
        if total_items:
            print(f"[WARNING][DataPlotter] Data-quality preflight found {total_items} issue(s).")
        else:
            print("[INFO][DataPlotter] Data-quality preflight found no issues.")

        report_path = data_quality_report.write_data_quality_report(self.plots_dir, sections)
        print(f"[INFO][DataPlotter] Wrote data quality report: {report_path}")

    # ------------------------------------------------------------
    # PIPELINE STAGES
    # ------------------------------------------------------------

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
                print(f"[WARNING][DataPlotter] Missing data file for run '{run_name}': {file_path}. Skipping run.")
                continue

            try:
                use_python_engine = (run_name == "car")
                self.run_required_cols[run_name] = self._get_required_source_columns(run.get("type", run_name))

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
                    f"[WARNING][DataPlotter] Failed to load run '{run_name}' from {file_path}: {exc}. "
                    "Skipping run."
                )
                self.run_required_cols.pop(run_name, None)
                continue

            self.run_filepaths[run_name] = file_path
            self.run_data[run_name] = data
            self.run_units[run_name] = units
            loaded_runs.append(run)

        if loaded_runs:
            self.runs = loaded_runs
        else:
            self.runs = []

        self._loaded = True
        return self.run_data

    def preprocess_data(self):
        """Apply mappings, transforms, calculated channels, and filters."""
        if not self._loaded:
            raise RuntimeError("Data must be loaded before preprocessing.")

        self._gated_data_cache.clear()
        self.apply_channel_mappings()
        self.apply_transformations()
        self.clean_data()
        self.apply_calculated_channels()
        self.apply_lowpass_filters()
        self.run_data_quality_checks()
        self._preprocessed = True
        return self.run_data

    # ------------------------------------------------------------
    # WAVEFORM PLOTS
    # ------------------------------------------------------------

    def _normalize_waveform_row_spec(self, row_spec):
        """Normalize a waveform row spec to (primary_channel, secondary_channel_or_None)."""
        if isinstance(row_spec, str):
            return row_spec, None

        if isinstance(row_spec, (list, tuple)):
            if len(row_spec) == 1 and isinstance(row_spec[0], str):
                return row_spec[0], None
            if len(row_spec) == 2 and all(isinstance(v, str) for v in row_spec):
                return row_spec[0], row_spec[1]

        raise ValueError(
            "Waveform channel row must be 'channel' or ('primary_channel', 'secondary_channel')."
        )

    def _normalize_waveform_axis_limits(self, raw_limits, has_secondary, row_name):
        """Normalize waveform y-limit config for one row."""
        if raw_limits is None:
            return None, None

        if not has_secondary:
            return raw_limits, None

        if (
            isinstance(raw_limits, (list, tuple))
            and len(raw_limits) == 2
            and all(isinstance(v, (list, tuple)) or v is None for v in raw_limits)
        ):
            return raw_limits[0], raw_limits[1]

        # print(  # Reduce verbosity
        #     f"[WARNING][DataPlotter] Waveform row '{row_name}': dual-channel row expects axis limits as "
        #     f"((y1_min,y1_max),(y2_min,y2_max)). Applying provided limits to primary channel only."
        # )
        return raw_limits, None

    def _normalize_waveform_reference_lines(self, raw_refs, has_secondary):
        """Normalize waveform reference-line config for one row."""
        if raw_refs is None:
            return None, None

        if not has_secondary:
            return raw_refs, None

        if isinstance(raw_refs, (list, tuple)) and len(raw_refs) == 2:
            return raw_refs[0], raw_refs[1]

        return raw_refs, None

    def _prepare_waveform_channels(self, channels, axis_limits, reference_lines, subplot_heights):
        """Build validated waveform rows with optional two-channel overlays."""
        prepared_rows = []
        row_heights = []

        for i, row_spec in enumerate(channels):
            primary, secondary = self._normalize_waveform_row_spec(row_spec)

            p_count = sum(primary in self.run_data[r["name"].lower()].columns for r in self.runs)
            if secondary is not None:
                s_count = sum(secondary in self.run_data[r["name"].lower()].columns for r in self.runs)
            else:
                s_count = 0

            if p_count == 0 and (secondary is None or s_count == 0):
                missing_name = (
                    f"'{primary}' and '{secondary}'" if secondary is not None else f"'{primary}'"
                )
                # print(f"[WARNING][DataPlotter] Waveform row {missing_name} missing from all runs. Skipping row.")  # Reduce verbosity
                continue

            if p_count == 0 and secondary is not None and s_count > 0:
                # print(  # Reduce verbosity
                #     f"[WARNING][DataPlotter] Waveform row primary channel '{primary}' missing in all runs; "
                #     f"using '{secondary}' as single-channel row."
                # )
                primary, secondary = secondary, None
                p_count = s_count
                s_count = 0

            if p_count < len(self.runs):
                pass  # print(  # Reduce verbosity
                #     f"[WARNING][DataPlotter] Waveform channel '{primary}' present in {p_count}/{len(self.runs)} runs. Plotting available runs only."
                # )

            if secondary is not None:
                if s_count == 0:
                    # print(  # Reduce verbosity
                    #     f"[WARNING][DataPlotter] Waveform secondary channel '{secondary}' missing from all runs; "
                    #     "rendering row as single-channel."
                    # )
                    secondary = None
                elif s_count < len(self.runs):
                    pass  # print(  # Reduce verbosity
                    #     f"[WARNING][DataPlotter] Waveform secondary channel '{secondary}' present in {s_count}/{len(self.runs)} runs. Plotting available runs only."
                    # )

            raw_lim = axis_limits[i] if axis_limits and i < len(axis_limits) else None
            raw_ref = reference_lines[i] if reference_lines and i < len(reference_lines) else None
            y1_lim, y2_lim = self._normalize_waveform_axis_limits(raw_lim, secondary is not None, primary)
            y1_refs, y2_refs = self._normalize_waveform_reference_lines(raw_ref, secondary is not None)

            row = {
                "primary": primary,
                "secondary": secondary,
                "y1_lim": y1_lim,
                "y2_lim": y2_lim,
                "y1_refs": y1_refs,
                "y2_refs": y2_refs,
            }
            prepared_rows.append(row)
            row_heights.append(subplot_heights[i] if subplot_heights and i < len(subplot_heights) else 1.0)

        return prepared_rows, row_heights

    def _format_waveform_channel_label(self, channel, *, secondary=False, show_style_hint=False):
        """Format waveform channel label and optional line-style hint."""
        base = datafunctions.add_units_to_label(channel, units_map=self.units_map)
        if not show_style_hint:
            return base
        style_hint = "- - - - -" if secondary else "_______"
        return f"{base}\n{style_hint}"

    def generate_waveform_plots(self):
        """Generate all configured waveform subplot figures."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(0)
        
        for plot_def in plots:
            # Optional x-axis limits (sLap zoom)
            x_limits = None

            if len(plot_def) == 4:
                plot_name, channels, axis_limits, ref_lines = plot_def
                subplot_heights = None
            elif len(plot_def) == 5:
                plot_name, channels, axis_limits, ref_lines, subplot_heights = plot_def
            elif len(plot_def) == 6:
                plot_name, channels, axis_limits, ref_lines, subplot_heights, x_limits = plot_def
            else:
                raise ValueError("Waveform plot definition malformed")


            # print(f"Creating waveform plot: {plot_name}")  # Reduce verbosity

            (prepared_rows, avail_heights) = \
                self._prepare_waveform_channels(channels, axis_limits, ref_lines, subplot_heights)

            if not prepared_rows:
                # print(f"  No valid channels for {plot_name}")  # Reduce verbosity
                continue

            filename = self._sanitize_plot_filename("waveform", plot_name)
            min_height = 1.6 * sum(avail_heights)
            figsize = self._resolve_plot_figsize(filename, self.waveform_figsize, min_height=min_height)

            fig, axes = plt.subplots(
                len(prepared_rows),
                1,
                figsize=figsize,
                sharex=True,
                squeeze=False,
                gridspec_kw={"height_ratios": avail_heights},
            )
            axes = axes.flatten()
            plotted_runs = set()

            xlabel = "sLap (m)" if all(
                "sLap" in self.run_data[r["name"].lower()].columns for r in self.runs
            ) else "Sample"

            # Draw channels
            for idx, row in enumerate(prepared_rows):
                ax = axes[idx]
                ch_primary = row["primary"]
                ch_secondary = row["secondary"]
                ax_right = ax.twinx() if ch_secondary is not None else None

                for run in self.runs:
                    rn = run["name"].lower()
                    if rn not in self.run_data:
                        continue

                    df = self.run_data[rn]
                    if ch_primary not in df.columns:
                        continue

                    x_vals = df["sLap"] if "sLap" in df.columns else df.index
                    y_vals = df[ch_primary]

                    x_plot, y_plot = datafunctions.mask_waveform_discontinuities(x_vals, y_vals)
                    ax.plot(
                        x_plot,
                        y_plot,
                        linewidth=1.6,
                        color=run["color"],
                        label=run["name"].upper(),
                        alpha=0.85,
                    )
                    plotted_runs.add(rn)

                    if ax_right is not None and ch_secondary in df.columns:
                        y2_vals = df[ch_secondary]
                        x2_plot, y2_plot = datafunctions.mask_waveform_discontinuities(x_vals, y2_vals)
                        ax_right.plot(
                            x2_plot,
                            y2_plot,
                            linewidth=1.45,
                            linestyle="--",
                            color=run["color"],
                            label="_nolegend_",
                            alpha=0.85,
                        )
                        plotted_runs.add(rn)

                ax.set_ylabel(
                    self._format_waveform_channel_label(
                        ch_primary,
                        secondary=False,
                        show_style_hint=(ch_secondary is not None),
                    ),
                    fontsize=8.2,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                )
                ax.yaxis.set_label_coords(-0.035, 0.5)
                ax.grid(True, axis="y", alpha=0.28, linewidth=0.45)

                if row["y1_lim"] is not None:
                    yl, yh = row["y1_lim"]
                    if yl is not None or yh is not None:
                        yl = yl if (yl is None or np.isscalar(yl)) else None
                        yh = yh if (yh is None or np.isscalar(yh)) else None
                        if yl is not None or yh is not None:
                            ax.set_ylim(bottom=yl, top=yh)

                if row["y1_refs"] is not None:
                    vals = row["y1_refs"]
                    if np.isscalar(vals):
                        vals = [vals]
                    for vv in vals:
                        ax.axhline(vv, linestyle="--", color="gray", alpha=0.4)

                if ax_right is not None:
                    ax_right.set_ylabel(
                        self._format_waveform_channel_label(
                            ch_secondary,
                            secondary=True,
                            show_style_hint=True,
                        ),
                        fontsize=8.2,
                        fontweight="bold",
                        rotation=0,
                        ha="left",
                        va="center",
                    )
                    ax_right.yaxis.set_label_coords(1.03, 0.5)
                    ax_right.spines["top"].set_visible(False)
                    ax_right.grid(False)
                    ax_right.tick_params(axis="y", labelsize=8.5)

                    if row["y2_lim"] is not None:
                        yl2, yh2 = row["y2_lim"]
                        if yl2 is not None or yh2 is not None:
                            yl2 = yl2 if (yl2 is None or np.isscalar(yl2)) else None
                            yh2 = yh2 if (yh2 is None or np.isscalar(yh2)) else None
                            if yl2 is not None or yh2 is not None:
                                ax_right.set_ylim(bottom=yl2, top=yh2)

                    if row["y2_refs"] is not None:
                        vals2 = row["y2_refs"]
                        if np.isscalar(vals2):
                            vals2 = [vals2]
                        for vv2 in vals2:
                            ax_right.axhline(vv2, linestyle=":", color="gray", alpha=0.36)

                if idx < len(prepared_rows) - 1:
                    ax.tick_params(labelbottom=False)

            # Style x-axis
            bottom = axes[-1]
            bottom.set_xlabel(xlabel, fontweight="bold")
            bottom.tick_params(axis="x", labelsize=10)

            # Apply optional sLap zoom window
            if x_limits is not None:
                xmin, xmax = x_limits
                if xmin is not None or xmax is not None:
                    for ax in axes:
                        ax.set_xlim(left=xmin, right=xmax)
            else:
                # Default auto-range behaviour
                if xlabel == "sLap (m)":
                    xmaxs = []
                    for ax in axes:
                        _, xm = ax.get_xlim()
                        if xm > 0:
                            xmaxs.append(xm)
                    if xmaxs:
                        xv = max(xmaxs)
                        xv = np.ceil(xv / 100) * 100
                        for ax in axes:
                            ax.set_xlim(0, xv)

                for ax in axes:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=5, steps=[1, 2, 2.5, 5, 10]))
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                    ax.grid(True, which="major", axis="x", alpha=0.45, linewidth=0.5)
                    ax.grid(True, which="minor", axis="x", alpha=0.225, linewidth=0.3)

            # Run legend (only once, outside data area)
            run_handles = []
            run_labels = []
            for run in self.runs:
                rn = run["name"].lower()
                if rn not in plotted_runs:
                    continue
                run_handles.append(Line2D([0], [0], color=run["color"], linewidth=2.0))
                run_labels.append(run["name"].upper())
            self._add_waveform_figure_legend(fig, run_handles, run_labels)

            plt.tight_layout(pad=0.3, h_pad=-0.4, rect=(0, 0, 1, 0.95))
            fig.savefig(self.plots_dir / filename, dpi=300, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # SCATTER PLOTS
    # ------------------------------------------------------------

    def generate_scatter_plots(self):
        """Generate all configured scatter plots and optional fit overlays."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(1)

        for plot_def in plots:
            show_equations = True  # Default: display equations
            show_error = True      # Default: display error box
            gate_spec = None

            if len(plot_def) == 4:
                plot_name, (x_var, y_var), axis_limits, best_fit = plot_def
            elif len(plot_def) == 5:
                plot_name, (x_var, y_var), axis_limits, best_fit, item5 = plot_def
                # 5th item can be gate_spec or show_equations (boolean)
                if isinstance(item5, bool):
                    show_equations = item5
                    gate_spec = None
                elif datafunctions.is_gate_spec(item5):
                    gate_spec = item5
                    show_equations = True
                else:
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 5th item must be gate_spec or boolean show_equations."
                    )
            elif len(plot_def) == 6:
                plot_name, (x_var, y_var), axis_limits, best_fit, item5, item6 = plot_def
                # Handle: gate_spec + show_equations, or two booleans
                if isinstance(item6, bool):
                    # Last item is boolean
                    if isinstance(item5, bool):
                        # Both are booleans: [name, (x,y), limits, best_fit, show_equations, show_error]
                        show_equations = item5
                        show_error = item6
                        gate_spec = None
                    else:
                        # item5 is gate_spec, item6 is show_equations or show_error
                        gate_spec = item5
                        show_equations = item6
                        show_error = True
                        if not datafunctions.is_gate_spec(gate_spec):
                            raise ValueError(
                                f"Scatter plot '{plot_name}': 5th item must be gate_spec when 6th item is boolean."
                            )
                elif isinstance(item5, bool):
                    # item5 is boolean, item6 is gate_spec
                    show_equations = item5
                    gate_spec = item6
                    show_error = True
                    if not datafunctions.is_gate_spec(gate_spec):
                        raise ValueError(
                            f"Scatter plot '{plot_name}': 6th item must be gate_spec when 5th item is boolean."
                        )
                else:
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 6-item format requires gate_spec + show_equations, or two booleans."
                    )
            elif len(plot_def) == 7:
                plot_name, (x_var, y_var), axis_limits, best_fit, item5, item6, item7 = plot_def
                # Standard 7-item: [name, (x,y), limits, best_fit, gate_spec, show_equations, show_error]
                gate_spec = item5
                show_equations = item6
                show_error = item7
                if not datafunctions.is_gate_spec(gate_spec):
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 5th item (gate_spec) must be a valid gate specification."
                    )
                if not isinstance(show_equations, bool) or not isinstance(show_error, bool):
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 6th (show_equations) and 7th (show_error) items must be booleans."
                    )
            else:
                raise ValueError(
                    f"Scatter plot definition for '{plot_def[0] if plot_def else 'unknown'}' must have 4-7 items"
                )

            # Backward-compatible fallback: treat None as no-fit scatter.
            if best_fit is None:
                print(
                    f"[WARNING][DataPlotter] Scatter plot '{plot_name}': best_fit=None interpreted as 0 (no fit)."
                )
                best_fit = 0

            print(f"Creating scatter plot: {plot_name} ({x_var} vs {y_var})")

            filename = self._sanitize_plot_filename("scatter", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.scatter_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)

            ax.set_xlabel(
                datafunctions.add_units_to_label(x_var, self.units_map),
                fontweight="bold",
                fontsize=14,
            )
            ax.set_ylabel(
                datafunctions.add_units_to_label(y_var, self.units_map),
                fontweight="bold",
                fontsize=14,
            )

            eq_list = []

            for run in self.runs:
                rn = run["name"].lower()
                if rn not in self.run_data:
                    continue

                df = self._get_filtered_run_dataframe(rn, gate_spec)
                if df is None:
                    continue

                if x_var not in df.columns or y_var not in df.columns:
                    print(
                        f"[WARNING][DataPlotter] Scatter plot '{plot_name}': missing '{x_var}' or '{y_var}' in run '{rn}'. Skipping run."
                    )
                    continue

                xy_index, x_values, y_values = self._prepare_scatter_xy(df, x_var, y_var)
                if x_values is None or y_values is None:
                    print(
                        f"[WARNING][DataPlotter] Scatter plot '{plot_name}': no valid points in run '{rn}'. Skipping run."
                    )
                    continue

                point_size, point_alpha = self._resolve_scatter_plot_style(len(x_values))

                if isinstance(best_fit, (list, tuple)) and best_fit and isinstance(best_fit[0], (list, tuple)):
                    fit_condition_data = datafunctions.build_fit_condition_data(
                        df,
                        xy_index,
                        best_fit,
                        plot_name=plot_name,
                        run_name=rn,
                    )
                    ok, slopes, intercepts, eq_text, color = datafunctions.plot_scatter_with_multi_fit(
                        ax, x_values, y_values,
                        run["name"].upper(), run["color"],
                        point_alpha, point_size,
                        x_var, y_var,
                        fit_defs=best_fit,
                        fit_condition_data=fit_condition_data,
                        render_mode=self.SCATTER_RENDER_MODE,
                        density_threshold=self.SCATTER_DENSITY_THRESHOLD,
                        max_points=self.SCATTER_MAX_POINTS,
                        hexbin_gridsize=self.SCATTER_HEXBIN_GRIDSIZE,
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slopes))

                elif best_fit == 0:
                    datafunctions.plot_scatter(
                        ax, x_values, y_values,
                        run["name"].upper(), run["color"],
                        point_alpha, point_size,
                        x_var, y_var,
                        render_mode=self.SCATTER_RENDER_MODE,
                        density_threshold=self.SCATTER_DENSITY_THRESHOLD,
                        max_points=self.SCATTER_MAX_POINTS,
                        hexbin_gridsize=self.SCATTER_HEXBIN_GRIDSIZE,
                    )

                elif best_fit == 1:
                    ok, slope, intercept, eq_text, color = datafunctions.plot_scatter_with_1fit(
                        ax, x_values, y_values,
                        run["name"].upper(), run["color"],
                        point_alpha, point_size,
                        x_var, y_var,
                        render_mode=self.SCATTER_RENDER_MODE,
                        density_threshold=self.SCATTER_DENSITY_THRESHOLD,
                        max_points=self.SCATTER_MAX_POINTS,
                        hexbin_gridsize=self.SCATTER_HEXBIN_GRIDSIZE,
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slope))

                elif best_fit == 2:
                    print(
                        f"[WARNING][DataPlotter] Scatter plot '{plot_name}': best_fit=2 uses removed fit_split behavior; falling back to single fit."
                    )
                    ok, slope, intercept, eq_text, color = datafunctions.plot_scatter_with_1fit(
                        ax, x_values, y_values,
                        run["name"].upper(), run["color"],
                        point_alpha, point_size,
                        x_var, y_var,
                        render_mode=self.SCATTER_RENDER_MODE,
                        density_threshold=self.SCATTER_DENSITY_THRESHOLD,
                        max_points=self.SCATTER_MAX_POINTS,
                        hexbin_gridsize=self.SCATTER_HEXBIN_GRIDSIZE,
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slope))

            # Axis limits
            has_x_limits = False
            has_y_limits = False
            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None or xmax is not None:
                    ax.set_xlim(left=xmin, right=xmax)
                    has_x_limits = True
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)
                    has_y_limits = True

            self._add_axis_edge_padding(
                ax,
                x_pad_ratio=(0 if has_x_limits else 0.02),
                y_pad_ratio=(0 if has_y_limits else 0.03),
            )

            # Axis lines
            xl, xr = ax.get_xlim()
            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)
            if xl <= 0 <= xr:
                ax.axvline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            ax.grid(True, alpha=0.26)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Trendline boxes and error box positioning
            anchor = None
            if eq_list:
                # Display equations if requested (this computes and returns anchor)
                if show_equations:
                    anchor = self._display_equations(ax, eq_list)
                
                # Display error box if requested
                if show_error:
                    fit_labels = None
                    if (
                        isinstance(best_fit, (list, tuple))
                        and best_fit
                        and isinstance(best_fit[0], (list, tuple))
                    ):
                        fit_labels = self._build_gradient_segment_labels(
                            best_fit, x_var=x_var, y_var=y_var
                        )
                    txt = self._format_gradient_error_text(
                        eq_list, x_var, y_var, fit_labels=fit_labels
                    )
                    if txt:
                        # If equations are not shown, still need anchor for error box positioning
                        if anchor is None:
                            x_anchor, y_anchor, halign, valign = self._select_trendline_anchor(ax, eq_list)
                            # Create a minimal anchor tuple for error box (without boxes list)
                            # We'll use a simple positioning: place error box at the anchor position
                            anchor = (x_anchor, halign, valign, [y_anchor])
                        self._display_gradient_error(ax, txt, anchor)

            # Legend first, then gate callout to avoid overlaps.
            legend = self._add_standard_legend(ax, loc="best")

            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    self._display_gate_info(
                        ax,
                        gate_text,
                        legend=legend,
                        trend_anchor=anchor,
                    )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, facecolor="white")
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # PSD PLOTS
    # ------------------------------------------------------------

    def generate_psd_plots(self):
        """Create PSD plots from definitions, skipping only runs with unavailable/invalid channel data."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(2)

        for plot_def in plots:
            # Parse plot definition
            if len(plot_def) == 3:
                plot_name, channel, axis_limits = plot_def
                log_scale = True
                nperseg = 512
            elif len(plot_def) == 4:
                plot_name, channel, axis_limits, log_scale = plot_def
                nperseg = 512
            elif len(plot_def) == 5:
                plot_name, channel, axis_limits, log_scale, nperseg = plot_def
            else:
                raise ValueError("Invalid PSD plot definition")

            print(f"Creating PSD plot: {plot_name} ({channel})")

            # Setup figure
            filename = self._sanitize_plot_filename("psd", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.psd_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
            ax.set_ylabel(
                datafunctions.format_psd_ylabel(channel, self.units_map),
                fontsize=13, fontweight='bold'
            )

            plotted_any = False

            # ---- RUN LOOP ----
            for run in self.runs:
                run_name = run['name'].lower()
                if run_name not in self.run_data:
                    print(f"[WARNING][DataPlotter] PSD plot '{plot_name}': run '{run_name}' has no loaded dataframe. Skipping run.")
                    continue
                df = self.run_data[run_name]

                if channel not in df.columns:
                    print(
                        f"[WARNING][DataPlotter] PSD plot '{plot_name}': channel '{channel}' missing in run '{run_name}'. Skipping run."
                    )
                    continue

                signal = df[channel]

                # SAFETY CHECKS: ensure valid signal type
                # ----------------------------------------
                if isinstance(signal, tuple):
                    print(
                        f"[WARNING][DataPlotter] PSD plot '{plot_name}': channel '{channel}' in run '{run_name}' has invalid tuple type. Skipping run."
                    )
                    continue

                if not isinstance(signal, (pd.Series, np.ndarray, list)):
                    print(
                        f"[WARNING][DataPlotter] PSD plot '{plot_name}': channel '{channel}' in run '{run_name}' has unsupported type {type(signal)}. Skipping run."
                    )
                    continue

                # Convert to Series for safe processing
                signal = np.asarray(signal, dtype=float)

                # Must be numeric
                if not np.issubdtype(signal.dtype, np.number):
                    print(
                        f"[WARNING][DataPlotter] PSD plot '{plot_name}': channel '{channel}' in run '{run_name}' is non-numeric. Skipping run."
                    )
                    continue

                # Compute PSD
                freq, power = datafunctions.calculate_psd(
                    signal,
                    self.FILTER_SAMPLE_RATE,
                    nperseg=nperseg
                )

                if freq is None:
                    print(
                        f"[WARNING][DataPlotter] PSD plot '{plot_name}': not enough data for channel '{channel}' in run '{run_name}'. Skipping run."
                    )
                    continue

                # Plot PSD
                plot_func = ax.semilogy if log_scale else ax.plot
                plot_func(freq, power,
                          linewidth=1.8,
                          color=run['color'],
                          alpha=0.9,
                          label=run['name'].upper())
                plotted_any = True

            if not plotted_any:
                print(
                    f"[WARNING][DataPlotter] PSD plot '{plot_name}': no valid runs available for channel '{channel}'. Plot not saved."
                )
                plt.close(fig)
                continue

            # Axis limits
            has_x_limits = False
            has_y_limits = False
            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None or xmax is not None:
                    ax.set_xlim(left=xmin, right=xmax)
                    has_x_limits = True
                if ymin is not None or ymax is not None:
                    if log_scale and ymin is not None:
                        ymin = max(ymin, 1e-4)  # avoid log(0) issues
                    ax.set_ylim(bottom=ymin, top=ymax)
                    has_y_limits = True

            # Padding & styling
            default_y_pad = 0 if log_scale else 0.04
            self._add_axis_edge_padding(
                ax,
                x_pad_ratio=(0 if has_x_limits else 0.02),
                y_pad_ratio=(0 if has_y_limits else default_y_pad),
            )
            ax.grid(True, which='major', alpha=0.3)
            ax.grid(True, which='minor', alpha=0.15)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Legend
            self._add_standard_legend(ax, loc="best")

            plt.tight_layout(pad=0.25)
            fig.savefig(
                self.plots_dir / filename,
                dpi=300,
                pad_inches=0.05,
                facecolor='white'
            )
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # HISTOGRAM PLOTS
    # ------------------------------------------------------------

    def generate_histogram_plots(self):
        """Create histogram plots based on HISTOGRAM_PLOT_DEFINITIONS"""
        self._ensure_preprocessed()
        plots = self._get_plot_group(3)

        for plot_def in plots:
            plot_name, channel, axis_limits, log_scale = plot_def
            print(f"Creating histogram plot: {plot_name} ({channel})")

            filename = self._sanitize_plot_filename("histogram", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.histogram_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontsize=13, fontweight='bold'
            )
            ax.set_ylabel('Time (s)', fontsize=13, fontweight='bold')

            # ---- Collect data from all runs (for shared bins) ----
            all_data = []

            for run in self.runs:
                run_name = run['name'].lower()
                df = self.run_data[run_name]

                if channel not in df.columns:
                    continue

                vals = df[channel].dropna()
                if not vals.empty:
                    all_data.append(vals.values)

            if not all_data:
                print(
                    f"[WARNING][DataPlotter] Histogram plot '{plot_name}': no valid data for channel '{channel}'. Plot not saved."
                )
                continue

            all_data = np.concatenate(all_data)

            # ---- Compute shared bins ----
            num_bins = 30
            bins = datafunctions.compute_nice_histogram_bins(all_data, num_bins=num_bins)

            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None or xmax is not None:
                    ax.set_xlim(left=xmin, right=xmax)
                if xmin is not None and xmax is not None:
                    bins = datafunctions.compute_equal_width_bins_in_limits(xmin, xmax, bins)
                if ymin is not None or ymax is not None:
                    if log_scale and ymin is not None:
                        ymin = max(ymin, 1e-6)  # avoid log(0) issues
                    ax.set_ylim(bottom=ymin, top=ymax)

            # ---- Plot using shared bins ----
            histogram_data = []
            histogram_weights = []
            histogram_colors = []
            histogram_labels = []
            dt = 1.0 / self.FILTER_SAMPLE_RATE

            for run in self.runs:
                run_name = run['name'].lower()
                df = self.run_data[run_name]

                if channel not in df.columns:
                    continue

                data = df[channel].dropna()
                if data.empty:
                    continue

                histogram_data.append(data.to_numpy())
                histogram_weights.append(np.full(len(data), dt))
                histogram_colors.append(run['color'])
                histogram_labels.append(run['name'].upper())

            if histogram_data:
                ax.hist(
                    histogram_data,
                    bins=bins,
                    weights=histogram_weights,
                    alpha=0.9,
                    color=histogram_colors,
                    label=histogram_labels,
                    edgecolor='black',
                    linewidth=0.5,
                    log=log_scale,
                    stacked=False,
                    histtype='bar',
                    rwidth=0.9
                )

            if len(bins) > 1:
                max_major_ticks = 8
                major_step = max(1, int(np.ceil((len(bins) - 1) / (max_major_ticks - 1))))
                major_ticks = bins[::major_step]
                if not np.isclose(major_ticks[-1], bins[-1]):
                    major_ticks = np.append(major_ticks, bins[-1])

                ax.set_xticks(major_ticks)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f"{x:.4g}")
                )

                if len(bins) <= 31:
                    ax.set_xticks(bins, minor=True)
                    ax.grid(True, which='minor', axis='x', alpha=0.12, linewidth=0.3)

                ax.grid(True, which='major', axis='x', alpha=0.22, linewidth=0.45)


            # Padding & styling
            has_x_limits = bool(
                axis_limits and (axis_limits[0][0] is not None or axis_limits[0][1] is not None)
            )
            has_y_limits = bool(
                axis_limits and (axis_limits[1][0] is not None or axis_limits[1][1] is not None)
            )
            self._add_axis_edge_padding(
                ax,
                x_pad_ratio=(0 if has_x_limits else 0.02),
                y_pad_ratio=(0 if has_y_limits else 0.03),
            )
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Zero-line (if within axis limits)
            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            # Legend
            self._add_standard_legend(ax, loc="best")

            plt.tight_layout(pad=0.25)
            fig.savefig(
                self.plots_dir / filename,
                dpi=300,
                pad_inches=0.05,
                facecolor='white'
            )
            plt.close(fig)
            print(f"  Saved: {filename}")


    # ------------------------------------------------------------
    # BAR PLOTS
    # ------------------------------------------------------------

    def generate_bar_plots(self):
        """Create grouped bar plots for aggregated channel metrics."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(4)

        for plot_def in plots:
            # Format:
            # ["Name", ("ch1", "ch2", ...)]
            # ["Name", (("ch1", "integral"), ("ch2", "sum")), default_agg(optional), (ymin, ymax)(optional)]
            plot_name = plot_def[0]
            metric_specs_raw = plot_def[1] if len(plot_def) > 1 else ()
            default_agg = plot_def[2] if len(plot_def) > 2 and isinstance(plot_def[2], str) else "last"
            axis_limits = plot_def[3] if len(plot_def) > 3 else None

            metric_specs = datafunctions.normalize_bar_metric_specs(
                metric_specs_raw,
                default_aggregation=default_agg,
            )

            if not metric_specs:
                print(f"[WARNING][DataPlotter] Bar plot '{plot_name}' has no valid metric specs. Skipping.")
                continue

            filename = self._sanitize_plot_filename("bar", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.histogram_FIGSIZE)
            fig, ax = plt.subplots(figsize=figsize)

            metric_labels = [metric for metric, _ in metric_specs]
            x = np.arange(len(metric_specs))

            loaded_runs = [run for run in self.runs if run["name"].lower() in self.run_data]
            if not loaded_runs:
                print(f"[WARNING][DataPlotter] Bar plot '{plot_name}' has no loaded runs. Skipping.")
                plt.close(fig)
                continue

            group_width = 0.82
            bar_width = group_width / max(len(loaded_runs), 1)
            left_edge = -group_width / 2.0

            run_bar_data = []
            all_values = []

            for run_index, run in enumerate(loaded_runs):
                run_name = run["name"].lower()
                df = self.run_data[run_name]

                values = []
                for channel, aggregation in metric_specs:
                    if channel not in df.columns:
                        print(
                            f"[WARNING][DataPlotter] Bar plot '{plot_name}': missing channel '{channel}' "
                            f"in run '{run_name.upper()}'."
                        )
                        values.append(np.nan)
                        continue

                    metric_value = datafunctions.aggregate_channel_for_bar(
                        df[channel],
                        aggregation=aggregation,
                        sample_rate=self.FILTER_SAMPLE_RATE,
                        time_series=df['tLap'] if 'tLap' in df.columns else None,
                    )
                    values.append(metric_value)

                offsets = x + left_edge + (run_index + 0.5) * bar_width
                run_bar_data.append(
                    {
                        "run": run,
                        "offsets": offsets,
                        "values": np.array(values, dtype=float),
                    }
                )
                all_values.extend([abs(v) for v in values if not np.isnan(v)])

            ax2 = None
            secondary_threshold = None
            if len(all_values) > 1:
                max_abs = max(all_values)
                candidate_secondary_threshold = max_abs / max(1.0, self.BAR_SECONDARY_AXIS_RATIO)
                lower_group = [v for v in all_values if v < candidate_secondary_threshold]
                if lower_group:
                    max_lower = max(lower_group)
                    if max_lower > 0 and max_abs / max_lower >= self.BAR_SECONDARY_AXIS_RATIO:
                        ax2 = ax.twinx()
                        ax2.spines["right"].set_visible(True)
                        ax2.spines["right"].set_color("black")
                        ax2.spines["right"].set_linewidth(2.0)
                        ax2.tick_params(axis="y", labelsize=10, colors="black", width=1.5)
                        secondary_threshold = candidate_secondary_threshold

            plotted_labels = set()
            bar_info = []
            for item in run_bar_data:
                run = item["run"]
                offsets = item["offsets"]
                values = item["values"]
                run_label = run["name"].upper()

                if ax2 is not None:
                    primary_values = []
                    secondary_values = []
                    for value in values:
                        if np.isnan(value):
                            primary_values.append(0.0)
                            secondary_values.append(0.0)
                        elif abs(value) >= secondary_threshold:
                            primary_values.append(0.0)
                            secondary_values.append(value)
                        else:
                            primary_values.append(value)
                            secondary_values.append(0.0)

                    primary_label = run_label if run_label not in plotted_labels and any(v != 0.0 for v in primary_values) else "_nolegend_"
                    if primary_label != "_nolegend_":
                        plotted_labels.add(primary_label)
                    ax.bar(
                        offsets,
                        primary_values,
                        width=bar_width,
                        color=run["color"],
                        alpha=0.9,
                        label=primary_label,
                        edgecolor="white",
                        linewidth=0.6,
                    )

                    secondary_label = run_label if run_label not in plotted_labels and any(v != 0.0 for v in secondary_values) else "_nolegend_"
                    if secondary_label != "_nolegend_":
                        plotted_labels.add(secondary_label)
                    ax2.bar(
                        offsets,
                        secondary_values,
                        width=bar_width,
                        color=run["color"],
                        alpha=0.9,
                        label=secondary_label,
                        edgecolor="white",
                        linewidth=0.6,
                    )

                    for offset, value in zip(offsets, values):
                        axis = ax2 if not np.isnan(value) and abs(value) >= secondary_threshold else ax
                        bar_info.append((offset, value, axis))
                else:
                    ax.bar(
                        offsets,
                        values,
                        width=bar_width,
                        color=run["color"],
                        alpha=0.9,
                        label=run_label if run_label not in plotted_labels else "_nolegend_",
                        edgecolor="white",
                        linewidth=0.6,
                    )
                    plotted_labels.add(run_label)
                    for offset, value in zip(offsets, values):
                        bar_info.append((offset, value, ax))

            ax.set_xticks(x)
            metric_labels = [f"{metric}\n({aggregation})" for metric, aggregation in metric_specs]
            ax.set_xticklabels(metric_labels, rotation=0, fontweight="bold")
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            if ax2 is not None:
                ax2.spines["right"].set_visible(True)
                ax2.spines["right"].set_color("black")
                ax2.spines["right"].set_linewidth(2.0)
                ax2.tick_params(axis="y", labelsize=10, colors="black", width=1.5, grid_linestyle="--")

            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)

            self._add_axis_edge_padding(ax, x_pad_ratio=0.06, y_pad_ratio=0.04)
            if ax2 is not None:
                self._add_axis_edge_padding(ax2, x_pad_ratio=0.06, y_pad_ratio=0.04)

            axis_ranges = {ax: ax.get_ylim()[1] - ax.get_ylim()[0]}
            if ax2 is not None:
                axis_ranges[ax2] = ax2.get_ylim()[1] - ax2.get_ylim()[0]

            # Add numeric labels above/below bars
            for offset, value, axis in bar_info:
                if not np.isnan(value):
                    y_range = axis_ranges.get(axis, ax.get_ylim()[1] - ax.get_ylim()[0])
                    padding = 0.02 * y_range
                    y_pos = value + (padding if value >= 0 else -padding)
                    va = 'bottom' if value >= 0 else 'top'
                    axis.text(offset, y_pos, f'{value:.2f}', ha='center', va=va, fontsize=10, fontweight='bold', color='black')

            for axis in (ax2, ax) if ax2 is not None else (ax,):
                y0, y1 = axis.get_ylim()
                if y0 <= 0 <= y1:
                    axis.axhline(
                        0,
                        color="#4F4F4F",
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.9,
                        zorder=1,
                    )

            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if ax2 is not None:
                ax2.grid(True, axis="y", alpha=0.2)
                ax2.set_axisbelow(True)

            handles = []
            labels = []
            for axis in (ax, ax2) if ax2 is not None else (ax,):
                h, l = axis.get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label and label != "_nolegend_" and label not in labels:
                        handles.append(handle)
                        labels.append(label)

            self._add_standard_legend(ax, handles=handles, labels=labels, loc="upper right")

            plt.tight_layout(pad=0.25)
            fig.savefig(
                self.plots_dir / filename,
                dpi=300,
                pad_inches=0.05,
                facecolor="white",
            )
            plt.close(fig)
            print(f"  Saved: {filename}")



    # ------------------------------------------------------------
    # TRENDLINE & GRADIENT BOXES
    # ------------------------------------------------------------

    def _select_trendline_anchor(self, ax, equations_list):
        """Place text in the least-crowded corner."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xs = []
        ys = []
        for _, _, _, xv, yv, _ in equations_list:
            xs.extend(xv)
            ys.extend(yv)

        xs = np.array(xs)
        ys = np.array(ys)

        corners = {
            "tl": (0.03, 0.97, "left", "top"),
            "tr": (0.97, 0.97, "right", "top"),
            "bl": (0.03, 0.03, "left", "bottom"),
            "br": (0.97, 0.03, "right", "bottom"),
        }

        def count(corner):
            """Count points that fall inside a candidate annotation box."""
            xa, ya, hal, val = corners[corner]
            # define box size
            w = (x1 - x0) * 0.22
            h = (y1 - y0) * 0.28

            if hal == "left":
                x_min = x0 + xa * (x1 - x0)
                x_max = x_min + w
            else:
                x_max = x0 + xa * (x1 - x0)
                x_min = x_max - w

            if val == "top":
                y_max = y0 + ya * (y1 - y0)
                y_min = y_max - h
            else:
                y_min = y0 + ya * (y1 - y0)
                y_max = y_min + h

            inside = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
            return inside.sum()

        # choose corner with fewest points
        best = min(corners.keys(), key=count)
        return corners[best]

    def _format_trendline_text(self, label, equation):
        """Normalize trendline text and ensure each line carries the run label."""
        lines = []
        for line in str(equation).splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            prefix = f"{label} "
            if not cleaned.startswith(prefix):
                cleaned = f"{prefix}{cleaned}"
            lines.append(cleaned)
        return "\n".join(lines) if lines else f"{label} fit unavailable"

    def _colorize_legend_labels(self, legend):
        """Match legend text color to the corresponding series color."""
        if legend is None:
            return

        for text, handle in zip(legend.get_texts(), legend.legend_handles):
            color = None

            # Line-based legend handles (e.g., waveform/PSD)
            if hasattr(handle, "get_color") and not isinstance(handle, Patch):
                color = handle.get_color()
                if isinstance(color, (list, tuple, np.ndarray)):
                    if len(color) == 0:
                        color = None
                    elif isinstance(color[0], (list, tuple, np.ndarray)):
                        color = color[0]

            # Patch-based legend handles (e.g., histogram)
            elif isinstance(handle, Patch):
                fc = handle.get_facecolor()
                if isinstance(fc, (list, tuple, np.ndarray)) and len(fc) >= 3:
                    color = fc[:3]

            # Collection handles (e.g., scatter PathCollection)
            if color is None and hasattr(handle, "get_facecolor"):
                fc = handle.get_facecolor()
                if isinstance(fc, np.ndarray) and fc.size > 0:
                    color = fc[0]
                elif isinstance(fc, (list, tuple)) and len(fc) > 0:
                    color = fc[0] if isinstance(fc[0], (list, tuple, np.ndarray)) else fc

            if color is not None:
                text.set_color(color)

    def _add_standard_legend(self, ax, handles=None, labels=None, loc="best", bbox_to_anchor=None, ncol=1):
        """Add a consistently styled axis legend and colorize labels."""
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return None

        legend = ax.legend(
            handles,
            labels,
            fontsize=10,
            framealpha=1,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            borderpad=0.35,
            handlelength=1.8,
            ncol=ncol,
            prop={"family": "Montserrat", "weight": "bold", "size": 12},
        )
        self._colorize_legend_labels(legend)
        return legend

    def _add_waveform_figure_legend(self, fig, handles, labels):
        """Place waveform legend above subplots to avoid covering trace data."""
        if not handles:
            return None

        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=max(1, min(len(handles), 5)),
            framealpha=1,
            borderpad=0.35,
            handlelength=1.8,
            prop={"family": "Montserrat", "weight": "bold", "size": 11},
        )
        self._colorize_legend_labels(legend)
        return legend



    def _display_equations(self, ax, eq_list):
        """Render trendline equation callouts and return their anchor metadata."""
        x_anchor, y_anchor, halign, valign = self._select_trendline_anchor(ax, eq_list)
        line_height = 0.042
        box_gap = 0.018
        boxes = []
        cursor = y_anchor

        for i, (label, equation, color, _, _, _) in enumerate(eq_list):
            text = self._format_trendline_text(label, equation)
            line_count = max(1, len(text.splitlines()))
            box_height = line_count * line_height

            if valign == "top":
                ypos = cursor
                cursor -= (box_height + box_gap)
            else:
                ypos = cursor
                cursor += (box_height + box_gap)

            ax.text(
                x_anchor,
                ypos,
                text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment=valign,
                horizontalalignment=halign,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=1.6,
                ),
                color=color,
                fontweight="bold",
                family="Montserrat",
            )
            boxes.append(ypos)

        return x_anchor, halign, valign, boxes
    def _format_gradient_error_text(
        self, equations_list, x_var=None, y_var=None, fit_labels=None
    ):
        """
        Create baseline-relative gradient error text.
        The first run in RUNS is treated as baseline and excluded from listed rows.
        """
        if len(equations_list) < 2:
            return None

        baseline_target = self.runs[0]["name"].upper() if self.runs else None
        baseline_entry = next(
            (entry for entry in equations_list if entry[0].upper() == baseline_target),
            equations_list[0],
        )
        baseline_label, _, _, _, _, baseline_slopes = baseline_entry
        comparison_entries = [entry for entry in equations_list if entry is not baseline_entry]
        if not comparison_entries:
            return None
        ordered_entries = comparison_entries

        def percent_error(value, baseline):
            """Compute percentage difference vs baseline slope."""
            if value is None or baseline is None or baseline == 0:
                return None
            return ((value - baseline) / baseline) * 100

        def fmt(value):
            """Format percent values while preserving undefined states."""
            return "undefined" if value is None else f"{value:+.1f}%"

        lines = [f"Gradient Error (%) vs {baseline_label.upper()}:"]
        label_width = max(len(entry[0].upper()) for entry in ordered_entries)

        if isinstance(baseline_slopes, tuple):
            segment_count = len(baseline_slopes)
            for idx in range(segment_count):
                if fit_labels and idx < len(fit_labels):
                    segment_name = fit_labels[idx]
                else:
                    segment_name = f"Segment {idx + 1}"

                lines.append(f"For {segment_name}:")
                base_val = baseline_slopes[idx] if idx < len(baseline_slopes) else None

                for label, _, _, _, _, run_slopes in ordered_entries:
                    run_val = (
                        run_slopes[idx]
                        if isinstance(run_slopes, tuple) and idx < len(run_slopes)
                        else None
                    )
                    lines.append(
                        f"  {label.upper():<{label_width}} : {fmt(percent_error(run_val, base_val))}"
                    )
        else:
            lines.append("Overall:")
            for label, _, _, _, _, run_slopes in ordered_entries:
                lines.append(
                    f"  {label.upper():<{label_width}} : {fmt(percent_error(run_slopes, baseline_slopes))}"
                )

        return "\n".join(lines)

    def _display_gradient_error(self, ax, text, anchor):
        """Render slope-error callout below the equation boxes."""
        if anchor is None:
            return

        x_anchor, halign, valign, boxes = anchor
        offset = 0.06

        if valign == "top":
            ypos = min(boxes) - offset
        else:
            ypos = max(boxes) + offset

        ax.text(
            x_anchor,
            ypos,
            text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment=valign,
            horizontalalignment=halign,
            bbox=dict(
                boxstyle="round,pad=0.26",
                facecolor="white",
                alpha=0.9,
                edgecolor="#6E6E6E",
                linewidth=1.2,
            ),
            color="#3F3F3F",
            fontweight="bold",
            family="Montserrat",
        )

    def _display_gate_info(self, ax, text, legend=None, trend_anchor=None):
        """Render gate condition callout while avoiding legend/trendline overlap."""
        candidates = [
            (0.03, 0.97, "left", "top"),
            (0.97, 0.97, "right", "top"),
            (0.03, 0.03, "left", "bottom"),
            (0.97, 0.03, "right", "bottom"),
        ]

        # Avoid the trendline corner entirely when present.
        if trend_anchor is not None:
            _, trend_halign, trend_valign, _ = trend_anchor
            candidates = [
                c for c in candidates if not (c[2] == trend_halign and c[3] == trend_valign)
            ] or candidates

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer) if legend is not None else None

        def _candidate_overlaps_legend(candidate):
            """Check whether a candidate gate-info box collides with the legend."""
            if legend_bbox is None:
                return False
            xa, ya, hal, val = candidate
            probe = ax.text(
                xa,
                ya,
                text,
                transform=ax.transAxes,
                fontsize=9.5,
                verticalalignment=val,
                horizontalalignment=hal,
                bbox=dict(boxstyle="round,pad=0.26"),
                visible=False,
            )
            bbox = probe.get_window_extent(renderer)
            probe.remove()
            return bbox.overlaps(legend_bbox)

        chosen = candidates[0]
        for candidate in candidates:
            if not _candidate_overlaps_legend(candidate):
                chosen = candidate
                break

        x_anchor, y_anchor, halign, valign = chosen
        ax.text(
            x_anchor,
            y_anchor,
            text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment=valign,
            horizontalalignment=halign,
            bbox=dict(
                boxstyle="round,pad=0.26",
                facecolor="white",
                alpha=0.92,
                edgecolor="#5E5E5E",
                linewidth=1.1,
            ),
            color="#333333",
            fontweight="bold",
            family="Montserrat",
        )

    # ------------------------------------------------------------
    # BOX PLOTS
    # ------------------------------------------------------------

    def _parse_boxplot_definition(self, plot_def):
        """Normalize a box-plot definition into a consistent structure."""
        if not isinstance(plot_def, (list, tuple)) or len(plot_def) < 4:
            raise ValueError("Box plot definitions must have at least 4 items.")

        plot_name = plot_def[0]
        channels = plot_def[1]
        aggregation_mode = plot_def[2] if len(plot_def) > 2 else "per_run"
        axis_limits = plot_def[3] if len(plot_def) > 3 else None
        gate_spec = None
        options = {}

        if len(plot_def) > 4:
            item5 = plot_def[4]
            item6 = plot_def[5] if len(plot_def) > 5 else None

            if datafunctions.is_gate_spec(item5):
                gate_spec = item5
                if isinstance(item6, dict):
                    options = item6
            elif isinstance(item5, dict):
                options = item5
                if datafunctions.is_gate_spec(item6):
                    gate_spec = item6
            elif item5 is None:
                if datafunctions.is_gate_spec(item6):
                    gate_spec = item6
                elif isinstance(item6, dict):
                    options = item6
            else:
                # Keep the parser forgiving: unknown 5th item is treated as gate_spec
                # only if it matches the supported gate shape.
                if datafunctions.is_gate_spec(item5):
                    gate_spec = item5
                elif isinstance(item5, dict):
                    options = item5

        if channels is None:
            channels = []
        elif isinstance(channels, str):
            channels = [channels]
        else:
            channels = list(channels)

        if not isinstance(options, dict):
            options = {}

        return plot_name, channels, aggregation_mode, axis_limits, gate_spec, options

    def _collect_boxplot_point_series(self, channel, gate_spec=None):
        """Collect per-run point series for optional box-plot overlays."""
        series = []
        for run in self.runs:
            run_name = run["name"].lower()
            df = self.run_data.get(run_name)
            if df is None:
                continue

            filtered_df = datafunctions.apply_gate_to_dataframe(df, gate_spec) if gate_spec is not None else df
            if channel not in filtered_df.columns:
                continue

            values = pd.to_numeric(filtered_df[channel], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue

            series.append((run["name"].upper(), run["color"], values))

        return series

    def _collect_boxplot_point_series_from_data(self, channel, filtered_run_data):
        """Collect per-run point series using prefiltered run data."""
        series = []
        for run in self.runs:
            run_name = run["name"].lower()
            df = filtered_run_data.get(run_name)
            if df is None or channel not in df.columns:
                continue

            values = pd.to_numeric(df[channel], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue

            series.append((run["name"].upper(), run["color"], values))

        return series

    def _apply_boxplot_artist_style(self, bp, box_settings, colors=None, facecolor=None, alpha=0.7):
        """Apply consistent styling to a matplotlib boxplot result."""
        box_linewidth = box_settings.get("box_linewidth", 1.5)
        median_color = box_settings.get("medianline_color", "#000000")
        median_width = box_settings.get("medianline_width", 2.0)
        whisker_color = box_settings.get("box_edge_color", "#4A4A4A")
        cap_color = box_settings.get("box_edge_color", "#4A4A4A")

        for idx, patch in enumerate(bp.get("boxes", [])):
            patch.set_linewidth(box_linewidth)
            if colors and idx < len(colors):
                patch.set_facecolor(colors[idx])
            elif facecolor is not None:
                patch.set_facecolor(facecolor)
            patch.set_alpha(alpha)

        for item in bp.get("whiskers", []):
            item.set(color=whisker_color, linewidth=box_linewidth)

        for item in bp.get("caps", []):
            item.set(color=cap_color, linewidth=box_linewidth)

        for median in bp.get("medians", []):
            median.set(color=median_color, linewidth=median_width)

        for flier in bp.get("fliers", []):
            flier.set(markerfacecolor=median_color, markeredgecolor=median_color, alpha=0.7)

    def generate_box_plots(self):
        """Generate box plots for distribution analysis across runs."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(5)

        if not plots:
            return

        box_settings = getattr(self, "BOX_PLOT_SETTINGS", {})

        for plot_def in plots:
            try:
                plot_name, channels, aggregation_mode, axis_limits, gate_spec, options = self._parse_boxplot_definition(plot_def)
            except ValueError as exc:
                print(f"[WARNING][DataPlotter] {exc} Skipping box plot definition: {plot_def!r}")
                continue

            if not channels:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no channels specified. Skipping.")
                continue

            plot_options = dict(box_settings)
            plot_options.update(options or {})

            print(f"Creating box plot: {plot_name}")

            # Determine figure size
            num_channels = len(channels)
            if num_channels == 1:
                figsize = box_settings.get("figsize_single_channel", self.boxplot_FIGSIZE)
            else:
                figsize = box_settings.get("figsize_multi_channel", self.boxplot_FIGSIZE)

            # Generate plot based on mode
            if aggregation_mode == "per_run":
                self._generate_boxplot_per_run(
                    plot_name, channels, axis_limits, gate_spec, plot_options, figsize
                )
            elif aggregation_mode == "aggregated":
                self._generate_boxplot_aggregated(
                    plot_name, channels, axis_limits, gate_spec, plot_options, figsize
                )
            else:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}': unknown aggregation_mode '{aggregation_mode}'. Skipping.")

    def _generate_boxplot_per_run(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
        """Generate per-run box plots (one box per run per channel)."""
        box_settings = dict(self.BOX_PLOT_SETTINGS or {})
        box_settings.update(options or {})
        filtered_run_data = {
            run_name: self._get_filtered_run_dataframe(run_name, gate_spec)
            for run_name in self.run_data
        }
        # Aggregate data
        agg_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data,
            channels,
            aggregation_mode='per_run',
            gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )

        if not agg_data:
            print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no data after aggregation. Skipping.")
            return

        # Get run names in order
        run_names = [run['name'].lower() for run in self.runs if run['name'].lower() in agg_data]
        if not run_names:
            print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no valid runs. Skipping.")
            return

        # Build run color map
        run_colors = {run['name'].lower(): run['color'] for run in self.runs}
        show_points = bool(box_settings.get("show_points", False))
        show_fliers = bool(box_settings.get("show_fliers", True))
        point_alpha = float(box_settings.get("point_alpha", 0.25))
        point_size = float(box_settings.get("point_size", 18))
        jitter = float(box_settings.get("jitter", 0.15))
        box_width = float(box_settings.get("box_width", 0.6))
        gate_text = datafunctions.format_gate_text(gate_spec) if gate_spec is not None else None

        # Create figure with subplots (one per channel if multiple)
        num_channels = len(channels)
        if num_channels == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        else:
            fig, axes = plt.subplots(
                num_channels,
                1,
                figsize=(figsize[0], max(figsize[1], 4.5) * num_channels * 0.62),
                sharex=False,
            )
            axes = list(np.atleast_1d(axes))

        fig.suptitle(plot_name, fontsize=16, fontweight="bold")
        rng = np.random.default_rng(42)

        for ax_index, (ax, channel) in enumerate(zip(axes, channels)):
            # Prepare data for boxplot
            data_list = []
            labels_list = []
            colors_list = []
            overlay_series = []

            for run_name in run_names:
                if channel in agg_data[run_name]:
                    values = agg_data[run_name][channel]
                    if len(values) > 0:
                        data_list.append(values)
                        labels_list.append(run_name.upper())
                        colors_list.append(run_colors.get(run_name, '#3498DB'))
                        if show_points:
                            overlay_series.append(
                                (run_name.upper(), run_colors.get(run_name, "#3498DB"), values)
                            )

            if not data_list:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}' channel '{channel}': no data for any runs. Skipping subplot.")
                continue

            # Create box plot
            bp = ax.boxplot(
                data_list,
                labels=labels_list,
                patch_artist=True,
                widths=box_width,
                showfliers=show_fliers,
            )

            self._apply_boxplot_artist_style(
                bp,
                box_settings,
                colors=colors_list,
                alpha=box_settings.get("per_run_box_alpha", 0.7),
            )

            # Axis labels
            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight='bold', fontsize=12
            )
            ax.set_title(channel, fontweight="bold", fontsize=12)

            # Apply axis limits if specified
            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)

            # Styling
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Zero line if within limits
            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            if show_points:
                for box_index, (_, color, values) in enumerate(overlay_series, start=1):
                    x_points = np.full(len(values), box_index, dtype=float)
                    x_points += rng.uniform(-jitter, jitter, size=len(values))
                    ax.scatter(
                        x_points,
                        values,
                        s=point_size,
                        alpha=point_alpha,
                        color=color,
                        edgecolors="none",
                        zorder=3,
                    )

        if gate_text:
            self._display_gate_info(axes[0], gate_text)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.05, facecolor='white')
        plt.close(fig)
        print(f"  Saved: {filename}")

    def _generate_boxplot_aggregated(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
        """Generate aggregated box plots (all runs combined into single/multiple boxes)."""
        box_settings = dict(self.BOX_PLOT_SETTINGS or {})
        box_settings.update(options or {})
        filtered_run_data = {
            run_name: self._get_filtered_run_dataframe(run_name, gate_spec)
            for run_name in self.run_data
        }
        # Aggregate data
        agg_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data,
            channels,
            aggregation_mode='aggregated',
            gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )

        if not agg_data:
            print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no data after aggregation. Skipping.")
            return

        show_points = bool(box_settings.get("show_points", False))
        show_fliers = bool(box_settings.get("show_fliers", True))
        point_alpha = float(box_settings.get("point_alpha", 0.25))
        point_size = float(box_settings.get("point_size", 18))
        jitter = float(box_settings.get("jitter", 0.15))
        box_width = float(box_settings.get("box_width", 0.6))
        aggregated_box_color = box_settings.get("aggregated_box_color", "#3498DB")
        aggregated_box_alpha = float(box_settings.get("aggregated_box_alpha", 0.7))
        gate_text = datafunctions.format_gate_text(gate_spec) if gate_spec is not None else None

        # Create figure with subplots (one per channel if multiple)
        num_channels = len(channels)
        if num_channels == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        else:
            fig, axes = plt.subplots(
                num_channels,
                1,
                figsize=(figsize[0], max(figsize[1], 4.5) * num_channels * 0.62),
                sharex=False,
            )
            axes = list(np.atleast_1d(axes))

        fig.suptitle(plot_name, fontsize=16, fontweight="bold")
        rng = np.random.default_rng(42)
        legend_handles = []
        legend_labels = []

        for ax_index, (ax, channel) in enumerate(zip(axes, channels)):
            if channel not in agg_data:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}' channel '{channel}': no aggregated data. Skipping subplot.")
                continue

            data = agg_data[channel]
            if len(data) == 0:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}' channel '{channel}': no valid values. Skipping subplot.")
                continue

            # Create box plot
            bp = ax.boxplot(
                [data],
                labels=[channel],
                patch_artist=True,
                widths=box_width,
                showfliers=show_fliers,
            )

            self._apply_boxplot_artist_style(
                bp,
                box_settings,
                facecolor=aggregated_box_color,
                alpha=aggregated_box_alpha,
            )

            # Axis labels
            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight='bold', fontsize=12
            )
            ax.set_title(channel, fontweight="bold", fontsize=12)

            # Apply axis limits if specified
            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)

            # Styling
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Zero line if within limits
            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            if show_points:
                overlay_series = self._collect_boxplot_point_series_from_data(channel, filtered_run_data)
                for run_label, color, values in overlay_series:
                    x_points = np.full(len(values), 1.0, dtype=float)
                    x_points += rng.uniform(-jitter, jitter, size=len(values))
                    ax.scatter(
                        x_points,
                        values,
                        s=point_size,
                        alpha=point_alpha,
                        color=color,
                        edgecolors="none",
                        zorder=3,
                    )
                    if run_label not in legend_labels:
                        legend_handles.append(Line2D([0], [0], marker='o', linestyle='None', color=color))
                        legend_labels.append(run_label)

        if show_points and legend_handles:
            self._add_standard_legend(axes[0], handles=legend_handles, labels=legend_labels, loc="upper right")
        if gate_text:
            self._display_gate_info(axes[0], gate_text, legend=axes[0].get_legend() if show_points and legend_handles else None)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.05, facecolor='white')
        plt.close(fig)
        print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # RUN ALL
    # ------------------------------------------------------------

    def plot_data(self):
        """Run all plot generators in sequence."""
        self._ensure_preprocessed()
        self.generate_waveform_plots()
        self.generate_scatter_plots()
        self.generate_psd_plots()
        self.generate_histogram_plots()
        self.generate_bar_plots()
        self.generate_box_plots()
        print(f"\nAll plots saved to: {self.plots_dir}")

    def plot_all(self):
        """Backward-compatible alias for plot_data()."""
        self.plot_data()

