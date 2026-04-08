"""Data loading, preprocessing, and plotting pipeline for correlation reports."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import font_manager
from pathlib import Path
import datafunctions
from collections import Counter
from matplotlib.patches import Patch

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
        fig_size=[(15.5, 6.4), (10, 8), (10, 8), (10, 8)],
        units_map=None,
        plot_aspect_ratios=None,
        sample_rate=100,
        scatter_dot_size=5,
        scatter_transparency=0.8,
    ):
        """Build a plotter instance and run the preprocessing pipeline."""
        self.runs = runs
        self._configure_plot_style()

        # Store config
        self.PLOT_DEFINITIONS = plot_definitions
        self.CHANNEL_MAPPINGS = channel_mappings
        self.CALCULATED_CHANNELS = calculated_channels

        self.run_filepaths = {}
        self.run_data = {}
        self.run_units = {}
        self.run_required_cols = {}

        for run in self.runs:
            run_name = run["name"].lower()
            file_path = Path(root_folder) / run["file"]
            self.run_filepaths[run_name] = file_path

            if not file_path.exists():
                print(f"[WARNING][DataPlotter] Missing data file for run '{run_name}': {file_path}. Skipping run.")
                continue

            use_python_engine = (run_name == "car")
            self.run_required_cols[run_name] = self._get_required_source_columns(run_name)

            data, _, units = self._load_run_data(
                file_path,
                use_python_engine=use_python_engine,
                columns_to_load=self.run_required_cols[run_name],
            )

            self.run_data[run_name] = data
            self.run_units[run_name] = units

        self.CHANNEL_TRANSFORMS = channel_transforms
        self.units_map = units_map
        self.FILTER_SAMPLE_RATE = sample_rate
        self.LOW_PASS_FILTERS = low_pass_filters

        self.SCATTER_DOT_SIZE = scatter_dot_size
        self.SCATTER_TRANSPARENCY = scatter_transparency

        self.waveform_figsize = fig_size[0]
        self.scatter_FIGSIZE = fig_size[1]
        self.psd_FIGSIZE = fig_size[2]
        self.histogram_FIGSIZE = fig_size[3]
        self.plot_aspect_ratios = plot_aspect_ratios or {}

        # Create plots directory
        self.plots_dir = Path(root_folder) / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Pipeline
        self.apply_channel_mappings()
        self.apply_transformations()
        self.clean_data()
        self.apply_calculated_channels()
        self.apply_lowpass_filters()

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

        # Always include sLap for waveform alignment
        if (
            self.PLOT_DEFINITIONS
            and len(self.PLOT_DEFINITIONS) > 0
            and self.PLOT_DEFINITIONS[0]
        ):
            required_channels.add("sLap")

        # Scan all plot definitions
        if self.PLOT_DEFINITIONS:
            for plot_group in self.PLOT_DEFINITIONS:
                if plot_group is None:
                    continue
                for plot_def in plot_group:
                    if len(plot_def) >= 2:
                        if isinstance(plot_def[1], tuple):
                            required_channels.update(plot_def[1])
                        elif isinstance(plot_def[1], str):
                            required_channels.add(plot_def[1])

        # Resolve calculated dependencies
        resolved_channels = set()
        to_process = list(required_channels)
        processed = set()

        while to_process:
            channel = to_process.pop(0)
            if channel in processed:
                continue
            processed.add(channel)

            calc_set = self.CALCULATED_CHANNELS
            if isinstance(calc_set, dict):
                calc_set = calc_set.get(source_type) or calc_set

            if isinstance(calc_set, dict) and channel in calc_set:
                import inspect, re

                try:
                    source = inspect.getsource(calc_set[channel])
                    matches = re.findall(
                        r"df\['([^']+)'\]|df\[\"([^\"]+)\"\]", source
                    )
                    for m in matches:
                        dep = m[0] or m[1]
                        if dep not in processed:
                            to_process.append(dep)
                except Exception:
                    pass
            else:
                resolved_channels.add(channel)

        # Apply channel mappings: convert mapped names to original raw names
        source_columns = set()
        mappings = self.CHANNEL_MAPPINGS.get(source_type) if self.CHANNEL_MAPPINGS else {}
        for ch in resolved_channels:
            found_src = None
            for raw, mapped in mappings.items():
                if mapped == ch:
                    found_src = raw
                    break
            source_columns.add(found_src or ch)

        return source_columns

    # ------------------------------------------------------------
    # LOAD RUN DATA
    # ------------------------------------------------------------

    def _load_run_data(self, file_path, use_python_engine=False, columns_to_load=None):
        """Load CSV/TXT or Parquet, applying column filtering."""
        try:
            if file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
                df.columns = make_unique([str(c) for c in df.columns])
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
        for run_name, df in self.run_data.items():
            df = datafunctions.convert_yes_no_to_binary(df)

        for run_name in list(self.run_data.keys()):
            df = self.run_data[run_name]
            for col in list(df.columns):
                if df[col].dtype == "object":
                    non_nan = df[col].dropna()
                    if any(isinstance(x, str) for x in non_nan):
                        df.drop(col, axis=1, inplace=True)
                        print(f"Dropped {col} from run {run_name} (string column)")
                        continue

                df[col] = datafunctions.sanitize_numeric_series(df[col])
                df[col] = df[col].interpolate(method="linear")

    # ------------------------------------------------------------
    # MAPPINGS / TRANSFORMS / CALCULATED / FILTERS
    # ------------------------------------------------------------

    def apply_channel_mappings(self):
        """Apply source-specific channel renaming for every loaded run."""
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_channel_mappings(
                    self.run_data[name], self.CHANNEL_MAPPINGS, name
                )

    def apply_transformations(self):
        """Apply configured per-source numeric transforms to each run."""
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_transformations(
                    self.run_data[name], name, self.CHANNEL_TRANSFORMS
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

    # ------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------

    def _get_plot_group(self, index, plot_type):
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

    def _mask_waveform_discontinuities(self, x_values, y_values):
        """Mask invalid lap-distance regions so line plots break at discontinuities."""
        xs = pd.Series(x_values).reset_index(drop=True)
        ys = pd.Series(y_values).reset_index(drop=True).copy()

        neg_mask = xs < 0
        xs.loc[neg_mask] = np.nan
        ys.loc[neg_mask] = np.nan

        if xs.notna().sum() > 1:
            reset_mask = xs.diff() < 0
            ys.loc[reset_mask] = np.nan

        return xs, ys

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

    def _format_psd_ylabel(self, channel):
        """Format PSD y-axis text with units when available."""
        units = ""
        if self.units_map:
            for key, value in self.units_map.items():
                if key.lower() == channel.lower():
                    units = value
                    break

        if units:
            return f"{channel} PSD ({units}^2/Hz)"
        return f"{channel} PSD"

    def _compute_nice_histogram_bins(self, data, num_bins=30):
        """Compute round-number histogram bins with integer-preferred widths."""
        values = np.asarray(data, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.array([0.0, 1.0])

        data_min = float(np.min(values))
        data_max = float(np.max(values))

        if np.isclose(data_min, data_max):
            start = np.floor(data_min)
            return np.array([start, start + 1.0])

        raw_step = (data_max - data_min) / max(num_bins, 1)
        exponent = np.floor(np.log10(raw_step))
        fraction = raw_step / (10 ** exponent)

        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10

        step = nice_fraction * (10 ** exponent)
        if step >= 1:
            step = max(1.0, float(np.round(step)))

        start = np.floor(data_min / step) * step
        end = np.ceil(data_max / step) * step
        bins = np.arange(start, end + step * 0.5, step)

        if bins.size < 2:
            bins = np.array([start, start + step])

        return bins

    def _compute_equal_width_bins_in_limits(self, xmin, xmax, reference_bins):
        """Compute equal-width bins in [xmin, xmax] with a count derived from a near-nice step."""
        xmin = float(xmin)
        xmax = float(xmax)
        if xmax <= xmin:
            return np.array([xmin, xmin + 1.0])

        if reference_bins is not None and len(reference_bins) > 1:
            target_step = float(reference_bins[1] - reference_bins[0])
        else:
            target_step = (xmax - xmin) / 30.0

        if target_step <= 0:
            target_step = (xmax - xmin) / 30.0

        bin_count = max(1, int(np.round((xmax - xmin) / target_step)))
        return np.linspace(xmin, xmax, bin_count + 1)

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

    # ------------------------------------------------------------
    # WAVEFORM PLOTS
    # ------------------------------------------------------------

    def _prepare_waveform_channels(self, channels, axis_limits, reference_lines, subplot_heights):
        """Filter waveform channels to those available in at least one run."""
        available_channels = []
        avail_lims = []
        avail_refs = []
        avail_heights = []

        for i, ch in enumerate(channels):
            count = sum(ch in self.run_data[r["name"].lower()].columns for r in self.runs)

            if count == 0:
                print(f"[WARNING][DataPlotter] Waveform channel '{ch}' missing from all runs. Skipping channel.")
                continue

            if count < len(self.runs):
                print(
                    f"[WARNING][DataPlotter] Waveform channel '{ch}' present in {count}/{len(self.runs)} runs. Plotting available runs only."
                )

            available_channels.append(ch)
            avail_lims.append(axis_limits[i] if axis_limits and i < len(axis_limits) else None)
            avail_refs.append(reference_lines[i] if reference_lines and i < len(reference_lines) else None)
            avail_heights.append(subplot_heights[i] if subplot_heights and i < len(subplot_heights) else 1.0)

        return available_channels, avail_lims, avail_refs, avail_heights

    def generate_waveform_plots(self):
        """Generate all configured waveform subplot figures."""
        plots = self._get_plot_group(0, "waveform")

        for plot_def in plots:
            if len(plot_def) == 4:
                plot_name, channels, axis_limits, ref_lines = plot_def
                subplot_heights = None
            elif len(plot_def) == 5:
                plot_name, channels, axis_limits, ref_lines, subplot_heights = plot_def
            else:
                raise ValueError("Waveform plot definition malformed")

            print(f"Creating waveform plot: {plot_name}")

            (avail_channels, avail_lims, avail_refs, avail_heights) = \
                self._prepare_waveform_channels(channels, axis_limits, ref_lines, subplot_heights)

            if not avail_channels:
                print(f"  No valid channels for {plot_name}")
                continue

            filename = self._sanitize_plot_filename("waveform", plot_name)
            min_height = 1.6 * sum(avail_heights)
            figsize = self._resolve_plot_figsize(filename, self.waveform_figsize, min_height=min_height)

            fig, axes = plt.subplots(
                len(avail_channels),
                1,
                figsize=figsize,
                sharex=True,
                squeeze=False,
                gridspec_kw={"height_ratios": avail_heights},
            )
            axes = axes.flatten()

            xlabel = "sLap (m)" if all(
                "sLap" in self.run_data[r["name"].lower()].columns for r in self.runs
            ) else "Sample"

            # Draw channels
            for idx, ch in enumerate(avail_channels):
                ax = axes[idx]

                for run in self.runs:
                    rn = run["name"].lower()
                    if rn not in self.run_data:
                        continue

                    df = self.run_data[rn]
                    if ch not in df.columns:
                        continue

                    x_vals = df["sLap"] if "sLap" in df.columns else df.index
                    y_vals = df[ch]

                    x_plot, y_plot = self._mask_waveform_discontinuities(x_vals, y_vals)
                    ax.plot(
                        x_plot,
                        y_plot,
                        linewidth=1.6,
                        color=run["color"],
                        label=run["name"].upper(),
                        alpha=0.85,
                    )

                ax.set_ylabel(
                    datafunctions.add_units_to_label(ch, units_map=self.units_map),
                    fontsize=8.2,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                )
                ax.yaxis.set_label_coords(-0.035, 0.5)
                ax.grid(True, axis="y", alpha=0.28, linewidth=0.45)

                if avail_lims[idx] is not None:
                    yl, yh = avail_lims[idx]
                    if yl is not None or yh is not None:
                        ax.set_ylim(bottom=yl, top=yh)

                if avail_refs[idx] is not None:
                    vals = avail_refs[idx]
                    if np.isscalar(vals):
                        vals = [vals]
                    for vv in vals:
                        ax.axhline(vv, linestyle="--", color="gray", alpha=0.4)

                if idx < len(avail_channels) - 1:
                    ax.tick_params(labelbottom=False)

            # Style x-axis
            bottom = axes[-1]
            bottom.set_xlabel(xlabel, fontweight="bold")
            bottom.tick_params(axis="x", labelsize=10)

            if xlabel == "sLap (m)":
                xmaxs = []
                for ax in axes:
                    _, xm = ax.get_xlim()
                    if xm > 0:
                        xmaxs.append(xm)
                if xmaxs:
                    xv = max(xmaxs)
                    xv = np.ceil(xv/100) * 100
                    for ax in axes:
                        ax.set_xlim(0, xv)

                for ax in axes:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=5, steps=[1, 2, 2.5, 5, 10]))
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                    ax.grid(True, which="major", axis="x", alpha=0.45, linewidth=0.5)
                    ax.grid(True, which="minor", axis="x", alpha=0.225, linewidth=0.3)

            # Legend (only show once, outside data area)
            handles, labels = axes[0].get_legend_handles_labels()
            self._add_waveform_figure_legend(fig, handles, labels)
            plt.tight_layout(pad=0.3, h_pad=-0.4, rect=(0, 0, 1, 0.95))
            fig.savefig(self.plots_dir / filename, dpi=300, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # SCATTER PLOTS
    # ------------------------------------------------------------

    def generate_scatter_plots(self):
        """Generate all configured scatter plots and optional fit overlays."""
        plots = self._get_plot_group(1, "scatter")

        for plot_def in plots:
            if len(plot_def) == 4:
                plot_name, (x_var, y_var), axis_limits, best_fit = plot_def
                fit_split = None
            elif len(plot_def) == 5:
                plot_name, (x_var, y_var), axis_limits, best_fit, fit_split = plot_def
            else:
                raise ValueError(
                    f"Scatter plot definition for '{plot_def[0] if plot_def else 'unknown'}' must have 4 or 5 items"
                )
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

                df = self.run_data[rn]

                if x_var not in df.columns or y_var not in df.columns:
                    print(
                        f"[WARNING][DataPlotter] Scatter plot '{plot_name}': missing '{x_var}' or '{y_var}' in run '{rn}'. Skipping run."
                    )
                    continue

                x = df[x_var].dropna()
                y = df[y_var].reindex(x.index).dropna()
                x = x.reindex(y.index)

                if isinstance(best_fit, (list, tuple)) and best_fit and isinstance(best_fit[0], (list, tuple)):
                    ok, slopes, intercepts, eq_text, color = datafunctions.plot_scatter_with_multi_fit(
                        ax, x.values, y.values,
                        run["name"].upper(), run["color"],
                        self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                        x_var, y_var,
                        fit_defs=best_fit
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x.values, y.values, slopes))

                elif best_fit == 0:
                    datafunctions.plot_scatter(
                        ax, x.values, y.values,
                        run["name"].upper(), run["color"],
                        self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                        x_var, y_var
                    )

                elif best_fit == 1:
                    ok, slope, intercept, eq_text, color = datafunctions.plot_scatter_with_1fit(
                        ax, x.values, y.values,
                        run["name"].upper(), run["color"],
                        self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                        x_var, y_var
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x.values, y.values, slope))

                elif best_fit == 2:
                    ok, slopes, intercepts, eq_text, color = datafunctions.plot_scatter_with_double_fit(
                        ax, x.values, y.values,
                        run["name"].upper(), run["color"],
                        self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                        x_var, y_var,
                        fit_split=fit_split
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x.values, y.values, slopes))

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

            # Trendline boxes
            if eq_list:
                anchor = self._display_equations(ax, eq_list)

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
                    eq_list, x_var, fit_split, y_var, fit_labels=fit_labels
                )
                if txt:
                    self._display_gradient_error(ax, txt, anchor)

            # Legend
            self._add_standard_legend(ax, loc="best")

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, facecolor="white")
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # PSD PLOTS
    # ------------------------------------------------------------

    def generate_psd_plots(self):
        """Create PSD plots from definitions, skipping only runs with unavailable/invalid channel data."""
        plots = self._get_plot_group(2, 'psd')

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
                self._format_psd_ylabel(channel),
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
        plots = self._get_plot_group(3, 'histogram')

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
            bins = self._compute_nice_histogram_bins(all_data, num_bins=num_bins)

            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None or xmax is not None:
                    ax.set_xlim(left=xmin, right=xmax)
                if xmin is not None and xmax is not None:
                    bins = self._compute_equal_width_bins_in_limits(xmin, xmax, bins)
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
                    alpha=0.7,
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
        """Normalize equation text before placing it on the plot."""
        lines = []
        for line in str(equation).splitlines():
            cleaned = line.strip()
            prefix = f"{label} "
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
            lines.append(cleaned)
        return "\n".join(lines)

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
        self, equations_list, x_var=None, fit_split=None, y_var=None, fit_labels=None
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
            if value is None or baseline is None or baseline == 0:
                return None
            return ((value - baseline) / baseline) * 100

        def fmt(value):
            return "undefined" if value is None else f"{value:+.1f}%"

        lines = [f"% Error vs {baseline_label.upper()}:"]
        label_width = max(len(entry[0].upper()) for entry in ordered_entries)

        if isinstance(baseline_slopes, tuple):
            segment_count = len(baseline_slopes)
            for idx in range(segment_count):
                if fit_labels and idx < len(fit_labels):
                    segment_name = fit_labels[idx]
                elif fit_split is not None and segment_count == 2:
                    split_axis, split_value = fit_split
                    axis_name = x_var if split_axis == "x" else y_var
                    segment_name = (
                        f"{axis_name} < {split_value}" if idx == 0 else f"{axis_name} >= {split_value}"
                    )
                else:
                    segment_name = f"Segment {idx + 1}"

                lines.append(f"{segment_name}:")
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

    # ------------------------------------------------------------
    # RUN ALL
    # ------------------------------------------------------------

    def plot_all(self):
        """Run all plot generators in sequence."""
        self.generate_waveform_plots()
        self.generate_scatter_plots()
        self.generate_psd_plots()
        self.generate_histogram_plots()
        print(f"\nAll plots saved to: {self.plots_dir}")

