import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import font_manager
from pathlib import Path
import datafunctions
from collections import Counter


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
        fig_size=[(15.5, 6.4), (10, 8), (10, 8)],
        units_map=None,
        plot_aspect_ratios=None,
        sample_rate=100,
        scatter_dot_size=5,
        scatter_transparency=0.8,
    ):
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
                print(f"Data file for {run_name} not found: {file_path}, skipping this run")
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
            print(f"Error loading {file_path}: {e}")
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
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_channel_mappings(
                    self.run_data[name], self.CHANNEL_MAPPINGS, name
                )

    def apply_transformations(self):
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                self.run_data[name] = datafunctions.apply_transformations(
                    self.run_data[name], name, self.CHANNEL_TRANSFORMS
                )

    def apply_calculated_channels(self):
        for run in self.runs:
            name = run["name"].lower()
            if name in self.run_data:
                datafunctions.apply_calculated_channels(
                    self.run_data[name], name, self.CALCULATED_CHANNELS
                )

    def apply_lowpass_filters(self):
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
        if not self.PLOT_DEFINITIONS or len(self.PLOT_DEFINITIONS) <= index:
            return []
        return self.PLOT_DEFINITIONS[index] or []

    def _sanitize_plot_filename(self, prefix, plot_name, suffix=""):
        safe = (
            plot_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("\\", "_")
        )
        return f"{prefix}_{safe}{suffix}.png"

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

        return (w, h)

    def _mask_waveform_discontinuities(self, x_values, y_values):
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
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if xmax > xmin:
            pad = (xmax - xmin) * x_pad_ratio
            ax.set_xlim(xmin - pad, xmax + pad)

        if ymax > ymin:
            pad = (ymax - ymin) * y_pad_ratio
            ax.set_ylim(ymin - pad, ymax + pad)

    # ------------------------------------------------------------
    # WAVEFORM PLOTS
    # ------------------------------------------------------------

    def _prepare_waveform_channels(self, channels, axis_limits, reference_lines, subplot_heights):
        available_channels = []
        avail_lims = []
        avail_refs = []
        avail_heights = []

        for i, ch in enumerate(channels):
            count = sum(ch in self.run_data[r["name"].lower()].columns for r in self.runs)

            if count == 0:
                print(f"  Warning: Channel {ch} missing from all runs, skipping")
                continue

            if count < len(self.runs):
                print(f"  Warning: Channel {ch} only present in {count}/{len(self.runs)}, skipping")
                continue

            available_channels.append(ch)
            avail_lims.append(axis_limits[i] if axis_limits and i < len(axis_limits) else None)
            avail_refs.append(reference_lines[i] if reference_lines and i < len(reference_lines) else None)
            avail_heights.append(subplot_heights[i] if subplot_heights and i < len(subplot_heights) else 1.0)

        return available_channels, avail_lims, avail_refs, avail_heights

    def generate_waveform_plots(self):
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
                    if yl is not None and yh is not None:
                        ax.set_ylim(yl, yh)

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
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
                    ax.grid(True, which="major", axis="x", alpha=0.45, linewidth=0.5)
                    ax.grid(True, which="minor", axis="x", alpha=0.225, linewidth=0.3)

            plt.tight_layout(pad=0.3, h_pad=-0.8)

            # Legend (only show once)
            handles, labels = axes[0].get_legend_handles_labels()
            legend = axes[-1].legend(
                handles,
                labels,
                loc="lower left",
                bbox_to_anchor=(0.015, 0.02),
                framealpha=0.99,
                prop={"family": "Montserrat", "weight": "bold", "size": 10.4},
            )

            self._colorize_legend_labels(legend)
            fig.savefig(self.plots_dir / filename, dpi=300, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # SCATTER PLOTS
    # ------------------------------------------------------------

    def generate_scatter_plots(self):
        plots = self._get_plot_group(1, "scatter")

        for plot_def in plots:
            plot_name, (x_var, y_var), axis_limits, best_fit, fit_split = plot_def
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
                    print(f"  Missing {x_var} or {y_var} in run {rn}, skipping")
                    continue

                x = df[x_var].dropna()
                y = df[y_var].reindex(x.index).dropna()
                x = x.reindex(y.index)

                if best_fit == 0:
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
            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None and xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)

            self._add_axis_edge_padding(ax)

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

                # gradient comparison only meaningful for exactly 2 runs
                if len(self.runs) == 2:
                    txt = self._format_gradient_error_text(eq_list, x_var, fit_split, y_var)
                    if txt:
                        self._display_gradient_error(ax, txt, anchor)

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(
                    fontsize=10,
                    framealpha=1,
                    loc="best",
                    handlelength=1.8,
                    prop={"family": "Montserrat", "weight": "bold", "size": 12},
                )
            self._colorize_legend_labels(legend)

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, facecolor="white")
            plt.close(fig)
            print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # PSD PLOTS
    # ------------------------------------------------------------

    def generate_psd_plots(self):
        """Create PSD plots based on PSD_PLOT_DEFINITIONS"""
        plots = self._get_plot_group(2, 'psd')

        for plot_def in plots:
            # Parse plot definition
            if len(plot_def) == 3:
                plot_name, channel, axis_limits = plot_def
                log_scale = True
                nperseg = 256
            elif len(plot_def) == 4:
                plot_name, channel, axis_limits, log_scale = plot_def
                nperseg = 256
            elif len(plot_def) == 5:
                plot_name, channel, axis_limits, log_scale, nperseg = plot_def
            else:
                raise ValueError("Invalid PSD plot definition")

            print(f"Creating PSD plot: {plot_name} ({channel})")

            # Ensure the channel exists in all runs
            missing = [
                run['name'] for run in self.runs
                if channel not in self.run_data[run['name'].lower()].columns
            ]
            if missing:
                print(f"  Warning: PSD skipped for {channel} — missing in runs: {missing}")
                continue

            # Setup figure
            filename = self._sanitize_plot_filename("psd", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.psd_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
            ax.set_ylabel(
                f"{datafunctions.add_units_to_label(channel, self.units_map)} PSD",
                fontsize=13, fontweight='bold'
            )

            # ---- RUN LOOP ----
            for run in self.runs:
                run_name = run['name'].lower()
                df = self.run_data[run_name]

                signal = df[channel]

                # SAFETY CHECKS: ensure valid signal type
                # ----------------------------------------
                if isinstance(signal, tuple):
                    print(f"  Warning: PSD skipped for {channel} in run '{run_name}' (signal is tuple)")
                    continue

                if not isinstance(signal, (pd.Series, np.ndarray, list)):
                    print(f"  Warning: PSD skipped for {channel} in run '{run_name}' (invalid type {type(signal)})")
                    continue

                # Convert to Series for safe processing
                signal = np.asarray(signal, dtype=float)

                # Must be numeric
                if not np.issubdtype(signal.dtype, np.number):
                    print(f"  Warning: PSD skipped for {channel} in run '{run_name}' (non-numeric dtype)")
                    continue

                # Compute PSD
                freq, power = datafunctions.calculate_psd(
                    signal,
                    self.FILTER_SAMPLE_RATE,
                    nperseg=nperseg
                )

                if freq is None:
                    print(f"  Warning: Not enough data for PSD of {channel} in run '{run_name}'")
                    continue

                # Plot PSD
                plot_func = ax.semilogy if log_scale else ax.plot
                plot_func(freq, power,
                          linewidth=1.8,
                          color=run['color'],
                          alpha=0.9,
                          label=run['name'].upper())

            # Axis limits
            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None and xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if ymin is not None and ymax is not None:
                    if log_scale:
                        ymin = max(ymin, 1e-6)  # avoid log(0) issues
                    ax.set_ylim(ymin, ymax)

            # Padding & styling
            self._add_axis_edge_padding(ax, x_pad_ratio=0.02, y_pad_ratio=0.04)
            ax.grid(True, which='major', alpha=0.3)
            ax.grid(True, which='minor', alpha=0.15)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Legend
            legend = ax.legend(
                fontsize=10,
                framealpha=1,
                loc='best',
                borderpad=0.35,
                handlelength=1.8,
                prop={'family': 'Montserrat', 'weight': 'bold', 'size': 12}
            )
            self._colorize_legend_labels(legend)

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
        lines = []
        for line in str(equation).splitlines():
            cleaned = line.strip()
            prefix = f"{label} "
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
            lines.append(cleaned)
        return "\n".join(lines)


    def _colorize_legend_labels(self, legend):
        """
        Match legend text color to the plotted line/marker color.
        This keeps multi-run legends visually consistent.
        """
        if legend is None:
            return

        for text, handle in zip(legend.get_texts(), legend.legend_handles):
            color = None

            # Line2D objects (common)
            if hasattr(handle, "get_color"):
                color = handle.get_color()

            # Patch objects (scatter markers)
            elif hasattr(handle, "get_facecolor"):
                fc = handle.get_facecolor()
                if isinstance(fc, (list, tuple, np.ndarray)) and len(fc) > 0:
                    color = fc[0]

            if color is not None:
                text.set_color(color)


    def _display_equations(self, ax, eq_list):
        x_anchor, y_anchor, halign, valign = self._select_trendline_anchor(ax, eq_list)
        y_step = 0.06 / max(len(eq_list) - 1, 1)
        boxes = []

        for i, (label, equation, color, _, _, _) in enumerate(eq_list):
            ypos = y_anchor - i * y_step if valign == "top" else y_anchor + i * y_step
            text = self._format_trendline_text(label, equation)
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

    def _format_gradient_error_text(self, equations_list, x_var=None, fit_split=None, y_var=None):
        """
        Create gradient error text comparing slopes between the first two runs.
        Safely handles cases where one or both slope values are None.
        """

        # Only defined for exactly two runs
        if len(equations_list) != 2:
            return None

        # Extract labels + slopes
        label_a, _, _, _, _, slopes_a = equations_list[0]
        label_b, _, _, _, _, slopes_b = equations_list[1]

        def percent_error(a, b):
            """Return percentage error or None if undefined."""
            if a is None or b is None or b == 0:
                return None
            return ((a - b) / b) * 100

        # Formatter that handles None safely
        def fmt(v):
            return "undefined" if v is None else f"{v:+.1f}%"

        lines = [f"% Error in {label_a.upper()} w.r.t. {label_b.upper()}:"]

        # --------------------------------------------------------
        # DOUBLE-FIT CASE (tuple slopes)
        # --------------------------------------------------------
        if isinstance(slopes_a, tuple) and isinstance(slopes_b, tuple) and fit_split is not None:
            split_axis, split_value = fit_split
            axis_name = x_var if split_axis == "x" else y_var

            a1, a2 = slopes_a
            b1, b2 = slopes_b

            e1 = percent_error(a1, b1)
            e2 = percent_error(a2, b2)

            lines.append(f"{axis_name} < {split_value}: {fmt(e1)}")
            lines.append(f"{axis_name} ≥ {split_value}: {fmt(e2)}")

        # --------------------------------------------------------
        # SINGLE-FIT CASE (one slope each)
        # --------------------------------------------------------
        else:
            e = percent_error(slopes_a, slopes_b)
            lines.append(fmt(e))

        return "\n".join(lines)

    def _display_gradient_error(self, ax, text, anchor):
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
        self.generate_waveform_plots()
        self.generate_scatter_plots()
        self.generate_psd_plots()
        print(f"\nAll plots saved to: {self.plots_dir}")