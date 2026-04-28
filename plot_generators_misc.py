"""PSD and Histogram plot generator mixin for DataPlotter."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import datafunctions

try:
    from tqdm import tqdm as _tqdm_raw
    def _tqdm(it, **kw): return _tqdm_raw(it, file=__import__('sys').stderr, dynamic_ncols=True, force=True, **kw)
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable


class PsdHistMixin:
    """PSD and Histogram plot generation methods. Mixed into DataPlotter."""

    # ------------------------------------------------------------------
    # PSD
    # ------------------------------------------------------------------

    def generate_psd_plots(self):
        """Create PSD plots, skipping runs with unavailable or invalid channel data."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(2)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="PSD", unit="plot", leave=True)
        for plot_def in plot_iter:
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

            # channel may be a single string or a list/tuple of strings
            channels_list = [channel] if isinstance(channel, str) else list(channel)
            line_styles = ["-", "--", ":", "-."]

            if self.verbose:
                print(f"Creating PSD plot: {plot_name} ({', '.join(channels_list)})")

            filename = self._sanitize_plot_filename("psd", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.psd_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel("Frequency (Hz)", fontweight="bold")
            primary_ch = channels_list[0]
            ax.set_ylabel(
                datafunctions.format_psd_ylabel(primary_ch, self.units_map),
                fontweight="bold",
            )

            plotted_any = False
            multi = len(channels_list) > 1
            for run in self.runs:
                run_name = run["name"].lower()
                if run_name not in self.run_data:
                    if self.verbose:
                        print(
                            f"[WARNING][DataPlotter] PSD '{plot_name}': run '{run_name}' "
                            "has no loaded dataframe. Skipping."
                        )
                    continue
                df = self.run_data[run_name]

                for ch_idx, ch in enumerate(channels_list):
                    if ch not in df.columns:
                        if self.verbose:
                            print(
                                f"[WARNING][DataPlotter] PSD '{plot_name}': channel '{ch}' "
                                f"missing in run '{run_name}'. Skipping."
                            )
                        continue

                    signal = df[ch]
                    if isinstance(signal, tuple) or not hasattr(signal, "__iter__"):
                        if self.verbose:
                            print(
                                f"[WARNING][DataPlotter] PSD '{plot_name}': channel '{ch}' "
                                f"in run '{run_name}' has invalid type. Skipping."
                            )
                        continue

                    signal = np.asarray(signal, dtype=float)
                    freq, power = datafunctions.calculate_psd(signal, self.FILTER_SAMPLE_RATE, nperseg=nperseg)
                    if freq is None:
                        if self.verbose:
                            print(
                                f"[WARNING][DataPlotter] PSD '{plot_name}': not enough data "
                                f"for '{ch}' in run '{run_name}'. Skipping."
                            )
                        continue

                    lstyle = line_styles[ch_idx % len(line_styles)]
                    lbl = f"{run['name'].upper()} — {ch}" if multi else run["name"].upper()
                    plot_func = ax.semilogy if log_scale else ax.plot
                    plot_func(
                        freq, power,
                        linewidth=1.8, color=run["color"],
                        linestyle=lstyle, alpha=0.9, label=lbl,
                    )
                    plotted_any = True

            if not plotted_any:
                if self.verbose:
                    print(
                        f"[WARNING][DataPlotter] PSD '{plot_name}': no valid data for "
                        f"'{', '.join(channels_list)}'. Plot not saved."
                    )
                plt.close(fig)
                continue

            has_x_limits = has_y_limits = False
            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None or xmax is not None:
                    ax.set_xlim(left=xmin, right=xmax)
                    has_x_limits = True
                if ymin is not None or ymax is not None:
                    if log_scale and ymin is not None:
                        ymin = max(ymin, 1e-4)
                    ax.set_ylim(bottom=ymin, top=ymax)
                    has_y_limits = True

            default_y_pad = 0 if log_scale else 0.04
            self._add_axis_edge_padding(
                ax,
                x_pad_ratio=(0 if has_x_limits else 0.02),
                y_pad_ratio=(0 if has_y_limits else default_y_pad),
            )
            ax.grid(True, which="major", alpha=0.3)
            ax.grid(True, which="minor", alpha=0.22)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            self._add_standard_legend(ax, loc="best")

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.15, facecolor="white")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def generate_histogram_plots(self):
        """Create histogram plots showing time distribution of channel values."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(3)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="Histogram", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name, channel, axis_limits, log_scale = plot_def

            if self.verbose:
                print(f"Creating histogram plot: {plot_name} ({channel})")

            filename = self._sanitize_plot_filename("histogram", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.histogram_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontsize=13, fontweight="bold",
            )
            ax.set_ylabel("Time (s)", fontsize=13, fontweight="bold")

            # Collect all data for shared bins
            all_values = []
            for run in self.runs:
                run_name = run["name"].lower()
                df = self.run_data.get(run_name)
                if df is None or channel not in df.columns:
                    continue
                vals = df[channel].dropna()
                if not vals.empty:
                    all_values.append(vals.values)

            if not all_values:
                if self.verbose:
                    print(
                        f"[WARNING][DataPlotter] Histogram '{plot_name}': "
                        f"no valid data for '{channel}'. Plot not saved."
                    )
                plt.close(fig)
                continue

            combined = np.concatenate(all_values)
            bins = datafunctions.compute_nice_histogram_bins(combined, num_bins=30)

            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None or xmax is not None:
                    ax.set_xlim(left=xmin, right=xmax)
                if xmin is not None and xmax is not None:
                    bins = datafunctions.compute_equal_width_bins_in_limits(xmin, xmax, bins)
                if ymin is not None or ymax is not None:
                    if log_scale and ymin is not None:
                        ymin = max(ymin, 1e-6)
                    ax.set_ylim(bottom=ymin, top=ymax)

            hist_data, hist_weights, hist_colors, hist_labels = [], [], [], []
            dt = 1.0 / self.FILTER_SAMPLE_RATE
            for run in self.runs:
                run_name = run["name"].lower()
                df = self.run_data.get(run_name)
                if df is None or channel not in df.columns:
                    continue
                data = df[channel].dropna()
                if data.empty:
                    continue
                hist_data.append(data.to_numpy())
                hist_weights.append(np.full(len(data), dt))
                hist_colors.append(run["color"])
                hist_labels.append(run["name"].upper())

            if hist_data:
                ax.hist(
                    hist_data, bins=bins, weights=hist_weights,
                    alpha=0.72, color=hist_colors, label=hist_labels,
                    edgecolor="white", linewidth=0.8,
                    log=log_scale, stacked=False, histtype="bar", rwidth=0.9,
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
                    ax.grid(True, which="minor", axis="x", alpha=0.12, linewidth=0.3)
                ax.grid(True, which="major", axis="x", alpha=0.22, linewidth=0.45)

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
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            self._add_standard_legend(ax, loc="best")

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.05, facecolor="white")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")
