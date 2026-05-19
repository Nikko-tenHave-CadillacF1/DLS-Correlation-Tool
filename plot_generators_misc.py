"""PSD and Histogram plot generator mixin for DataPlotter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import datafunctions
from datafunctions import _tqdm


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
            # Typed dataclass access (#9/#24).
            plot_name   = plot_def.name
            channel     = plot_def.channel
            axis_limits = plot_def.axis_limits
            log_scale   = plot_def.log_scale
            nperseg     = plot_def.nperseg if plot_def.nperseg is not None else 512
            annotate_at = plot_def.annotate_at
            markers     = plot_def.markers

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
            psd_curves = []  # (run_color, freq_array, power_array) for annotate_at
            for run in self.runs:
                run_name = run["name"].lower()
                if run_name not in self.run_data:
                    print(
                        f"[WARNING][DataPlotter] PSD '{plot_name}': run '{run_name}' "
                        "has no loaded dataframe. Skipping."
                    )
                    continue
                df = self.run_data[run_name]

                for ch_idx, ch in enumerate(channels_list):
                    if ch not in df.columns:
                        hint = self._format_missing_channel_hint(run_name, ch)
                        print(
                            f"[WARNING][DataPlotter] PSD '{plot_name}': channel '{ch}' "
                            f"missing in run '{run_name}'. Skipping."
                            + (f"\n{hint}" if hint else "")
                        )
                        continue

                    signal = df[ch]
                    if isinstance(signal, tuple) or not hasattr(signal, "__iter__"):
                        print(
                            f"[WARNING][DataPlotter] PSD '{plot_name}': channel '{ch}' "
                            f"in run '{run_name}' has invalid type. Skipping."
                        )
                        continue

                    signal = np.asarray(signal, dtype=float)
                    freq, power = self._cached_psd(run_name, ch, nperseg)
                    if freq is None:
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
                    psd_curves.append((run["color"], freq, power))
                    plotted_any = True

            if not plotted_any:
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
            self._apply_grid(ax, which="both")
            # On log-scale PSD plots, force denser minor ticks (one per decade
            # intermediate 2..9) so reviewers can read off intermediate decades.
            if log_scale:
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=24))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.grid(True, which="minor", axis="y", alpha=0.20, linewidth=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            self._add_standard_legend(ax, loc="best")

            # ── annotate_at: mark PSD values at specific frequencies ──────
            if annotate_at is not None and psd_curves:
                if isinstance(annotate_at, (list, tuple)):
                    freq_targets = [float(v) for v in annotate_at]
                else:
                    freq_targets = [float(annotate_at)]

                xl, xr = ax.get_xlim()
                for f_at in freq_targets:
                    if not (xl <= f_at <= xr):
                        continue
                    ax.axvline(f_at, color="#5E5E5E", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

                    # Collect annotation points at this frequency
                    ann_items = []
                    for (run_color, freq_arr, power_arr) in psd_curves:
                        idx = np.argmin(np.abs(freq_arr - f_at))
                        p_at = power_arr[idx]
                        ann_items.append((p_at, run_color))

                    if ann_items:
                        ann_items.sort(key=lambda t: t[0])
                        trans = ax.transData
                        display_ys = [trans.transform((f_at, item[0]))[1] for item in ann_items]
                        min_sep = 16
                        adjusted_display_ys = list(display_ys)
                        for i in range(1, len(adjusted_display_ys)):
                            gap = adjusted_display_ys[i] - adjusted_display_ys[i - 1]
                            if gap < min_sep:
                                adjusted_display_ys[i] = adjusted_display_ys[i - 1] + min_sep

                        for i, (p_at, color_e) in enumerate(ann_items):
                            nudge_pts = adjusted_display_ys[i] - display_ys[i]
                            y_offset = 8 + nudge_pts
                            ax.scatter([f_at], [p_at], color=color_e, s=50, zorder=10,
                                       edgecolors="white", linewidths=1.2)
                            ax.annotate(
                                f"{p_at:.3g}",
                                xy=(f_at, p_at), xytext=(10, y_offset),
                                textcoords="offset points",
                                fontsize=9, fontweight="bold", color=color_e,
                                zorder=11,
                                arrowprops=dict(arrowstyle="-", color=color_e,
                                                lw=0.8, alpha=0.6),
                                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                          alpha=0.92, edgecolor=color_e, linewidth=0.8),
                            )

            # ── Markers (#6) ──
            # Only static (x-valued) markers apply to PSD plots.
            if markers:
                xl_m, xr_m = ax.get_xlim()
                for m in markers:
                    if m.condition is not None:
                        continue  # condition markers are waveform-only
                    if not (xl_m <= m.x <= xr_m):
                        continue
                    mcolor = m.color or "#5E5E5E"
                    ax.axvline(m.x, color=mcolor, linestyle=m.linestyle,
                               linewidth=1.2, alpha=0.7, zorder=2)
                    if m.label:
                        ax.text(
                            m.x, 1.01, m.label,
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color=mcolor,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      edgecolor=mcolor, linewidth=0.8, alpha=0.9),
                            zorder=12,
                        )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
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
            # Typed dataclass access (#9/#24).
            plot_name   = plot_def.name
            channel     = plot_def.channel
            axis_limits = plot_def.axis_limits
            log_scale   = plot_def.log_scale
            markers     = plot_def.markers

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
                    ax.grid(True, which="minor", axis="x", **self.GRID_STYLE["minor"])
                ax.grid(True, which="major", axis="x", **self.GRID_STYLE["major"])

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
            self._apply_grid(ax, which="major", axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            self._add_standard_legend(ax, loc="best")

            # ── Markers (#6) ──
            # Only static (x-valued) markers apply to histograms.
            if markers:
                xl_m, xr_m = ax.get_xlim()
                for m in markers:
                    if m.condition is not None:
                        continue  # condition markers are waveform-only
                    if not (xl_m <= m.x <= xr_m):
                        continue
                    mcolor = m.color or "#5E5E5E"
                    ax.axvline(m.x, color=mcolor, linestyle=m.linestyle,
                               linewidth=1.2, alpha=0.7, zorder=2)
                    if m.label:
                        ax.text(
                            m.x, 1.01, m.label,
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color=mcolor,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      edgecolor=mcolor, linewidth=0.8, alpha=0.9),
                            zorder=12,
                        )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.05, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")


# ============================================================================
# Heatmap (#5)
# ============================================================================


class HeatmapMixin:
    """2-D binned aggregate heatmap generation. Mixed into DataPlotter."""

    def generate_heatmap_plots(self):
        """Create one heatmap PNG per (heatmap definition × run).

        For each run we render a panel showing the aggregate of z_channel
        over (x_channel, y_channel) bins. When z_channel is None, we plot
        a 2-D histogram (counts).
        """
        self._ensure_preprocessed()
        plots = self._get_plot_group(6)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="Heatmap", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name   = plot_def.name
            x_channel   = plot_def.x_channel
            y_channel   = plot_def.y_channel
            z_channel   = plot_def.z_channel
            agg         = plot_def.aggregation
            bins        = plot_def.bins
            axis_limits = plot_def.axis_limits
            cmap        = plot_def.cmap
            z_limits    = plot_def.z_limits
            gate_spec   = plot_def.gate
            markers     = plot_def.markers
            min_count   = plot_def.min_count

            if self.verbose:
                print(f"Creating heatmap plot: {plot_name}")

            # Loaded, gated run frames
            run_frames = []
            for run in self.runs:
                rn = run["name"].lower()
                df = self._get_filtered_run_dataframe(rn, gate_spec)
                if df is None or x_channel not in df.columns or y_channel not in df.columns:
                    continue
                if z_channel is not None and z_channel not in df.columns:
                    continue
                run_frames.append((run, df))

            if not run_frames:
                print(f"[WARNING][DataPlotter] Heatmap '{plot_name}': no usable runs. Skipping.")
                continue

            ncols = len(run_frames)
            base_w = self.histogram_FIGSIZE[0]
            fig, axes = plt.subplots(
                1, ncols,
                figsize=(base_w * 0.9 * ncols, self.histogram_FIGSIZE[1]),
                squeeze=False,
            )
            axes = axes[0]

            # Determine shared bin edges from data range across all runs
            xs_all = np.concatenate([
                pd.to_numeric(df[x_channel], errors="coerce").dropna().to_numpy()
                for _, df in run_frames
            ])
            ys_all = np.concatenate([
                pd.to_numeric(df[y_channel], errors="coerce").dropna().to_numpy()
                for _, df in run_frames
            ])
            if axis_limits and axis_limits[0]:
                x_lo = axis_limits[0][0] if axis_limits[0][0] is not None else float(np.nanmin(xs_all))
                x_hi = axis_limits[0][1] if axis_limits[0][1] is not None else float(np.nanmax(xs_all))
            else:
                x_lo, x_hi = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
            if axis_limits and axis_limits[1]:
                y_lo = axis_limits[1][0] if axis_limits[1][0] is not None else float(np.nanmin(ys_all))
                y_hi = axis_limits[1][1] if axis_limits[1][1] is not None else float(np.nanmax(ys_all))
            else:
                y_lo, y_hi = float(np.nanmin(ys_all)), float(np.nanmax(ys_all))

            if isinstance(bins, tuple):
                nx, ny = bins
            else:
                nx = ny = int(bins)
            x_edges = np.linspace(x_lo, x_hi, nx + 1)
            y_edges = np.linspace(y_lo, y_hi, ny + 1)

            # Pre-compute z range across all runs for shared colour scale
            all_grids = []
            for run, df in run_frames:
                xv = pd.to_numeric(df[x_channel], errors="coerce").to_numpy()
                yv = pd.to_numeric(df[y_channel], errors="coerce").to_numpy()
                mask = np.isfinite(xv) & np.isfinite(yv)
                if z_channel is None:
                    grid, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[x_edges, y_edges])
                    counts = grid
                else:
                    zv = pd.to_numeric(df[z_channel], errors="coerce").to_numpy()
                    mask = mask & np.isfinite(zv)
                    counts, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[x_edges, y_edges])
                    sums, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[x_edges, y_edges], weights=zv[mask])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        if agg == "mean":
                            grid = sums / counts
                        elif agg == "sum":
                            grid = sums
                        elif agg == "count":
                            grid = counts
                        else:
                            # median/std/max/min via per-cell iteration on the
                            # (relatively small) bin count.
                            xi = np.clip(np.digitize(xv[mask], x_edges) - 1, 0, nx - 1)
                            yi = np.clip(np.digitize(yv[mask], y_edges) - 1, 0, ny - 1)
                            zfin = zv[mask]
                            grid = np.full((nx, ny), np.nan)
                            from collections import defaultdict
                            buckets = defaultdict(list)
                            for i_, j_, z_ in zip(xi, yi, zfin):
                                buckets[(i_, j_)].append(z_)
                            reducer = {
                                "median": np.median, "std": np.std,
                                "max": np.max, "min": np.min,
                            }[agg]
                            for (i_, j_), values in buckets.items():
                                if len(values) >= min_count:
                                    grid[i_, j_] = reducer(values)
                    grid = np.where(counts >= min_count, grid, np.nan)
                all_grids.append((run, grid, counts))

            zs_combined = np.concatenate([g.ravel() for _, g, _ in all_grids])
            zs_combined = zs_combined[np.isfinite(zs_combined)]
            if z_limits and (z_limits[0] is not None or z_limits[1] is not None):
                z_min = z_limits[0] if z_limits[0] is not None else float(np.nanmin(zs_combined)) if len(zs_combined) else 0.0
                z_max = z_limits[1] if z_limits[1] is not None else float(np.nanmax(zs_combined)) if len(zs_combined) else 1.0
            elif len(zs_combined):
                z_min, z_max = float(np.nanmin(zs_combined)), float(np.nanmax(zs_combined))
            else:
                z_min, z_max = 0.0, 1.0

            for ax, (run, grid, counts) in zip(axes, all_grids):
                # imshow uses (rows=y, cols=x) so we transpose to get x on the x-axis
                im = ax.imshow(
                    grid.T, origin="lower", aspect="auto",
                    extent=(x_lo, x_hi, y_lo, y_hi),
                    cmap=cmap, vmin=z_min, vmax=z_max,
                    interpolation="nearest",
                )
                ax.set_title(run["name"].upper(), fontsize=11, fontweight="bold", color=run["color"])
                ax.set_xlabel(datafunctions.add_units_to_label(x_channel, self.units_map),
                              fontweight="bold")
                ax.set_ylabel(datafunctions.add_units_to_label(y_channel, self.units_map),
                              fontweight="bold")
                self._apply_grid(ax, which="major")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Markers (#6) — static only (heatmap x-axis is a value axis)
                if markers:
                    for m in markers:
                        if m.condition is not None:
                            continue
                        mcolor = m.color or "#5E5E5E"
                        ax.axvline(m.x, color=mcolor, linestyle=m.linestyle,
                                   linewidth=1.2, alpha=0.7, zorder=5)
                        if m.label:
                            ax.text(m.x, y_hi, f" {m.label}",
                                    ha="left", va="top",
                                    fontsize=8, fontweight="bold", color=mcolor,
                                    zorder=12)

            # Shared colourbar on the right
            cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
            if z_channel is None:
                cbar.set_label("Count", fontweight="bold")
            else:
                cbar.set_label(
                    f"{agg}({datafunctions.add_units_to_label(z_channel, self.units_map)})",
                    fontweight="bold",
                )

            filename = self._sanitize_plot_filename("heatmap", plot_name)
            fig.suptitle(plot_name, fontweight="bold", fontsize=13)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi,
                        pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")
