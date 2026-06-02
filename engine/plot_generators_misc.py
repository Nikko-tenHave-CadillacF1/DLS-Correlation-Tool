"""PSD, Histogram, and Heatmap plot generator mixins for DataPlotter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from . import datafunctions
from .datafunctions import _tqdm
from .logger import log


class PsdHistMixin:
    """PSD and Histogram plot generation methods. Mixed into DataPlotter."""

    # ------------------------------------------------------------------
    # PSD
    # ------------------------------------------------------------------

    def generate_psd_plots(self):
        """Create PSD plots, skipping runs with unavailable or invalid channel data.

        Supports per-plot ``gate`` (segment-aware Welch — gated regions are
        sliced into contiguous runs and each long-enough run contributes a
        Welch periodogram, then averaged with weighting by sub-segment count)
        and per-run ``group`` keys (runs sharing a group are averaged in PSD
        space and drawn as a single overlay line).
        """
        self._ensure_preprocessed()
        plots = self._get_plot_group(2)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="PSD", unit="plot", leave=True)
        for plot_def in plot_iter:
            # Typed dataclass access (#9/#24).
            plot_name     = plot_def.name
            channel       = plot_def.channel
            axis_limits   = plot_def.axis_limits
            log_scale     = plot_def.log_scale
            nperseg       = plot_def.nperseg if plot_def.nperseg is not None else 512
            annotate_at   = plot_def.annotate_at
            markers       = plot_def.markers
            gate_spec     = getattr(plot_def, "gate", None)
            show_envelope = bool(getattr(plot_def, "show_envelope", False))
            reference_lines = getattr(plot_def, "reference_lines", None)

            # channel may be a single string or a list/tuple of strings
            channels_list = [channel] if isinstance(channel, str) else list(channel)
            line_styles = ["-", "--", ":", "-."]

            if self.verbose:
                log.debug("Creating PSD plot: %s (%s)", plot_name, ', '.join(channels_list))

            filename = self._sanitize_plot_filename("psd", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.psd_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel("Frequency (Hz)", fontweight="bold")
            primary_ch = channels_list[0]
            ax.set_ylabel(
                datafunctions.format_psd_ylabel(primary_ch, self.units_map),
                fontweight="bold",
            )

            # ── Group runs by their ``group`` key (default = run name) ────
            # Order preserved by first appearance.
            run_groups = {}  # group_name -> list[run dict]
            for run in self.runs:
                gname = run.get("group") or run["name"]
                run_groups.setdefault(gname, []).append(run)

            plotted_any = False
            multi = len(channels_list) > 1
            psd_curves = []  # (line_color, freq_array, power_array) — for annotate_at
            nyquist_lines = {}  # {fs/2: color}  — populated as groups are plotted

            for group_name, group_runs in run_groups.items():
                # Skip groups whose runs have no loaded data at all
                loaded_runs = [r for r in group_runs if r["name"].lower() in self.run_data]
                if not loaded_runs:
                    log.warning(
                        "PSD '%s': group '%s' has no loaded runs. Skipping.",
                        plot_name, group_name,
                    )
                    continue

                primary_run = loaded_runs[0]
                group_color = primary_run["color"]

                # Track this group's Nyquist (use first run's sample rate)
                rate_info = self.run_sample_rates.get(
                    primary_run["name"].lower(),
                    (self.FILTER_SAMPLE_RATE, "default"),
                )
                fs = float(rate_info[0]) if rate_info and rate_info[0] else 0.0
                if fs > 0:
                    nyq = round(fs / 2.0, 3)
                    nyquist_lines.setdefault(nyq, group_color)

                for ch_idx, ch in enumerate(channels_list):
                    # Collect per-run PSDs with segment-count weights
                    per_run_psds = []  # list of (freq, power, n_segs)
                    for run in loaded_runs:
                        run_name = run["name"].lower()
                        df = self.run_data.get(run_name)
                        if df is None or ch not in df.columns:
                            hint = self._format_missing_channel_hint(run_name, ch)
                            log.warning(
                                "PSD '%s': channel '%s' missing in run '%s'. Skipping.%s",
                                plot_name, ch, run_name, f"\n{hint}" if hint else "",
                            )
                            continue
                        freq, power, n_segs = self._cached_psd_with_segments(
                            run_name, ch, nperseg, gate_spec=gate_spec,
                        )
                        if freq is None or power is None or n_segs <= 0:
                            continue
                        per_run_psds.append((freq, power, n_segs))

                    if not per_run_psds:
                        continue

                    # Aggregate to a single curve per (group, channel).
                    # Use the first run's frequency grid as reference; all
                    # share the same nperseg and (after resampling) the same
                    # fs, so freq grids match.
                    freq = per_run_psds[0][0]
                    if len(per_run_psds) == 1:
                        power_mean = per_run_psds[0][1]
                        power_std = None
                    else:
                        # Segment-count-weighted PSD-domain mean (Method B).
                        # Interp to common grid if grids ever differ (defensive).
                        powers = []
                        weights = []
                        for f_i, p_i, n_i in per_run_psds:
                            if not np.array_equal(f_i, freq):
                                p_i = np.interp(freq, f_i, p_i)
                            powers.append(np.asarray(p_i, dtype=float))
                            weights.append(float(n_i))
                        stacked = np.vstack(powers)
                        w = np.asarray(weights, dtype=float)
                        power_mean = (stacked * w[:, None]).sum(axis=0) / w.sum()
                        if show_envelope:
                            # Weighted std for envelope shading
                            var = (((stacked - power_mean) ** 2) * w[:, None]).sum(axis=0) / w.sum()
                            power_std = np.sqrt(var)
                        else:
                            power_std = None

                    lstyle = line_styles[ch_idx % len(line_styles)]
                    lbl = f"{group_name.upper()} — {ch}" if multi else group_name.upper()
                    plot_func = ax.semilogy if log_scale else ax.plot
                    plot_func(
                        freq, power_mean,
                        linewidth=1.8, color=group_color,
                        linestyle=lstyle, alpha=0.9, label=lbl,
                    )
                    if power_std is not None:
                        lower = np.clip(power_mean - power_std, 1e-30 if log_scale else None, None)
                        upper = power_mean + power_std
                        ax.fill_between(
                            freq, lower, upper,
                            color=group_color, alpha=0.15, linewidth=0, zorder=1,
                        )
                    psd_curves.append((group_color, freq, power_mean))
                    plotted_any = True

            if not plotted_any:
                log.warning(
                    "PSD '%s': no valid data for '%s'. Plot not saved.",
                    plot_name, ', '.join(channels_list),
                )
                plt.close(fig)
                continue

            has_x_limits, has_y_limits = self._apply_2d_axis_limits(
                ax, axis_limits, log_scale_y=log_scale,
            )
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

            # ── Gate annotation ───────────────────────────────────────────
            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    legend_obj = ax.get_legend()
                    self._display_gate_info(ax, gate_text, legend=legend_obj)

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

                    # Collect annotation points at this frequency.
                    # Skip curves whose Nyquist is below f_at — there is no
                    # real spectral content there, so annotating the nearest
                    # (final) bin would be misleading.
                    ann_items = []
                    for (run_color, freq_arr, power_arr) in psd_curves:
                        if len(freq_arr) == 0 or f_at > freq_arr[-1]:
                            continue
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

            self._draw_static_markers(ax, markers)

            # ── Reference lines (horizontal benchmark levels) ─────────────
            self._draw_horizontal_reference_lines(ax, reference_lines)

            # ── Nyquist line(s) — one per unique run sample rate ─────────
            # With resampling enabled all runs share one rate → one grey line.
            # If users disable RESAMPLE_RATE, per-run rates differ and each
            # is drawn in its run colour for clarity.
            if nyquist_lines:
                xl, xr = ax.get_xlim()
                single = len(nyquist_lines) == 1
                for nyq, color in nyquist_lines.items():
                    if not (xl <= nyq <= xr):
                        continue
                    line_color = "#5E5E5E" if single else color
                    ax.axvline(
                        nyq, color=line_color, linestyle=(0, (5, 3)),
                        linewidth=1.1, alpha=0.7, zorder=2,
                    )
                    label = "f_nyq"
                    ax.text(
                        nyq, 1.01, label,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=line_color,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor=line_color, linewidth=0.8, alpha=0.9),
                        zorder=12,
                    )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)

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
            gate_spec   = plot_def.gate
            reference_lines = plot_def.reference_lines

            if self.verbose:
                log.debug("Creating histogram plot: %s (%s)", plot_name, channel)

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
                if gate_spec is not None:
                    df = datafunctions.apply_gate_to_dataframe(df, gate_spec)
                    if df is None or df.empty:
                        continue
                vals = df[channel].dropna()
                if not vals.empty:
                    all_values.append(vals.values)

            if not all_values:
                log.warning(
                    "Histogram '%s': no valid data for '%s'. Plot not saved.",
                    plot_name, channel,
                )
                plt.close(fig)
                continue

            combined = np.concatenate(all_values)
            bins = datafunctions.compute_nice_histogram_bins(combined, num_bins=30)

            has_x_limits, has_y_limits = self._apply_2d_axis_limits(
                ax, axis_limits, log_scale_y=log_scale, y_floor=1e-6,
            )
            if has_x_limits and axis_limits[0][0] is not None and axis_limits[0][1] is not None:
                bins = datafunctions.compute_equal_width_bins_in_limits(
                    axis_limits[0][0], axis_limits[0][1], bins,
                )

            hist_data, hist_weights, hist_colors, hist_labels = [], [], [], []
            dt = 1.0 / self.FILTER_SAMPLE_RATE
            for run in self.runs:
                run_name = run["name"].lower()
                df = self.run_data.get(run_name)
                if df is None or channel not in df.columns:
                    continue
                if gate_spec is not None:
                    df = datafunctions.apply_gate_to_dataframe(df, gate_spec)
                    if df is None or df.empty:
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

            self._draw_horizontal_reference_lines(ax, reference_lines)

            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    legend_obj = ax.get_legend()
                    self._display_gate_info(ax, gate_text, legend=legend_obj)

            self._draw_static_markers(ax, markers)

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.05, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)


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
                log.debug("Creating heatmap plot: %s", plot_name)

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
                log.warning("Heatmap '%s': no usable runs. Skipping.", plot_name)
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
                self._draw_static_markers(ax, markers, x_clip=False)

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
                log.debug("Saved: %s", filename)
