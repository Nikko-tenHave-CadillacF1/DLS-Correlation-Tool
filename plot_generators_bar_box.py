"""Bar and Box plot generator mixin for DataPlotter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datafunctions

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable


class BarBoxMixin:
    """Bar and Box plot generation methods. Mixed into DataPlotter."""

    # ------------------------------------------------------------------
    # Bar plots
    # ------------------------------------------------------------------

    def generate_bar_plots(self):
        """Create grouped bar charts for aggregated channel metrics."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(4)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="Bar", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name = plot_def[0]
            metric_specs_raw = plot_def[1] if len(plot_def) > 1 else ()
            default_agg = plot_def[2] if len(plot_def) > 2 and isinstance(plot_def[2], str) else "last"
            axis_limits = plot_def[3] if len(plot_def) > 3 else None
            target_line = plot_def[4] if len(plot_def) > 4 else None

            metric_specs = datafunctions.normalize_bar_metric_specs(
                metric_specs_raw, default_aggregation=default_agg
            )
            if not metric_specs:
                if self.verbose:
                    print(f"[WARNING][DataPlotter] Bar plot '{plot_name}' has no valid metric specs. Skipping.")
                continue

            if self.verbose:
                print(f"Creating bar plot: {plot_name}")

            filename = self._sanitize_plot_filename("bar", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.histogram_FIGSIZE)
            fig, ax = plt.subplots(figsize=figsize)

            x = np.arange(len(metric_specs))
            loaded_runs = [run for run in self.runs if run["name"].lower() in self.run_data]
            if not loaded_runs:
                if self.verbose:
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
                        if self.verbose:
                            print(
                                f"[WARNING][DataPlotter] Bar plot '{plot_name}': "
                                f"missing channel '{channel}' in run '{run_name.upper()}'."
                            )
                        values.append(np.nan)
                        continue
                    values.append(datafunctions.aggregate_channel_for_bar(
                        df[channel],
                        aggregation=aggregation,
                        sample_rate=self.FILTER_SAMPLE_RATE,
                        time_series=df["tLap"] if "tLap" in df.columns else None,
                    ))

                offsets = x + left_edge + (run_index + 0.5) * bar_width
                run_bar_data.append({"run": run, "offsets": offsets, "values": np.array(values, dtype=float)})
                all_values.extend([abs(v) for v in values if not np.isnan(v)])

            # Detect if secondary axis is needed
            ax2 = None
            secondary_threshold = None
            if len(all_values) > 1:
                max_abs = max(all_values)
                candidate_threshold = max_abs / max(1.0, self.BAR_SECONDARY_AXIS_RATIO)
                lower_group = [v for v in all_values if v < candidate_threshold]
                if lower_group and max(lower_group) > 0:
                    ratio = max_abs / max(lower_group)
                    if ratio >= self.BAR_SECONDARY_AXIS_RATIO:
                        ax2 = ax.twinx()
                        ax2.spines["right"].set_visible(True)
                        ax2.spines["right"].set_color("black")
                        ax2.spines["right"].set_linewidth(2.0)
                        ax2.tick_params(axis="y", labelsize=10, colors="black", width=1.5)
                        secondary_threshold = candidate_threshold
                        print(
                            f"[INFO][DataPlotter] Bar plot '{plot_name}': secondary Y-axis activated "
                            f"(ratio={ratio:.1f}x ≥ threshold={self.BAR_SECONDARY_AXIS_RATIO:.0f}x)."
                        )

            plotted_labels = set()
            bar_info = []
            for item in run_bar_data:
                run = item["run"]
                offsets = item["offsets"]
                values = item["values"]
                run_label = run["name"].upper()

                if ax2 is not None:
                    primary_values = [
                        0.0 if np.isnan(v) or abs(v) >= secondary_threshold else v for v in values
                    ]
                    secondary_values = [
                        0.0 if np.isnan(v) or abs(v) < secondary_threshold else v for v in values
                    ]

                    p_label = run_label if run_label not in plotted_labels and any(v != 0.0 for v in primary_values) else "_nolegend_"
                    if p_label != "_nolegend_":
                        plotted_labels.add(p_label)
                    ax.bar(offsets, primary_values, width=bar_width, color=run["color"],
                           alpha=0.9, label=p_label, edgecolor="white", linewidth=0.6)

                    s_label = run_label if run_label not in plotted_labels and any(v != 0.0 for v in secondary_values) else "_nolegend_"
                    if s_label != "_nolegend_":
                        plotted_labels.add(s_label)
                    ax2.bar(offsets, secondary_values, width=bar_width, color=run["color"],
                            alpha=0.9, label=s_label, edgecolor="white", linewidth=0.6)

                    for offset, value in zip(offsets, values):
                        axis = ax2 if not np.isnan(value) and abs(value) >= secondary_threshold else ax
                        bar_info.append((offset, value, axis))
                else:
                    lbl = run_label if run_label not in plotted_labels else "_nolegend_"
                    plotted_labels.add(run_label)
                    ax.bar(offsets, values, width=bar_width, color=run["color"],
                           alpha=0.9, label=lbl, edgecolor="white", linewidth=0.6)
                    for offset, value in zip(offsets, values):
                        bar_info.append((offset, value, ax))

            ax.set_xticks(x)
            metric_labels = [f"{m}\n({a})" for m, a in metric_specs]
            ax.set_xticklabels(metric_labels, rotation=0, fontweight="bold")
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

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

            for offset, value, axis in bar_info:
                if not np.isnan(value):
                    y_range = axis_ranges.get(axis, 1.0)
                    padding = 0.02 * y_range
                    y_pos = value + (padding if value >= 0 else -padding)
                    va = "bottom" if value >= 0 else "top"
                    axis.text(offset, y_pos, f"{value:.2f}",
                              ha="center", va=va, fontsize=10, fontweight="bold", color="black")

            for axis in ([ax2, ax] if ax2 is not None else [ax]):
                y0, y1 = axis.get_ylim()
                if y0 <= 0 <= y1:
                    axis.axhline(0, color="#4F4F4F", linestyle="-", linewidth=1.0, alpha=0.9, zorder=1)

            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ax2 is not None:
                ax2.grid(True, axis="y", alpha=0.2)
                ax2.set_axisbelow(True)

            handles, labels = [], []
            for axis in ([ax, ax2] if ax2 is not None else [ax]):
                for h, l in zip(*axis.get_legend_handles_labels()):
                    if l and l != "_nolegend_" and l not in labels:
                        handles.append(h)
                        labels.append(l)

            self._add_standard_legend(ax, handles=handles, labels=labels, loc="upper right")

            if target_line is not None:
                tl = float(target_line)
                # Expand y-limits if the target line falls outside the current range
                y0, y1 = ax.get_ylim()
                padding = (y1 - y0) * 0.08
                new_y0 = min(y0, tl - padding)
                new_y1 = max(y1, tl + padding)
                ax.set_ylim(new_y0, new_y1)

                ax.axhline(
                    tl,
                    color="#333333", linestyle="--", linewidth=1.4, alpha=0.8, zorder=5,
                )
                ax.text(
                    0.99, tl,
                    f" {target_line:g}",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333",
                )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.05, facecolor="white")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")

    # ------------------------------------------------------------------
    # Box plot helpers
    # ------------------------------------------------------------------

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

        if channels is None:
            channels = []
        elif isinstance(channels, str):
            channels = [channels]
        else:
            channels = list(channels)

        return plot_name, channels, aggregation_mode, axis_limits, gate_spec, options or {}

    def _collect_boxplot_point_series_from_data(self, channel, filtered_run_data):
        """Collect per-run point series using prefiltered run data."""
        series = []
        for run in self.runs:
            run_name = run["name"].lower()
            df = filtered_run_data.get(run_name)
            if df is None or channel not in df.columns:
                continue
            values = pd.to_numeric(df[channel], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                series.append((run["name"].upper(), run["color"], values))
        return series

    def _apply_boxplot_artist_style(self, bp, box_settings, colors=None, facecolor=None, alpha=0.7):
        """Apply consistent styling to a matplotlib boxplot result."""
        box_lw = box_settings.get("box_linewidth", 1.5)
        median_color = box_settings.get("medianline_color", "#000000")
        median_width = box_settings.get("medianline_width", 2.0)
        whisker_color = box_settings.get("box_edge_color", "#4A4A4A")

        for idx, patch in enumerate(bp.get("boxes", [])):
            patch.set_linewidth(box_lw)
            if colors and idx < len(colors):
                patch.set_facecolor(colors[idx])
            elif facecolor is not None:
                patch.set_facecolor(facecolor)
            patch.set_alpha(alpha)

        for item in bp.get("whiskers", []):
            item.set(color=whisker_color, linewidth=box_lw)
        for item in bp.get("caps", []):
            item.set(color=whisker_color, linewidth=box_lw)
        for median in bp.get("medians", []):
            median.set(color=median_color, linewidth=median_width)
        for flier in bp.get("fliers", []):
            flier.set(markerfacecolor=median_color, markeredgecolor=median_color, alpha=0.7)

    # ------------------------------------------------------------------
    # Box plot generator
    # ------------------------------------------------------------------

    def generate_box_plots(self):
        """Generate box plots for distribution analysis across runs."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(5)
        if not plots:
            return

        box_settings = getattr(self, "BOX_PLOT_SETTINGS", {})
        plot_iter = plots if self.verbose else _tqdm(plots, desc="Box", unit="plot", leave=True)
        for plot_def in plot_iter:
            try:
                plot_name, channels, aggregation_mode, axis_limits, gate_spec, options = (
                    self._parse_boxplot_definition(plot_def)
                )
            except ValueError as exc:
                if self.verbose:
                    print(f"[WARNING][DataPlotter] {exc} Skipping box plot: {plot_def!r}")
                continue

            if not channels:
                if self.verbose:
                    print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no channels. Skipping.")
                continue

            if self.verbose:
                print(f"Creating box plot: {plot_name}")

            plot_options = {**box_settings, **(options or {})}
            figsize = (
                box_settings.get("figsize_single_channel", self.boxplot_FIGSIZE)
                if len(channels) == 1
                else box_settings.get("figsize_multi_channel", self.boxplot_FIGSIZE)
            )

            if aggregation_mode == "per_run":
                self._generate_boxplot_per_run(
                    plot_name, channels, axis_limits, gate_spec, plot_options, figsize
                )
            elif aggregation_mode == "aggregated":
                self._generate_boxplot_aggregated(
                    plot_name, channels, axis_limits, gate_spec, plot_options, figsize
                )
            else:
                if self.verbose:
                    print(
                        f"[WARNING][DataPlotter] Box plot '{plot_name}': "
                        f"unknown aggregation_mode '{aggregation_mode}'. Skipping."
                    )

    def _generate_boxplot_per_run(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
        """Generate per-run box plots (one box per run per channel)."""
        box_settings = options
        filtered_run_data = {
            rn: self._get_filtered_run_dataframe(rn, gate_spec) for rn in self.run_data
        }
        agg_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data, channels,
            aggregation_mode="per_run", gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )
        if not agg_data:
            if self.verbose:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no data after aggregation.")
            return

        run_names = [r["name"].lower() for r in self.runs if r["name"].lower() in agg_data]
        if not run_names:
            return

        run_colors = {r["name"].lower(): r["color"] for r in self.runs}
        show_points = bool(box_settings.get("show_points", False))
        show_fliers = bool(box_settings.get("show_fliers", True))
        point_alpha = float(box_settings.get("point_alpha", 0.25))
        point_size = float(box_settings.get("point_size", 18))
        jitter = float(box_settings.get("jitter", 0.15))
        box_width = float(box_settings.get("box_width", 0.6))
        gate_text = datafunctions.format_gate_text(gate_spec) if gate_spec is not None else None

        num_channels = len(channels)
        if num_channels == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        else:
            fig, axes = plt.subplots(
                num_channels, 1,
                figsize=(figsize[0], max(figsize[1], 4.5) * num_channels * 0.62),
                sharex=False,
            )
            axes = list(np.atleast_1d(axes))

        fig.suptitle(plot_name, fontsize=16, fontweight="bold")
        rng = np.random.default_rng(42)

        for ax, channel in zip(axes, channels):
            data_list, labels_list, colors_list, overlay_series = [], [], [], []
            for run_name in run_names:
                if channel not in agg_data.get(run_name, {}):
                    continue
                values = agg_data[run_name][channel]
                if len(values) == 0:
                    continue
                data_list.append(values)
                labels_list.append(run_name.upper())
                colors_list.append(run_colors.get(run_name, "#3498DB"))
                if show_points:
                    overlay_series.append((run_name.upper(), run_colors.get(run_name, "#3498DB"), values))

            if not data_list:
                if self.verbose:
                    print(f"[WARNING][DataPlotter] Box plot '{plot_name}' channel '{channel}': no data.")
                continue

            bp = ax.boxplot(
                data_list, labels=labels_list, patch_artist=True,
                widths=box_width, showfliers=show_fliers,
            )
            self._apply_boxplot_artist_style(
                bp, box_settings, colors=colors_list,
                alpha=box_settings.get("per_run_box_alpha", 0.7),
            )

            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight="bold", fontsize=12,
            )
            ax.set_title(channel, fontweight="bold", fontsize=12)

            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)

            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            if show_points:
                for box_index, (_, color, values) in enumerate(overlay_series, start=1):
                    x_pts = np.full(len(values), box_index, dtype=float)
                    x_pts += rng.uniform(-jitter, jitter, size=len(values))
                    ax.scatter(x_pts, values, s=point_size, alpha=point_alpha,
                               color=color, edgecolors="none", zorder=3)

        if gate_text:
            self._display_gate_info(axes[0], gate_text)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.05, facecolor="white")
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {filename}")

    def _generate_boxplot_aggregated(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
        """Generate aggregated box plots (all runs combined into one box per channel)."""
        box_settings = options
        filtered_run_data = {
            rn: self._get_filtered_run_dataframe(rn, gate_spec) for rn in self.run_data
        }
        agg_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data, channels,
            aggregation_mode="aggregated", gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )
        if not agg_data:
            if self.verbose:
                print(f"[WARNING][DataPlotter] Box plot '{plot_name}': no aggregated data.")
            return

        show_points = bool(box_settings.get("show_points", False))
        show_fliers = bool(box_settings.get("show_fliers", True))
        point_alpha = float(box_settings.get("point_alpha", 0.25))
        point_size = float(box_settings.get("point_size", 18))
        jitter = float(box_settings.get("jitter", 0.15))
        box_width = float(box_settings.get("box_width", 0.6))
        agg_color = box_settings.get("aggregated_box_color", "#3498DB")
        agg_alpha = float(box_settings.get("aggregated_box_alpha", 0.7))
        gate_text = datafunctions.format_gate_text(gate_spec) if gate_spec is not None else None

        num_channels = len(channels)
        if num_channels == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        else:
            fig, axes = plt.subplots(
                num_channels, 1,
                figsize=(figsize[0], max(figsize[1], 4.5) * num_channels * 0.62),
                sharex=False,
            )
            axes = list(np.atleast_1d(axes))

        fig.suptitle(plot_name, fontsize=16, fontweight="bold")
        rng = np.random.default_rng(42)
        legend_handles, legend_labels = [], []

        for ax, channel in zip(axes, channels):
            data = agg_data.get(channel, [])
            if not len(data):
                if self.verbose:
                    print(f"[WARNING][DataPlotter] Box plot '{plot_name}' channel '{channel}': no data.")
                continue

            bp = ax.boxplot([data], labels=[channel], patch_artist=True,
                            widths=box_width, showfliers=show_fliers)
            self._apply_boxplot_artist_style(bp, box_settings, facecolor=agg_color, alpha=agg_alpha)

            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight="bold", fontsize=12,
            )
            ax.set_title(channel, fontweight="bold", fontsize=12)

            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)

            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            if show_points:
                overlay_series = self._collect_boxplot_point_series_from_data(
                    channel, filtered_run_data
                )
                for run_label, color, values in overlay_series:
                    x_pts = np.full(len(values), 1.0, dtype=float)
                    x_pts += rng.uniform(-jitter, jitter, size=len(values))
                    ax.scatter(x_pts, values, s=point_size, alpha=point_alpha,
                               color=color, edgecolors="none", zorder=3)
                    if run_label not in legend_labels:
                        legend_handles.append(
                            Line2D([0], [0], marker="o", linestyle="None", color=color)
                        )
                        legend_labels.append(run_label)

        if show_points and legend_handles:
            self._add_standard_legend(axes[0], handles=legend_handles, labels=legend_labels,
                                      loc="upper right")
        if gate_text:
            legend_obj = axes[0].get_legend() if show_points and legend_handles else None
            self._display_gate_info(axes[0], gate_text, legend=legend_obj)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.05, facecolor="white")
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {filename}")
