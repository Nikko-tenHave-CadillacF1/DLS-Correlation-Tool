
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from . import datafunctions
from .datafunctions import _tqdm
from .logger import log

class BarBoxMixin:

    def generate_bar_plots(self):
        self._ensure_preprocessed()
        plots = self._get_plot_group(4)
        if not plots:
            return
        plot_iter = plots if self.verbose else _tqdm(plots, desc="Bar", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name = plot_def.name
            metric_specs_raw = plot_def.metrics or ()
            default_agg = plot_def.default_aggregation
            axis_limits = plot_def.axis_limits
            reference_lines = plot_def.reference_lines
            gate_spec = plot_def.gate
            error_metrics = getattr(plot_def, "error_metrics", None)
            metric_specs = datafunctions.normalize_bar_metric_specs(
                metric_specs_raw, default_aggregation=default_agg
            )
            if not metric_specs:
                log.warning("Bar plot '%s' has no valid metric specs. Skipping.", plot_name)
                continue
            if self.verbose:
                log.debug("Creating bar plot: %s", plot_name)
            filename = self._sanitize_plot_filename("bar", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.bar_FIGSIZE)
            fig, ax = plt.subplots(figsize=figsize)
            x = np.arange(len(metric_specs))
            loaded_runs = [run for run in self.runs if run["name"].lower() in self.run_data]
            if not loaded_runs:
                log.warning("Bar plot '%s' has no loaded runs. Skipping.", plot_name)
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
                if gate_spec is not None:
                    df = datafunctions.apply_gate_to_dataframe(df, gate_spec)
                    if df is None or df.empty:
                        run_bar_data.append({"run": run, "offsets": x + left_edge + (run_index + 0.5) * bar_width,
                                             "values": np.full(len(metric_specs), np.nan),
                                             "errors": None})
                        continue
                values = []
                for channel, aggregation in metric_specs:
                    if channel not in df.columns:
                        hint = self._format_missing_channel_hint(run_name, channel)
                        log.warning(
                            "Bar plot '%s': missing channel '%s' in run '%s'.%s",
                            plot_name, channel, run_name.upper(),
                            f"\n{hint}" if hint else "",
                        )
                        values.append(np.nan)
                        continue
                    values.append(datafunctions.aggregate_channel_for_bar(
                        df[channel],
                        aggregation=aggregation,
                        sample_rate=self._run_fs(run_name),
                        time_series=df["tLap"] if "tLap" in df.columns else None,
                    ))
                errors = None
                if error_metrics:
                    errors = []
                    for idx, (channel, aggregation) in enumerate(metric_specs):
                        err_ch = error_metrics[idx] if idx < len(error_metrics) else None
                        if not err_ch or err_ch not in df.columns:
                            errors.append(np.nan)
                            continue
                        errors.append(datafunctions.aggregate_channel_for_bar(
                            df[err_ch],
                            aggregation=aggregation,
                            sample_rate=self._run_fs(run_name),
                            time_series=df["tLap"] if "tLap" in df.columns else None,
                        ))
                offsets = x + left_edge + (run_index + 0.5) * bar_width
                run_bar_data.append({
                    "run": run, "offsets": offsets,
                    "values": np.array(values, dtype=float),
                    "errors": (np.array(errors, dtype=float) if errors is not None else None),
                })
                all_values.extend([abs(v) for v in values if not np.isnan(v)])
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
                        log.info(
                            "Bar plot '%s': secondary Y-axis activated (ratio=%.1fx ≥ threshold=%.0fx).",
                            plot_name, ratio, self.BAR_SECONDARY_AXIS_RATIO,
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
                errs = item.get("errors")
                if errs is not None and ax2 is None:
                    err_finite = np.array([
                        e if (np.isfinite(e) and not np.isnan(v)) else np.nan
                        for v, e in zip(values, errs)
                    ])
                    if np.isfinite(err_finite).any():
                        ax.errorbar(
                            offsets, values, yerr=err_finite,
                            fmt="none", ecolor="#1A1A1A", elinewidth=1.4,
                            capsize=4, capthick=1.4, alpha=0.85, zorder=4,
                        )
            ax.set_xticks(x)
            metric_labels = [f"{m}\n({a})" for m, a in metric_specs]
            ax.set_xticklabels(metric_labels, rotation=0, fontweight="bold")
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            if len(metric_specs) == 1:
                channel_name = metric_specs[0][0]
                ax.set_ylabel(
                    datafunctions.add_units_to_label(channel_name, self.units_map),
                    fontweight="bold", fontsize=12,
                )
            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)
            self._add_axis_edge_padding(ax, x_pad_ratio=0.06, y_pad_ratio=0.04)
            if ax2 is not None:
                self._add_axis_edge_padding(ax2, x_pad_ratio=0.06, y_pad_ratio=0.04)
            # When many runs share the same metric group the bars become
            # thin and horizontal value labels overlap. Rotate the labels
            # 90° (and shrink the font slightly) once we exceed ~4 runs,
            # and add extra y-padding so the rotated text doesn't clip.
            many_runs = len(loaded_runs) >= 5
            label_rotation = 90 if many_runs else 0
            label_fontsize = 8 if many_runs else 10
            label_pad_ratio = 0.04 if many_runs else 0.02
            if many_runs:
                # Vertical labels need headroom equal to roughly the
                # longest formatted value rendered upright; ~12% of the
                # axis range covers a typical "0.0053599"-length label.
                for axis in ([ax2, ax] if ax2 is not None else [ax]):
                    y0, y1 = axis.get_ylim()
                    extra = 0.12 * (y1 - y0)
                    if y1 > 0:
                        axis.set_ylim(top=y1 + extra)
                    if y0 < 0:
                        axis.set_ylim(bottom=y0 - extra)
            axis_ranges = {ax: ax.get_ylim()[1] - ax.get_ylim()[0]}
            if ax2 is not None:
                axis_ranges[ax2] = ax2.get_ylim()[1] - ax2.get_ylim()[0]
            for offset, value, axis in bar_info:
                if not np.isnan(value):
                    y_range = axis_ranges.get(axis, 1.0)
                    padding = label_pad_ratio * y_range
                    y_pos = value + (padding if value >= 0 else -padding)
                    va = "bottom" if value >= 0 else "top"
                    axis.text(offset, y_pos, datafunctions._fmt_g(value, sig=5),
                              ha="center", va=va, fontsize=label_fontsize,
                              fontweight="bold", color="#1A1A1A",
                              rotation=label_rotation)
            for axis in ([ax2, ax] if ax2 is not None else [ax]):
                y0, y1 = axis.get_ylim()
                if y0 <= 0 <= y1:
                    axis.axhline(0, color="#4F4F4F", linestyle="-", linewidth=1.0, alpha=0.9, zorder=1)
            self._apply_grid(ax, which="major", axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ax2 is not None:
                ax2.grid(True, axis="y", alpha=self.GRID_STYLE["minor"]["alpha"])
                ax2.set_axisbelow(True)
            handles, labels = [], []
            for axis in ([ax, ax2] if ax2 is not None else [ax]):
                for h, l in zip(*axis.get_legend_handles_labels()):
                    if l and l != "_nolegend_" and l not in labels:
                        handles.append(h)
                        labels.append(l)
            self._add_standard_legend(ax, handles=handles, labels=labels, loc="upper right")
            self._draw_horizontal_reference_lines(ax, reference_lines)
            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    legend_obj = ax.get_legend()
                    self._display_gate_info(ax, gate_text, legend=legend_obj)
            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)
    def _parse_boxplot_definition(self, plot_def):
        plot_name = plot_def.name
        channels = list(plot_def.channels) if plot_def.channels else []
        aggregation_mode = plot_def.aggregation_mode
        axis_limits = plot_def.axis_limits
        gate_spec = plot_def.gate
        options = dict(plot_def.options or {})
        if plot_def.reference_lines is not None and "_reference_lines" not in options:
            options["_reference_lines"] = plot_def.reference_lines
        return plot_name, channels, aggregation_mode, axis_limits, gate_spec, options
    def _collect_boxplot_point_series_from_data(self, channel, filtered_run_data):
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
    def _apply_boxplot_artist_style(self, bp, box_settings, colors=None, facecolor=None, alpha=0.82):
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
            _MAX_FLIERS = 200
            xdata = flier.get_xdata()
            ydata = flier.get_ydata()
            if len(xdata) > _MAX_FLIERS:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(xdata), size=_MAX_FLIERS, replace=False)
                flier.set_xdata(xdata[idx])
                flier.set_ydata(ydata[idx])
    def generate_box_plots(self):
        self._ensure_preprocessed()
        plots = self._get_plot_group(5)
        if not plots:
            return
        box_settings = getattr(self, "BOX_PLOT_SETTINGS", {})
        plot_iter = plots if self.verbose else _tqdm(plots, desc="Box", unit="plot", leave=True)
        for plot_def in plot_iter:
            if getattr(plot_def, "kind", None) == "box_grid":
                self._generate_boxplot_grid(plot_def, box_settings)
                continue
            try:
                plot_name, channels, aggregation_mode, axis_limits, gate_spec, options = (
                    self._parse_boxplot_definition(plot_def)
                )
            except ValueError as exc:
                log.warning("%s Skipping box plot: %r", exc, plot_def)
                continue
            if not channels:
                log.warning("Box plot '%s': no channels. Skipping.", plot_name)
                continue
            if self.verbose:
                log.debug("Creating box plot: %s", plot_name)
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
            elif aggregation_mode == "per_run_aggregated":
                self._generate_boxplot_per_run_aggregated(
                    plot_name, channels, axis_limits, gate_spec, plot_options, figsize
                )
            else:
                log.warning(
                    "Box plot '%s': unknown aggregation_mode '%s'. Skipping.",
                    plot_name, aggregation_mode,
                )
    def _generate_boxplot_per_run(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
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
            log.warning("Box plot '%s': no data after aggregation.", plot_name)
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
                log.warning("Box plot '%s' channel '%s': no data.", plot_name, channel)
                continue
            bp = ax.boxplot(
                data_list, labels=labels_list, patch_artist=True,
                widths=box_width, showfliers=show_fliers, whis=[5, 95]
            )
            self._apply_boxplot_artist_style(
                bp, box_settings, colors=colors_list,
                alpha=box_settings.get("per_run_box_alpha", 0.7),
            )
            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight="bold", fontsize=12,
            )
            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            self._apply_grid(ax, which="both", axis="y")
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
        ref_lines_spec = options.get("_reference_lines") if isinstance(options, dict) else None
        if ref_lines_spec:
            for ax in axes:
                self._draw_horizontal_reference_lines(ax, ref_lines_spec)
        plot_title = box_settings.get("title")
        if plot_title:
            fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout(pad=0.25)
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            log.debug("Saved: %s", filename)
    def _generate_boxplot_aggregated(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
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
            log.warning("Box plot '%s': no aggregated data.", plot_name)
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
        rng = np.random.default_rng(42)
        legend_handles, legend_labels = [], []
        for ax, channel in zip(axes, channels):
            data = agg_data.get(channel, [])
            if not len(data):
                log.warning("Box plot '%s' channel '%s': no data.", plot_name, channel)
                continue
            bp = ax.boxplot([data], labels=[channel], patch_artist=True,
                            widths=box_width, showfliers=show_fliers, whis=[5, 95])
            self._apply_boxplot_artist_style(bp, box_settings, facecolor=agg_color, alpha=agg_alpha)
            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight="bold", fontsize=12,
            )
            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            self._apply_grid(ax, which="both", axis="y")
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
        ref_lines_spec = options.get("_reference_lines") if isinstance(options, dict) else None
        if ref_lines_spec:
            for ax in axes:
                self._draw_horizontal_reference_lines(ax, ref_lines_spec)
        plot_title = box_settings.get("title")
        if plot_title:
            fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout(pad=0.25)
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            log.debug("Saved: %s", filename)
    def _generate_boxplot_per_run_aggregated(self, plot_name, channels, axis_limits, gate_spec, options, figsize):
        box_settings = options
        filtered_run_data = {
            rn: self._get_filtered_run_dataframe(rn, gate_spec) for rn in self.run_data
        }
        per_run_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data, channels,
            aggregation_mode="per_run", gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )
        agg_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data, channels,
            aggregation_mode="aggregated", gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )
        if not per_run_data and not agg_data:
            log.warning("Box plot '%s': no data after aggregation.", plot_name)
            return
        run_names = [r["name"].lower() for r in self.runs if r["name"].lower() in per_run_data]
        run_colors = {r["name"].lower(): r["color"] for r in self.runs}
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
        wider_figsize = (figsize[0] * 1.15, figsize[1])
        if num_channels == 1:
            fig, axes = plt.subplots(1, 1, figsize=wider_figsize)
            axes = [axes]
        else:
            fig, axes = plt.subplots(
                num_channels, 1,
                figsize=(wider_figsize[0], max(wider_figsize[1], 4.5) * num_channels * 0.62),
                sharex=False,
            )
            axes = list(np.atleast_1d(axes))
        rng = np.random.default_rng(42)
        for ax, channel in zip(axes, channels):
            data_list, labels_list, colors_list, overlay_series = [], [], [], []
            for run_name in run_names:
                if channel not in per_run_data.get(run_name, {}):
                    continue
                values = per_run_data[run_name][channel]
                if len(values) == 0:
                    continue
                data_list.append(values)
                labels_list.append(run_name.upper())
                colors_list.append(run_colors.get(run_name, "#3498DB"))
                if show_points:
                    overlay_series.append((run_name.upper(), run_colors.get(run_name, "#3498DB"), values))
            agg_values = agg_data.get(channel, np.array([]))
            if len(agg_values) > 0:
                data_list.append(agg_values)
                labels_list.append("ALL")
                colors_list.append(agg_color)
            if not data_list:
                log.warning("Box plot '%s' channel '%s': no data.", plot_name, channel)
                continue
            bp = ax.boxplot(
                data_list, labels=labels_list, patch_artist=True,
                widths=box_width, showfliers=show_fliers, whis=[5, 95]
            )
            per_run_alpha = box_settings.get("per_run_box_alpha", 0.7)
            for i, patch in enumerate(bp["boxes"]):
                color = colors_list[i]
                alpha = agg_alpha if labels_list[i] == "ALL" else per_run_alpha
                patch.set_facecolor(color)
                patch.set_alpha(alpha)
                patch.set_edgecolor("#2C3E50")
                patch.set_linewidth(1.2)
            for median in bp.get("medians", []):
                median.set(color="#2C3E50", linewidth=2)
            for whisker in bp.get("whiskers", []):
                whisker.set(color="#2C3E50", linewidth=1.2)
            for cap in bp.get("caps", []):
                cap.set(color="#2C3E50", linewidth=1.2)
            if not show_fliers:
                for flier in bp.get("fliers", []):
                    flier.set_visible(False)
            if len(agg_values) > 0:
                sep_x = len(data_list) - 0.5
                ax.axvline(sep_x, color="#AAAAAA", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.set_ylabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontweight="bold", fontsize=12,
            )
            if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                ymin, ymax = axis_limits
                if ymin is not None or ymax is not None:
                    ax.set_ylim(bottom=ymin, top=ymax)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            self._apply_grid(ax, which="both", axis="y")
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
        ref_lines_spec = options.get("_reference_lines") if isinstance(options, dict) else None
        if ref_lines_spec:
            for ax in axes:
                self._draw_horizontal_reference_lines(ax, ref_lines_spec)
        plot_title = box_settings.get("title")
        if plot_title:
            fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout(pad=0.25)
        filename = self._sanitize_plot_filename("box", plot_name)
        fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            log.debug("Saved: %s", filename)
    def _generate_boxplot_grid(self, grid_def, box_settings):
        from .plot_definitions import _normalise_gate_list
        plot_name = grid_def.name
        channels = list(grid_def.channels)
        aggregation_mode = grid_def.aggregation_mode
        axis_limits = grid_def.axis_limits
        options = {**box_settings, **(grid_def.options or {})}
        row_labels = list(grid_def.rows.keys())
        col_labels = list(grid_def.cols.keys())
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        show_fliers = bool(options.get("show_fliers", True))
        show_points = bool(options.get("show_points", False))
        point_alpha = float(options.get("point_alpha", 0.25))
        point_size = float(options.get("point_size", 18))
        jitter = float(options.get("jitter", 0.15))
        box_width = float(options.get("box_width", 0.6))
        agg_color = options.get("aggregated_box_color", "#3498DB")
        agg_alpha = float(options.get("aggregated_box_alpha", 0.7))
        rng = np.random.default_rng(42)
        for channel in channels:
            cell_w = options.get("grid_cell_width", 4.0)
            cell_h = options.get("grid_cell_height", 2.8)
            fig_w = max(cell_w * n_cols + 1.5, 10)
            fig_h = max(cell_h * n_rows + 1.2, 4)
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                sharey="row",
            )
            for r_idx, row_label in enumerate(row_labels):
                row_gate = grid_def.rows[row_label]
                for c_idx, col_label in enumerate(col_labels):
                    col_gate = grid_def.cols[col_label]
                    combined_gate = _normalise_gate_list(row_gate) + _normalise_gate_list(col_gate)
                    ax = axes[r_idx, c_idx]
                    if aggregation_mode == "aggregated":
                        self._render_grid_cell_aggregated(
                            ax, channel, combined_gate, options,
                            show_fliers, show_points, point_alpha, point_size,
                            jitter, box_width, agg_color, agg_alpha, rng,
                        )
                    elif aggregation_mode == "per_run":
                        self._render_grid_cell_per_run(
                            ax, channel, combined_gate, options,
                            show_fliers, show_points, point_alpha, point_size,
                            jitter, box_width, rng,
                        )
                    elif aggregation_mode == "per_run_aggregated":
                        self._render_grid_cell_per_run(
                            ax, channel, combined_gate, options,
                            show_fliers, show_points, point_alpha, point_size,
                            jitter, box_width, rng, append_aggregated=True,
                            agg_color=agg_color, agg_alpha=agg_alpha,
                        )
                    if c_idx == 0:
                        ax.set_ylabel(row_label, fontweight="bold", fontsize=11)
                    if r_idx == 0:
                        ax.set_title(col_label, fontweight="bold", fontsize=11)
                    if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
                        ymin, ymax = axis_limits
                        if ymin is not None or ymax is not None:
                            ax.set_ylim(bottom=ymin, top=ymax)
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                    self._apply_grid(ax, which="both", axis="y")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    yl, yr = ax.get_ylim()
                    if yl <= 0 <= yr:
                        ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)
            event_title = options.get("title", "")
            channel_label = datafunctions.add_units_to_label(channel, self.units_map)
            suptitle = f"{plot_name} — {channel_label}"
            if event_title:
                suptitle = f"{event_title} | {suptitle}"
            fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout(pad=0.4)
            grid_filename = self._sanitize_plot_filename("box_grid", f"{plot_name}_{channel}")
            fig.savefig(
                self.plots_dir / grid_filename,
                dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight",
            )
            plt.close(fig)
            if self.verbose:
                log.debug("Saved grid: %s", grid_filename)
    def _render_grid_cell_aggregated(self, ax, channel, gate_spec, options,
                                     show_fliers, show_points, point_alpha, point_size,
                                     jitter, box_width, agg_color, agg_alpha, rng):
        filtered_run_data = {
            rn: self._get_filtered_run_dataframe(rn, gate_spec) for rn in self.run_data
        }
        agg_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data, [channel],
            aggregation_mode="aggregated", gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )
        data = agg_data.get(channel, np.array([]))
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999999")
            ax.set_xticks([])
            return
        bp = ax.boxplot([data], labels=[channel], patch_artist=True,
                        widths=box_width, showfliers=show_fliers, whis=[5, 95])
        self._apply_boxplot_artist_style(bp, options, facecolor=agg_color, alpha=agg_alpha)
        ax.set_xticklabels([])
        if show_points:
            overlay_series = self._collect_boxplot_point_series_from_data(channel, filtered_run_data)
            for _, color, values in overlay_series:
                x_pts = np.full(len(values), 1.0, dtype=float)
                x_pts += rng.uniform(-jitter, jitter, size=len(values))
                ax.scatter(x_pts, values, s=point_size, alpha=point_alpha,
                           color=color, edgecolors="none", zorder=3)
    def _render_grid_cell_per_run(self, ax, channel, gate_spec, options,
                                  show_fliers, show_points, point_alpha, point_size,
                                  jitter, box_width, rng,
                                  append_aggregated=False, agg_color="#3498DB", agg_alpha=0.7):
        filtered_run_data = {
            rn: self._get_filtered_run_dataframe(rn, gate_spec) for rn in self.run_data
        }
        per_run_data = datafunctions.aggregate_channel_for_boxplot(
            self.run_data, [channel],
            aggregation_mode="per_run", gate_spec=gate_spec,
            filtered_run_data=filtered_run_data,
        )
        run_names = [r["name"].lower() for r in self.runs if r["name"].lower() in per_run_data]
        run_colors = {r["name"].lower(): r["color"] for r in self.runs}
        data_list, labels_list, colors_list = [], [], []
        for run_name in run_names:
            if channel not in per_run_data.get(run_name, {}):
                continue
            values = per_run_data[run_name][channel]
            if len(values) == 0:
                continue
            data_list.append(values)
            labels_list.append(run_name.upper())
            colors_list.append(run_colors.get(run_name, "#3498DB"))
        if append_aggregated:
            agg_data = datafunctions.aggregate_channel_for_boxplot(
                self.run_data, [channel],
                aggregation_mode="aggregated", gate_spec=gate_spec,
                filtered_run_data=filtered_run_data,
            )
            agg_values = agg_data.get(channel, np.array([]))
            if len(agg_values) > 0:
                data_list.append(agg_values)
                labels_list.append("ALL")
                colors_list.append(agg_color)
        if not data_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999999")
            ax.set_xticks([])
            return
        bp = ax.boxplot(
            data_list, labels=labels_list, patch_artist=True,
            widths=box_width, showfliers=show_fliers, whis=[5, 95],
        )
        self._apply_boxplot_artist_style(bp, options, colors=colors_list, alpha=0.7)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        if show_points:
            for box_index, run_name in enumerate(run_names, start=1):
                if channel not in per_run_data.get(run_name, {}):
                    continue
                values = per_run_data[run_name][channel]
                if len(values) == 0:
                    continue
                color = run_colors.get(run_name, "#3498DB")
                x_pts = np.full(len(values), box_index, dtype=float)
                x_pts += rng.uniform(-jitter, jitter, size=len(values))
                ax.scatter(x_pts, values, s=point_size, alpha=point_alpha,
                           color=color, edgecolors="none", zorder=3)
