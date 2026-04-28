"""Scatter plot generator mixin for DataPlotter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datafunctions

try:
    from tqdm import tqdm as _tqdm_raw
    def _tqdm(it, **kw): return _tqdm_raw(it, file=__import__('sys').stderr, dynamic_ncols=True, force=True, **kw)
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable


def _resolve_scatter_style(point_count, base_size, base_alpha):
    """Tune scatter styling slightly for dense plots."""
    if point_count <= 5000:
        return base_size, base_alpha
    if point_count <= 20000:
        return max(3.5, base_size * 0.9), min(0.75, base_alpha + 0.05)
    if point_count <= 60000:
        return max(3.0, base_size * 0.8), max(0.35, base_alpha * 0.8)
    return max(2.5, base_size * 0.7), max(0.22, base_alpha * 0.65)


class ScatterMixin:
    """Scatter plot generation methods. Mixed into DataPlotter."""

    # ------------------------------------------------------------------
    # Scatter helpers
    # ------------------------------------------------------------------

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
                labels.append(f"{axis_name}: [−∞, {max_val:g}]")
            elif max_val is None:
                labels.append(f"{axis_name}: [{min_val:g}, +∞]")
            else:
                labels.append(f"{axis_name}: [{min_val:g}, {max_val:g}]")

        return labels if labels else None

    def _select_trendline_anchor(self, ax, equations_list):
        """Place text in the least-crowded position (corners + mid-edges)."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xs, ys = [], []
        for _, _, _, xv, yv, _ in equations_list:
            xs.extend(xv)
            ys.extend(yv)
        xs = np.array(xs)
        ys = np.array(ys)

        candidates = {
            "tl": (0.03, 0.97, "left",   "top"),
            "tr": (0.97, 0.97, "right",  "top"),
            "bl": (0.03, 0.03, "left",   "bottom"),
            "br": (0.97, 0.03, "right",  "bottom"),
            "tc": (0.50, 0.97, "center", "top"),
            "bc": (0.50, 0.03, "center", "bottom"),
        }

        def _count_pts(key):
            xa, ya, hal, val = candidates[key]
            w = (x1 - x0) * 0.22
            h = (y1 - y0) * 0.28
            x_abs = x0 + xa * (x1 - x0)
            if hal == "left":
                x_min = x_abs
            elif hal == "right":
                x_min = x_abs - w
            else:  # center
                x_min = x_abs - w / 2
            x_max = x_min + w
            y_min = y0 + ya * (y1 - y0) - h if val == "top" else y0 + ya * (y1 - y0)
            y_max = y_min + h
            return ((xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)).sum()

        best = min(candidates.keys(), key=_count_pts)
        return candidates[best]

    def _format_trendline_text(self, label, equation):
        """Format trendline text: single line for one fit, grouped header for multi-segment."""
        raw_lines = [l.strip() for l in str(equation).splitlines() if l.strip()]
        if not raw_lines:
            return f"{label}  fit unavailable"
        if len(raw_lines) == 1:
            line = raw_lines[0]
            if not line.startswith(label):
                line = f"{label}  {line}"
            return line
        # Multi-segment: label as header, each segment indented below
        out = [label]
        for line in raw_lines:
            # Strip any existing label prefix (backward compatibility)
            if line.upper().startswith(label.upper() + " "):
                line = line[len(label):].lstrip(" :(")
            out.append(f"  {line}")
        return "\n".join(out)

    def _display_equations(self, ax, eq_list):
        """Render trendline equation callouts and return their anchor metadata."""
        x_anchor, y_anchor, halign, valign = self._select_trendline_anchor(ax, eq_list)
        fig_height = ax.get_figure().get_size_inches()[1]
        line_height = 0.042 * 8.0 / max(fig_height, 4.0)
        box_gap = 0.015
        boxes = []
        cursor = y_anchor

        for label, equation, color, _, _, _ in eq_list:
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
                x_anchor, ypos, text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment=valign,
                horizontalalignment=halign,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="white", alpha=0.9,
                    edgecolor=color, linewidth=1.6,
                ),
                color=color, fontweight="bold", family="Montserrat",
            )
            boxes.append(ypos)

        return x_anchor, halign, valign, boxes

    def _format_gradient_error_text(
        self, equations_list, x_var=None, y_var=None, fit_labels=None
    ):
        """Create baseline-relative gradient error text."""
        if len(equations_list) < 2:
            return None

        baseline_target = self.runs[0]["name"].upper() if self.runs else None
        baseline_entry = next(
            (e for e in equations_list if e[0].upper() == baseline_target),
            equations_list[0],
        )
        baseline_label, _, _, _, _, baseline_slopes = baseline_entry
        comparison_entries = [e for e in equations_list if e is not baseline_entry]
        if not comparison_entries:
            return None

        def percent_error(value, baseline):
            if value is None or baseline is None or baseline == 0:
                return None
            return ((value - baseline) / baseline) * 100

        def fmt(value):
            return "n/a" if value is None else f"{value:+.1f}%"

        lines = [f"Gradient Error vs {baseline_label.upper()}"]

        if isinstance(baseline_slopes, tuple):
            for idx in range(len(baseline_slopes)):
                segment_name = (
                    fit_labels[idx] if fit_labels and idx < len(fit_labels)
                    else f"Segment {idx + 1}"
                )
                lines.append(f"  {segment_name}")
                base_val = baseline_slopes[idx] if idx < len(baseline_slopes) else None
                for label, _, _, _, _, run_slopes in comparison_entries:
                    run_val = (
                        run_slopes[idx]
                        if isinstance(run_slopes, tuple) and idx < len(run_slopes)
                        else None
                    )
                    lines.append(
                        f"    {label.upper()}  \u2022  {fmt(percent_error(run_val, base_val))}"
                    )
        else:
            for label, _, _, _, _, run_slopes in comparison_entries:
                lines.append(
                    f"  {label.upper()}  \u2022  {fmt(percent_error(run_slopes, baseline_slopes))}"
                )

        return "\n".join(lines)

    def _display_gradient_error(self, ax, text, anchor):
        """Render slope-error callout below the equation boxes."""
        if anchor is None:
            return

        x_anchor, halign, valign, boxes = anchor
        offset = 0.06
        ypos = min(boxes) - offset if valign == "top" else max(boxes) + offset

        ax.text(
            x_anchor, ypos, text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment=valign,
            horizontalalignment=halign,
            bbox=dict(
                boxstyle="round,pad=0.26",
                facecolor="#F7F7F7", alpha=0.9,
                edgecolor="#6E6E6E", linewidth=1.2,
            ),
            color="#3F3F3F", fontweight="bold", family="Montserrat",
        )

    # ------------------------------------------------------------------
    # Scatter generator
    # ------------------------------------------------------------------

    def generate_scatter_plots(self):
        """Generate all configured scatter plots and optional fit overlays."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(1)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="Scatter", unit="plot", leave=True)
        for plot_def in plot_iter:
            show_equations = True
            show_error = True
            gate_spec = None
            color_gate = None
            annotate_fit_at = None

            if len(plot_def) == 4:
                plot_name, (x_var, y_var), axis_limits, best_fit = plot_def
            elif len(plot_def) == 5:
                plot_name, (x_var, y_var), axis_limits, best_fit, item5 = plot_def
                if isinstance(item5, bool):
                    show_equations = item5
                elif datafunctions.is_gate_spec(item5):
                    gate_spec = item5
                else:
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 5th item must be gate_spec or boolean."
                    )
            elif len(plot_def) == 6:
                plot_name, (x_var, y_var), axis_limits, best_fit, item5, item6 = plot_def
                if isinstance(item6, bool):
                    if isinstance(item5, bool):
                        show_equations, show_error = item5, item6
                    else:
                        gate_spec = item5
                        show_equations = item6
                        if not datafunctions.is_gate_spec(gate_spec):
                            raise ValueError(
                                f"Scatter plot '{plot_name}': 5th item must be gate_spec when 6th is boolean."
                            )
                elif isinstance(item5, bool):
                    show_equations = item5
                    gate_spec = item6
                    if not datafunctions.is_gate_spec(gate_spec):
                        raise ValueError(
                            f"Scatter plot '{plot_name}': 6th item must be gate_spec when 5th is boolean."
                        )
                else:
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 6-item format requires gate_spec + boolean, or two booleans."
                    )
            elif len(plot_def) >= 7:
                plot_name, (x_var, y_var), axis_limits, best_fit, gate_spec, show_equations, show_error = plot_def[:7]
                color_gate = plot_def[7] if len(plot_def) > 7 else None
                annotate_fit_at = plot_def[8] if len(plot_def) > 8 else None
                if gate_spec is not None and not datafunctions.is_gate_spec(gate_spec):
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 5th item (gate_spec) must be a valid gate specification or None."
                    )
                if not isinstance(show_equations, bool) or not isinstance(show_error, bool):
                    raise ValueError(
                        f"Scatter plot '{plot_name}': 6th and 7th items must be booleans."
                    )
            else:
                raise ValueError(
                    f"Scatter plot '{plot_def[0] if plot_def else 'unknown'}' must have 4–7 items."
                )

            if best_fit is None:
                if self.verbose:
                    print(
                        f"[WARNING][DataPlotter] Scatter '{plot_name}': best_fit=None → 0 (no fit)."
                    )
                best_fit = 0

            if self.verbose:
                print(f"Creating scatter plot: {plot_name} ({x_var} vs {y_var})")

            filename = self._sanitize_plot_filename("scatter", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.scatter_FIGSIZE)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(
                datafunctions.add_units_to_label(x_var, self.units_map),
                fontweight="bold", fontsize=14,
            )
            ax.set_ylabel(
                datafunctions.add_units_to_label(y_var, self.units_map),
                fontweight="bold", fontsize=14,
            )

            eq_list = []
            fit_line_params = {}  # run_label -> (slopes, intercepts) for annotate_fit_at

            for run in self.runs:
                rn = run["name"].lower()
                if rn not in self.run_data:
                    continue

                df = self._get_filtered_run_dataframe(rn, gate_spec)
                if df is None:
                    continue

                if x_var not in df.columns or y_var not in df.columns:
                    if self.verbose:
                        print(
                            f"[WARNING][DataPlotter] Scatter '{plot_name}': "
                            f"missing '{x_var}' or '{y_var}' in run '{rn}'. Skipping."
                        )
                    continue

                xy_index, x_values, y_values = self._prepare_scatter_xy(df, x_var, y_var)
                if x_values is None:
                    if self.verbose:
                        print(
                            f"[WARNING][DataPlotter] Scatter '{plot_name}': "
                            f"no valid points in run '{rn}'. Skipping."
                        )
                    continue

                point_size, point_alpha = self._resolve_scatter_plot_style(len(x_values))
                max_points = self.SCATTER_MAX_POINTS

                # ── color_gate: split points into gated vs normal ──────────
                if color_gate is not None and len(color_gate) >= 4:
                    cg_spec = color_gate[:3]
                    cg_color = color_gate[3]
                    df_cg = datafunctions.apply_gate_to_dataframe(df, cg_spec)
                    cg_idx = set(df_cg.index) if df_cg is not None and not df_cg.empty else set()
                    in_cg = np.isin(xy_index, list(cg_idx))

                    x_normal, y_normal = x_values[~in_cg], y_values[~in_cg]
                    x_cg,     y_cg     = x_values[in_cg],  y_values[in_cg]

                    if len(x_normal):
                        datafunctions.plot_scatter(
                            ax, x_normal, y_normal,
                            run["name"].upper(), run["color"],
                            point_alpha, point_size, x_var, y_var,
                            max_points=max_points,
                        )
                    if len(x_cg):
                        datafunctions.plot_scatter(
                            ax, x_cg, y_cg,
                            "_nolegend_", cg_color,
                            min(point_alpha + 0.1, 1.0), point_size * 1.3, x_var, y_var,
                            max_points=max_points,
                        )
                    # Fit uses all points regardless of gate colouring
                    x_fit, y_fit = x_values, y_values
                else:
                    x_fit, y_fit = x_values, y_values

                if isinstance(best_fit, (list, tuple)) and best_fit and isinstance(best_fit[0], (list, tuple)):
                    fit_condition_data = datafunctions.build_fit_condition_data(
                        df, xy_index, best_fit, plot_name=plot_name, run_name=rn,
                    )
                    ok, slopes, intercepts, eq_text, _ = datafunctions.plot_scatter_with_multi_fit(
                        ax, x_fit, y_fit,
                        run["name"].upper(), run["color"],
                        point_alpha, point_size, x_var, y_var,
                        fit_defs=best_fit, fit_condition_data=fit_condition_data,
                        max_points=max_points,
                    ) if color_gate is None else datafunctions.plot_scatter_with_multi_fit(
                        ax, x_fit, y_fit,
                        "_nolegend_", run["color"],
                        0, 0, x_var, y_var,
                        fit_defs=best_fit, fit_condition_data=fit_condition_data,
                        max_points=max_points,
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slopes))
                        fit_line_params[run["name"].upper()] = (slopes, intercepts)

                elif best_fit == 0:
                    if color_gate is None:
                        datafunctions.plot_scatter(
                            ax, x_fit, y_fit,
                            run["name"].upper(), run["color"],
                            point_alpha, point_size, x_var, y_var,
                            max_points=max_points,
                        )

                elif best_fit in (1, 2):
                    if best_fit == 2 and self.verbose:
                        print(
                            f"[WARNING][DataPlotter] Scatter '{plot_name}': "
                            "best_fit=2 (removed behavior) — falling back to single fit."
                        )
                    ok, slope, interc, eq_text, _ = datafunctions.plot_scatter_with_1fit(
                        ax, x_fit, y_fit,
                        run["name"].upper() if color_gate is None else "_nolegend_",
                        run["color"],
                        point_alpha, point_size, x_var, y_var,
                        max_points=max_points,
                    )
                    if ok:
                        eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slope))
                        fit_line_params[run["name"].upper()] = (slope, interc)

            # Axis limits
            has_x_limits = has_y_limits = False
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

            xl, xr = ax.get_xlim()
            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)
            if xl <= 0 <= xr:
                ax.axvline(0, color="#5E5E5E", linewidth=1, alpha=0.8)

            ax.grid(True, alpha=0.35, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Trend equations + error box
            anchor = None
            if eq_list:
                if show_equations:
                    anchor = self._display_equations(ax, eq_list)

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
                        if anchor is None:
                            x_anchor, y_anchor, halign, valign = self._select_trendline_anchor(
                                ax, eq_list
                            )
                            anchor = (x_anchor, halign, valign, [y_anchor])
                        self._display_gradient_error(ax, txt, anchor)

            legend = self._add_standard_legend(ax, loc="best")

            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    self._display_gate_info(ax, gate_text, legend=legend, trend_anchor=anchor)

            # ── color_gate legend patch ───────────────────────────────────
            if color_gate is not None and len(color_gate) >= 4:
                import matplotlib.patches as mpatches
                cg_label = datafunctions.format_gate_text(color_gate[:3]) or str(color_gate[:3])
                patch = mpatches.Patch(color=color_gate[3], label=f"Gate: {cg_label}")
                existing = legend.legend_handles if legend else []
                existing_labels = [t.get_text() for t in legend.get_texts()] if legend else []
                ax.legend(
                    handles=list(existing) + [patch],
                    labels=existing_labels + [f"Gate: {cg_label}"],
                    framealpha=0.9, fontsize=10,
                )

            # ── annotate_fit_at ───────────────────────────────────────────
            if annotate_fit_at is not None and fit_line_params:
                x_at = float(annotate_fit_at)
                xl, xr = ax.get_xlim()
                if xl <= x_at <= xr:
                    ax.axvline(x_at, color="#5E5E5E", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
                    for entry in eq_list:
                        label_name, _, color_e = entry[0], entry[1], entry[2]
                        if label_name not in fit_line_params:
                            continue
                        slopes_p, intercepts_p = fit_line_params[label_name]
                        if isinstance(slopes_p, tuple):
                            continue  # multi-fit: not well-defined at a single x
                        if slopes_p is None or intercepts_p is None:
                            continue
                        y_at = slopes_p * x_at + intercepts_p
                        ax.scatter([x_at], [y_at], color=color_e, s=50, zorder=10,
                                   edgecolors="white", linewidths=1.2)
                        ax.annotate(
                            f"{y_at:.3g}",
                            xy=(x_at, y_at), xytext=(8, 0),
                            textcoords="offset points",
                            fontsize=9, fontweight="bold", color=color_e,
                            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                      alpha=0.85, edgecolor=color_e, linewidth=0.8),
                        )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.15, facecolor="white")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")
