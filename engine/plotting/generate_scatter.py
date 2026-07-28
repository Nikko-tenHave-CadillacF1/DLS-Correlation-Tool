"""Module-level generator functions extracted from ``ScatterMixin``.

Extracted 2026-07 (Prompt 12 Phase 2). Function bodies are the class's
original method bodies with a mechanical ``self`` -> ``plotter`` rename;
the old file at ``engine/plot_generators_*.py`` keeps a thin ``class ..Mixin``
shim whose methods delegate here so ``DataPlotter``'s multiple inheritance
keeps working unchanged.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import datafunctions
from ..datafunctions import _tqdm
from ..logger import log


def _resolve_scatter_style(point_count, base_size, base_alpha):
    if point_count <= 5000:
        return base_size, base_alpha
    if point_count <= 20000:
        return max(3.5, base_size * 0.9), min(0.75, base_alpha + 0.05)
    if point_count <= 60000:
        return max(3.0, base_size * 0.8), max(0.35, base_alpha * 0.8)
    return max(2.5, base_size * 0.7), max(0.22, base_alpha * 0.65)


def _prepare_scatter_xy(plotter, df, x_var, y_var):
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


def _resolve_scatter_plot_style(plotter, point_count):
    return _resolve_scatter_style(point_count, plotter.SCATTER_DOT_SIZE, plotter.SCATTER_TRANSPARENCY)


def _build_gradient_segment_labels(plotter, fit_defs, x_var=None, y_var=None, data_bounds=None):
    if not isinstance(fit_defs, (list, tuple)):
        return None
    labels = []
    for idx, fit_def in enumerate(fit_defs, start=1):
        if not isinstance(fit_def, (list, tuple)) or len(fit_def) != 3:
            labels.append(f"Segment {idx}")
            continue
        axis, min_val, max_val = fit_def
        axis_name = x_var if axis == "x" else y_var if axis == "y" else str(axis)
        if data_bounds and axis_name in data_bounds:
            lo_bound, hi_bound = data_bounds[axis_name]
            if min_val is None:
                min_val = lo_bound
            if max_val is None:
                max_val = hi_bound
        if min_val is None and max_val is None:
            labels.append(f"{axis_name} $\\in$ ($-\\infty$, $+\\infty$)")
        elif min_val is None:
            labels.append(f"{axis_name} $\\in$ ($-\\infty$, {max_val:.4g}]")
        elif max_val is None:
            labels.append(f"{axis_name} $\\in$ [{min_val:.4g}, $+\\infty$)")
        else:
            labels.append(f"{axis_name} $\\in$ [{min_val:.4g}, {max_val:.4g}]")
    return labels if labels else None


def _select_trendline_anchor(plotter, ax, equations_list, avoid_corner=None, n_text_lines=0):
    w_frac = min(0.35, 0.22 + max(0, n_text_lines - 2) * 0.015)
    h_frac = 0.28 * min(1.5, 1.0 + max(0, n_text_lines - 4) * 0.06)
    xs = np.concatenate([np.asarray(xv) for _, _, _, xv, _, _ in equations_list]) if equations_list else np.array([])
    ys = np.concatenate([np.asarray(yv) for _, _, _, _, yv, _ in equations_list]) if equations_list else np.array([])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    def _density(corner):
        if xs.size == 0:
            return 0
        return plotter._count_points_in_region(xs, ys, x0, x1, y0, y1, corner[0], corner[1], w_frac, h_frac)

    corners = [
        c
        for c in plotter._INFO_CORNER_XY
        if c != avoid_corner and c[0] in ("left", "right") and c[1] in ("top", "bottom")
    ]
    corners.sort(key=_density)
    halign, valign = corners[0]
    x_anchor, y_anchor = plotter._INFO_CORNER_XY[(halign, valign)]
    return x_anchor, y_anchor, halign, valign


COMPACT_SEGMENT_THRESHOLD = 3


def _fmt_coeff(v):
    if v is None:
        return "n/a"
    if v == 0:
        return "0"
    raw = f"{v:.4g}"
    if "e" in raw or "E" in raw:
        abs_v = abs(v)
        if 1 <= abs_v < 1_000_000:
            sig = 4
            decimals = max(0, sig - len(str(int(abs_v))))
            formatted = f"{v:,.{decimals}f}"
            if decimals > 0:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
    return raw


def _parse_eq_list_to_segments(plotter, eq_list, fit_labels=None):
    if not eq_list:
        return []
    run_payload = []
    n_segments = 1
    for run_label, eq_text, color, x_vals, y_vals, slopes in eq_list:
        lines = [l.strip() for l in str(eq_text).splitlines() if l.strip()] if eq_text else []
        n_segments = max(n_segments, len(lines) or 1)
        run_payload.append((run_label, color, lines, slopes))
    segments = []
    for seg_idx in range(n_segments):
        if fit_labels and seg_idx < len(fit_labels):
            condition = fit_labels[seg_idx]
        else:
            condition = ""
            for _run_label, _color, lines, _slopes in run_payload:
                if seg_idx < len(lines):
                    line = lines[seg_idx]
                    if "   y = " in line:
                        condition = line.split("   y = ")[0].strip()
                    break
        runs_in_seg = []
        for run_label, color, lines, slopes in run_payload:
            if seg_idx >= len(lines):
                continue
            line = lines[seg_idx]
            eq_part = "y = " + line.split("   y = ")[1].strip() if "   y = " in line else line
            slope_val = slopes[seg_idx] if isinstance(slopes, tuple) and seg_idx < len(slopes) else slopes
            runs_in_seg.append((run_label, color, eq_part, slope_val))
        segments.append({"condition": condition, "runs": runs_in_seg})
    return segments


def _compute_segment_pct_errors(plotter, runs_in_seg, baseline_label):
    baseline_slope = next(
        (s for lbl, _, _, s in runs_in_seg if lbl.upper() == baseline_label.upper()),
        None,
    )
    errors = {}
    for lbl, _, _, slope in runs_in_seg:
        if lbl.upper() == baseline_label.upper():
            continue
        if slope is None or baseline_slope is None or slope == 0:
            errors[lbl] = None
        else:
            errors[lbl] = ((baseline_slope - slope) / slope) * 100
    return errors


def _format_pct_error(pct, as_factor=False):
    if pct is None:
        return "n/a"
    if as_factor:
        return f"x {1.0 + pct / 100.0:.3f}"
    return f"{pct:+.1f}%"


def _display_fit_info(
    plotter, ax, eq_list, show_equations, show_error, fit_labels=None, avoid_corner=None, error_as_factor=False
):
    segments = plotter._parse_eq_list_to_segments(eq_list, fit_labels)
    if not segments:
        return None
    baseline_label = plotter.runs[0]["name"].upper() if plotter.runs else (eq_list[0][0] if eq_list else "")
    n_text_lines = 0
    for seg in segments:
        has_cond = bool(seg["condition"])
        n_r = len(seg["runs"])
        n_c = sum(1 for lbl, _, _, _ in seg["runs"] if lbl.upper() != baseline_label.upper())
        n_text_lines += (1 if has_cond else 0) + (n_r if show_equations else n_c)
    x_anchor, y_anchor, halign, valign = plotter._select_trendline_anchor(
        ax, eq_list, avoid_corner=avoid_corner, n_text_lines=n_text_lines
    )
    if len(segments) > plotter.COMPACT_SEGMENT_THRESHOLD:
        return plotter._display_compact_fit_box(
            ax,
            segments,
            show_equations,
            show_error,
            baseline_label,
            x_anchor,
            y_anchor,
            halign,
            valign,
            error_as_factor=error_as_factor,
        )
    return plotter._display_segment_boxes(
        ax,
        segments,
        show_equations,
        show_error,
        baseline_label,
        x_anchor,
        y_anchor,
        halign,
        valign,
        error_as_factor=error_as_factor,
    )


def _display_segment_boxes(
    plotter,
    ax,
    segments,
    show_equations,
    show_error,
    baseline_label,
    x_anchor,
    y_anchor,
    halign,
    valign,
    error_as_factor=False,
):
    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

    fig = ax.get_figure()
    fig_h = fig.get_size_inches()[1]
    total_lines = 0
    for seg in segments:
        has_cond = bool(seg["condition"])
        n_r = len(seg["runs"])
        n_c = sum(1 for lbl, _, _, _ in seg["runs"] if lbl.upper() != baseline_label.upper())
        total_lines += (1 if has_cond else 0) + (n_r if show_equations else n_c)
    fontsize = 11 if total_lines <= 6 else (10 if total_lines <= 12 else 9)
    _ha_map = {"left": 0.0, "center": 0.5, "right": 1.0}
    _va_map = {"top": 1.0, "center": 0.5, "bottom": 0.0}
    ha_val = _ha_map.get(halign, 0.0)
    va_val = _va_map.get(valign, 1.0)
    ab_pad = 0
    sep_pts = 3
    vpk_pad = 8
    box_gap_frac = 0.015
    cursor = y_anchor
    all_ypos = []
    annotation_boxes = []
    for seg in segments:
        condition = seg["condition"]
        runs_in_seg = seg["runs"]
        pct_errors = {}
        if show_error and len(runs_in_seg) > 1:
            pct_errors = plotter._compute_segment_pct_errors(runs_in_seg, baseline_label)
        line_items = []
        if condition:
            line_items.append((f"{condition}:", "#2A2A2A"))
        for run_label, run_color, eq_part, _ in runs_in_seg:
            is_baseline = run_label.upper() == baseline_label.upper()
            if show_equations:
                text = eq_part
                if show_error and not is_baseline and run_label in pct_errors:
                    pct_str = plotter._format_pct_error(pct_errors[run_label], error_as_factor)
                    text += f" $\\rightarrow$ $\\delta$ = {pct_str}"
                line_items.append((text, run_color))
            elif show_error and not is_baseline:
                pct_str = plotter._format_pct_error(pct_errors.get(run_label), error_as_factor)
                line_items.append((f"$\\delta$ = {pct_str}", run_color))
        if not line_items:
            continue
        text_areas = [
            TextArea(
                text,
                textprops=dict(
                    color=color,
                    fontsize=fontsize,
                    fontweight="bold",
                    family="Montserrat",
                ),
            )
            for text, color in line_items
        ]
        vpacker = VPacker(children=text_areas, pad=vpk_pad, sep=sep_pts)
        ab = AnnotationBbox(
            vpacker,
            xy=(x_anchor, cursor),
            xycoords="axes fraction",
            box_alignment=(ha_val, va_val),
            bboxprops=dict(
                boxstyle="round,pad=0",
                facecolor="white",
                alpha=0.92,
                edgecolor="#3C3C3C",
                linewidth=1.4,
            ),
            frameon=True,
            pad=ab_pad,
        )
        ab.set_zorder(10)
        ax.add_artist(ab)
        all_ypos.append(cursor)
        annotation_boxes.append(ab)
        n = len(line_items)
        box_h_pts = n * fontsize * 1.3 + (n - 1) * sep_pts + 2 * vpk_pad
        box_h_frac = box_h_pts / (fig_h * 72)
        if valign == "top":
            cursor -= box_h_frac + box_gap_frac
        else:
            cursor += box_h_frac + box_gap_frac
    return (x_anchor, halign, valign, all_ypos) if all_ypos else None


def _display_compact_fit_box(
    plotter,
    ax,
    segments,
    show_equations,
    show_error,
    baseline_label,
    x_anchor,
    y_anchor,
    halign,
    valign,
    error_as_factor=False,
):
    fontsize = 9
    lines = []
    for seg in segments:
        condition = seg["condition"]
        runs_in_seg = seg["runs"]
        pct_errors = {}
        if show_error and len(runs_in_seg) > 1:
            pct_errors = plotter._compute_segment_pct_errors(runs_in_seg, baseline_label)
        parts = [f"{condition}:"] if condition else []
        for run_label, _, eq_part, slope in runs_in_seg:
            is_baseline = run_label.upper() == baseline_label.upper()
            if show_equations:
                part = f"{run_label} m={plotter._fmt_coeff(slope)}"
                if show_error and not is_baseline and run_label in pct_errors:
                    pct_str = plotter._format_pct_error(pct_errors[run_label], error_as_factor)
                    part += f" ({pct_str})"
                parts.append(part)
            elif show_error and not is_baseline:
                pct_str = plotter._format_pct_error(pct_errors.get(run_label), error_as_factor)
                parts.append(f"{run_label} {pct_str}")
        min_parts = 2 if condition else 1
        if len(parts) >= min_parts:
            lines.append("  ".join(parts))
    if not lines:
        return None
    if not show_equations and show_error:
        lines = [f"$\\delta$m vs {baseline_label}"] + [f"  {l}" for l in lines]
    ax.text(
        x_anchor,
        y_anchor,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=valign,
        horizontalalignment=halign,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.92,
            edgecolor="#3C3C3C",
            linewidth=1.4,
        ),
        color="#1A1A1A",
        fontweight="bold",
        family="Montserrat",
        zorder=10,
    )
    return x_anchor, halign, valign, [y_anchor]


def generate_scatter_plots(plotter):
    plotter._ensure_preprocessed()
    plots = plotter._get_plot_group(1)
    if not plots:
        return
    plot_iter = plots if plotter.verbose else _tqdm(plots, desc="Scatter", unit="plot", leave=True)
    for plot_def in plot_iter:
        plot_name = plot_def.name
        x_var = plot_def.x_channel
        y_var = plot_def.y_channel
        axis_limits = plot_def.axis_limits
        best_fit = plot_def.best_fit if plot_def.best_fit is not None else 0
        gate_spec = plot_def.gate
        show_equations = plot_def.show_equations
        show_error = plot_def.show_error
        error_as_factor = getattr(plot_def, "error_as_factor", False)
        color_gate = plot_def.color_gate
        annotate_fit_at = plot_def.annotate_fit_at
        markers = plot_def.markers
        robust = plot_def.robust
        robust_threshold = plot_def.robust_threshold
        reference_lines = plot_def.reference_lines
        if plotter.verbose:
            log.debug("Creating scatter plot: %s (%s vs %s)", plot_name, x_var, y_var)
        filename = plotter._sanitize_plot_filename("scatter", plot_name)
        figsize = plotter._resolve_plot_figsize(filename, plotter.scatter_FIGSIZE)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(
            datafunctions.add_units_to_label(x_var, plotter.units_map),
            fontweight="bold",
            fontsize=14,
        )
        ax.set_ylabel(
            datafunctions.add_units_to_label(y_var, plotter.units_map),
            fontweight="bold",
            fontsize=14,
        )
        eq_list = []
        fit_line_params = {}
        condition_data_bounds = {}
        for run in plotter.runs:
            rn = run["name"].lower()
            if rn not in plotter.run_data:
                continue
            df = plotter._get_filtered_run_dataframe(rn, gate_spec)
            if df is None:
                continue
            if x_var not in df.columns or y_var not in df.columns:
                missing = [ch for ch in (x_var, y_var) if ch not in df.columns]
                msg = f"Scatter '{plot_name}': missing {missing} in run '{rn}'. Skipping."
                for ch in missing:
                    hint = plotter._format_missing_channel_hint(rn, ch)
                    if hint:
                        msg += f"\n{hint}"
                log.warning("%s", msg)
                continue
            xy_index, x_values, y_values = plotter._prepare_scatter_xy(df, x_var, y_var)
            if x_values is None:
                log.warning(
                    "Scatter '%s': no valid points in run '%s'. Skipping.",
                    plot_name,
                    rn,
                )
                continue
            point_size, point_alpha = plotter._resolve_scatter_plot_style(len(x_values))
            max_points = plotter.SCATTER_MAX_POINTS
            if color_gate is not None and len(color_gate) >= 4:
                cg_spec = color_gate[:3]
                cg_color = color_gate[3]
                df_cg = datafunctions.apply_gate_to_dataframe(df, cg_spec)
                cg_idx = set(df_cg.index) if df_cg is not None and not df_cg.empty else set()
                in_cg = np.isin(xy_index, list(cg_idx))
                x_normal, y_normal = x_values[~in_cg], y_values[~in_cg]
                x_cg, y_cg = x_values[in_cg], y_values[in_cg]
                if len(x_normal):
                    datafunctions.plot_scatter(
                        ax,
                        x_normal,
                        y_normal,
                        run["name"].upper(),
                        run["color"],
                        point_alpha,
                        point_size,
                        x_var,
                        y_var,
                        max_points=max_points,
                    )
                if len(x_cg):
                    datafunctions.plot_scatter(
                        ax,
                        x_cg,
                        y_cg,
                        "_nolegend_",
                        cg_color,
                        min(point_alpha + 0.1, 1.0),
                        point_size * 1.3,
                        x_var,
                        y_var,
                        max_points=max_points,
                    )
                x_fit, y_fit = x_values, y_values
            else:
                x_fit, y_fit = x_values, y_values
            if isinstance(best_fit, (list, tuple)) and best_fit and isinstance(best_fit[0], (list, tuple)):
                fit_condition_data = datafunctions.build_fit_condition_data(
                    df,
                    xy_index,
                    best_fit,
                    plot_name=plot_name,
                    run_name=rn,
                )
                if fit_condition_data:
                    for ch_name, ch_arr in fit_condition_data.items():
                        finite_vals = ch_arr[np.isfinite(ch_arr)] if hasattr(ch_arr, "__len__") else np.array([])
                        if len(finite_vals) > 0:
                            ch_min, ch_max = float(np.min(finite_vals)), float(np.max(finite_vals))
                            if ch_name in condition_data_bounds:
                                prev_min, prev_max = condition_data_bounds[ch_name]
                                condition_data_bounds[ch_name] = (min(prev_min, ch_min), max(prev_max, ch_max))
                            else:
                                condition_data_bounds[ch_name] = (ch_min, ch_max)
                ok, slopes, intercepts, eq_text, fit_meta = (
                    datafunctions.plot_scatter_with_multi_fit(
                        ax,
                        x_fit,
                        y_fit,
                        run["name"].upper(),
                        run["color"],
                        point_alpha,
                        point_size,
                        x_var,
                        y_var,
                        fit_defs=best_fit,
                        fit_condition_data=fit_condition_data,
                        max_points=max_points,
                        robust=robust,
                        robust_threshold=robust_threshold,
                    )
                    if color_gate is None
                    else datafunctions.plot_scatter_with_multi_fit(
                        ax,
                        x_fit,
                        y_fit,
                        "_nolegend_",
                        run["color"],
                        0,
                        0,
                        x_var,
                        y_var,
                        fit_defs=best_fit,
                        fit_condition_data=fit_condition_data,
                        max_points=max_points,
                        robust=robust,
                        robust_threshold=robust_threshold,
                    )
                )
                if ok:
                    eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slopes))
                    fit_line_params[run["name"].upper()] = (slopes, intercepts)
                    if robust and isinstance(fit_meta, dict) and fit_meta.get("robust_info"):
                        info = fit_meta["robust_info"]
                        if info["n_outliers"] > 0:
                            plotter._outlier_log.append(
                                {
                                    "plot": plot_name,
                                    "run": run["name"].upper(),
                                    "n_outliers": info["n_outliers"],
                                    "n_total": info["n_total"],
                                    "pseudo_r2": None,
                                }
                            )
            elif best_fit == 0:
                if color_gate is None:
                    datafunctions.plot_scatter(
                        ax,
                        x_fit,
                        y_fit,
                        run["name"].upper(),
                        run["color"],
                        point_alpha,
                        point_size,
                        x_var,
                        y_var,
                        max_points=max_points,
                    )
            elif best_fit in (1, 2):
                ok, slope, interc, eq_text, fit_meta = datafunctions.plot_scatter_with_1fit(
                    ax,
                    x_fit,
                    y_fit,
                    run["name"].upper() if color_gate is None else "_nolegend_",
                    run["color"],
                    point_alpha,
                    point_size,
                    x_var,
                    y_var,
                    max_points=max_points,
                    robust=robust,
                    robust_threshold=robust_threshold,
                )
                if ok:
                    eq_list.append((run["name"].upper(), eq_text, run["color"], x_values, y_values, slope))
                    fit_line_params[run["name"].upper()] = (slope, interc)
                    if robust and isinstance(fit_meta, dict) and fit_meta.get("robust_info"):
                        info = fit_meta["robust_info"]
                        if info["n_outliers"] > 0:
                            plotter._outlier_log.append(
                                {
                                    "plot": plot_name,
                                    "run": run["name"].upper(),
                                    "n_outliers": info["n_outliers"],
                                    "n_total": info["n_total"],
                                    "pseudo_r2": info["pseudo_r2"],
                                }
                            )
        has_x_limits, has_y_limits = plotter._apply_2d_axis_limits(ax, axis_limits)
        plotter._add_axis_edge_padding(
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
        plotter._apply_grid(ax, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        anchor = None
        if eq_list and (show_equations or show_error):
            fit_labels = None
            if isinstance(best_fit, (list, tuple)) and best_fit and isinstance(best_fit[0], (list, tuple)):
                data_bounds = dict(condition_data_bounds)
                all_x = np.concatenate([e[3] for e in eq_list if e[3] is not None])
                all_y = np.concatenate([e[4] for e in eq_list if e[4] is not None])
                if len(all_x) > 0:
                    data_bounds[x_var] = (float(np.nanmin(all_x)), float(np.nanmax(all_x)))
                if len(all_y) > 0:
                    data_bounds[y_var] = (float(np.nanmin(all_y)), float(np.nanmax(all_y)))
                fit_labels = plotter._build_gradient_segment_labels(
                    best_fit, x_var=x_var, y_var=y_var, data_bounds=data_bounds
                )
            anchor = plotter._display_fit_info(
                ax,
                eq_list,
                show_equations,
                show_error,
                fit_labels=fit_labels,
                error_as_factor=error_as_factor,
            )
        fit_corner = (anchor[1], anchor[2]) if anchor else None
        legend = plotter._add_standard_legend(ax, avoid_corner=fit_corner)
        if gate_spec is not None:
            gate_text = datafunctions.format_gate_text(gate_spec)
            if gate_text:
                plotter._display_gate_info(ax, gate_text, legend=legend, trend_anchor=anchor)
        if color_gate is not None and len(color_gate) >= 4 and legend is not None:
            import matplotlib.patches as mpatches

            cg_label = datafunctions.format_gate_text(color_gate[:3]) or str(color_gate[:3])
            patch = mpatches.Patch(color=color_gate[3], label=f"Gate: {cg_label}")
            existing = list(legend.legend_handles)
            existing_labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            legend = plotter._add_standard_legend(
                ax,
                handles=existing + [patch],
                labels=existing_labels + [f"Gate: {cg_label}"],
                avoid_corner=fit_corner,
            )
        if annotate_fit_at is not None and fit_line_params:
            if isinstance(annotate_fit_at, (list, tuple)):
                x_at_values = [float(v) for v in annotate_fit_at]
            else:
                x_at_values = [float(annotate_fit_at)]
            xl, xr = ax.get_xlim()
            for x_at in x_at_values:
                if not (xl <= x_at <= xr):
                    continue
                ax.axvline(x_at, color="#5E5E5E", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
                ann_items = []
                for entry in eq_list:
                    label_name, _, color_e = entry[0], entry[1], entry[2]
                    if label_name not in fit_line_params:
                        continue
                    slopes_p, intercepts_p = fit_line_params[label_name]
                    if isinstance(slopes_p, tuple):
                        found = False
                        for s, ic in zip(slopes_p, intercepts_p):
                            if s is not None and ic is not None:
                                slopes_p, intercepts_p = s, ic
                                found = True
                                break
                        if not found:
                            continue
                    if slopes_p is None or intercepts_p is None:
                        continue
                    y_at = slopes_p * x_at + intercepts_p
                    ann_items.append((y_at, color_e, label_name))
                if ann_items:
                    ann_items.sort(key=lambda t: t[0])
                    n_items = len(ann_items)
                    trans = ax.transData
                    display_ys = [trans.transform((x_at, item[0]))[1] for item in ann_items]
                    x_frac = (x_at - xl) / (xr - xl) if (xr - xl) > 0 else 0.5
                    # Decide which side of the vertical guide each label lives on.
                    # For 2+ items we alternate sides so multi-run comparisons at
                    # the same x sit side-by-side instead of stacking vertically.
                    # `place_left` (x near right edge) flips the starting side so
                    # the first label always lands inside the plot area.
                    sides = []
                    for i in range(n_items):
                        if n_items == 1:
                            sides.append("left" if x_frac > 0.80 else "right")
                        else:
                            first_side = "left" if x_frac > 0.80 else "right"
                            if i % 2 == 0:
                                sides.append(first_side)
                            else:
                                sides.append("right" if first_side == "left" else "left")
                    # Vertical separation is only needed within items on the same
                    # side — labels on opposite sides never clash.
                    min_sep = 18
                    adjusted_display_ys = list(display_ys)
                    for side in ("left", "right"):
                        side_indices = [i for i, s in enumerate(sides) if s == side]
                        for k in range(1, len(side_indices)):
                            i_prev = side_indices[k - 1]
                            i_curr = side_indices[k]
                            gap = adjusted_display_ys[i_curr] - adjusted_display_ys[i_prev]
                            if gap < min_sep:
                                adjusted_display_ys[i_curr] = adjusted_display_ys[i_prev] + min_sep
                    for i, (y_at, color_e, _) in enumerate(ann_items):
                        base_display_y = display_ys[i]
                        target_display_y = adjusted_display_ys[i]
                        nudge_pts = target_display_y - base_display_y
                        y_offset = 8 + nudge_pts
                        if sides[i] == "left":
                            x_text_offset = -10
                            ha = "right"
                        else:
                            x_text_offset = 10
                            ha = "left"
                        ax.scatter([x_at], [y_at], color=color_e, s=50, zorder=10, edgecolors="white", linewidths=1.2)
                        ax.annotate(
                            f"{y_at:.3g}",
                            xy=(x_at, y_at),
                            xytext=(x_text_offset, y_offset),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            color=color_e,
                            ha=ha,
                            zorder=11,
                            arrowprops=dict(arrowstyle="-", color=color_e, lw=0.8, alpha=0.6),
                            bbox=dict(
                                boxstyle="round,pad=0.22",
                                facecolor="white",
                                alpha=0.92,
                                edgecolor=color_e,
                                linewidth=0.8,
                            ),
                        )
        plotter._draw_static_markers(ax, markers)
        plotter._draw_horizontal_reference_lines(ax, reference_lines)
        plt.tight_layout(pad=0.25)
        fig.savefig(
            plotter.plots_dir / filename,
            dpi=plotter.output_dpi,
            pad_inches=0.15,
            facecolor="white",
            bbox_inches="tight",
        )
        plt.close(fig)
        if plotter.verbose:
            log.debug("Saved: %s", filename)


def generate_scatter3d_plots(plotter, plots=None):
    plotter._ensure_preprocessed()
    if plots is None:
        plots = list(getattr(plotter, "debug_scatter3d_plots", []) or [])
    if not plots:
        return
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        log.warning("Scatter3D: mpl_toolkits.mplot3d unavailable; skipping.")
        return
    can_show = True
    try:
        backend = plt.get_backend().lower()
        if backend in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            can_show = False
    except Exception:
        can_show = False
    for plot_def in plots:
        plot_name = plot_def.name
        x_var = plot_def.x_channel
        y_var = plot_def.y_channel
        z_var = plot_def.z_channel
        gate_spec = plot_def.gate
        axis_limits = plot_def.axis_limits
        filename = plotter._sanitize_plot_filename("scatter3d", plot_name)
        figsize = plotter._resolve_plot_figsize(filename, plotter.scatter_FIGSIZE)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(x_var, fontweight="bold", fontsize=11)
        ax.set_ylabel(y_var, fontweight="bold", fontsize=11)
        ax.set_zlabel(z_var, fontweight="bold", fontsize=11)
        ax.set_title(plot_name, fontweight="bold", fontsize=13)
        any_plotted = False
        for run in plotter.runs:
            rn = run["name"].lower()
            if rn not in plotter.run_data:
                continue
            df = plotter._get_filtered_run_dataframe(rn, gate_spec)
            if df is None:
                continue
            missing = [ch for ch in (x_var, y_var, z_var) if ch not in df.columns]
            if missing:
                log.warning(
                    "Scatter3D '%s': missing %s in run '%s'. Skipping run.",
                    plot_name,
                    missing,
                    rn,
                )
                continue
            xyz = pd.concat(
                [
                    pd.to_numeric(df[x_var], errors="coerce").rename(x_var),
                    pd.to_numeric(df[y_var], errors="coerce").rename(y_var),
                    pd.to_numeric(df[z_var], errors="coerce").rename(z_var),
                ],
                axis=1,
            ).dropna()
            if xyz.empty:
                log.warning(
                    "Scatter3D '%s': no valid points in run '%s'. Skipping run.",
                    plot_name,
                    rn,
                )
                continue
            max_points = plotter.SCATTER_MAX_POINTS
            if len(xyz) > max_points:
                xyz = xyz.sample(n=max_points, random_state=0).sort_index()
            point_size, point_alpha = _resolve_scatter_style(
                len(xyz),
                plotter.SCATTER_DOT_SIZE,
                plotter.SCATTER_TRANSPARENCY,
            )
            ax.scatter(
                xyz[x_var].to_numpy(dtype=float),
                xyz[y_var].to_numpy(dtype=float),
                xyz[z_var].to_numpy(dtype=float),
                c=run["color"],
                s=point_size,
                alpha=point_alpha,
                label=run["name"].upper(),
                depthshade=True,
            )
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            log.warning("Scatter3D '%s': no runs plotted. Skipping figure.", plot_name)
            continue
        if axis_limits is not None:
            lo_x, hi_x = axis_limits[0]
            lo_y, hi_y = axis_limits[1]
            lo_z, hi_z = axis_limits[2]
            if lo_x is not None or hi_x is not None:
                ax.set_xlim(lo_x, hi_x)
            if lo_y is not None or hi_y is not None:
                ax.set_ylim(lo_y, hi_y)
            if lo_z is not None or hi_z is not None:
                ax.set_zlim(lo_z, hi_z)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(plotter.plots_dir / filename, dpi=plotter.output_dpi, facecolor="white", bbox_inches="tight")
        if plotter.verbose:
            log.debug("Saved: %s", filename)
        if can_show:
            try:
                plt.show(block=True)
            except Exception as exc:
                log.warning("Scatter3D '%s': interactive show failed (%s).", plot_name, exc)
                plt.close(fig)
        else:
            plt.close(fig)
