"""Scatter plot generator mixin for DataPlotter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datafunctions

try:
    from tqdm import tqdm as _tqdm_raw
    def _tqdm(it, **kw): return _tqdm_raw(it, file=__import__('sys').stderr, dynamic_ncols=True, **kw)
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

    def _build_gradient_segment_labels(self, fit_defs, x_var=None, y_var=None, data_bounds=None):
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

            # Replace None bounds with actual data min/max when available
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

    def _select_trendline_anchor(self, ax, equations_list, avoid_corner=None,
                                   n_text_lines=0):
        """Place text in the least-crowded position (corners + mid-edges).

        avoid_corner: optional (halign, valign) tuple; that corner is excluded
        so the fit box never lands on top of the legend.
        n_text_lines: estimated number of text lines in the box — used to scale
        the exclusion zone height proportionally (more lines → taller box).
        """
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xs, ys = [], []
        for _, _, _, xv, yv, _ in equations_list:
            xs.extend(xv)
            ys.extend(yv)
        xs = np.array(xs)
        ys = np.array(ys)

        # Expanded candidate set: 4 corners + 2 top/bottom center + 2 mid-edges
        candidates = {
            "tl": (0.03, 0.97, "left",   "top"),
            "tr": (0.97, 0.97, "right",  "top"),
            "bl": (0.03, 0.07, "left",   "bottom"),
            "br": (0.97, 0.07, "right",  "bottom"),
            "tc": (0.50, 0.97, "center", "top"),
            "bc": (0.50, 0.07, "center", "bottom"),
            "ml": (0.03, 0.50, "left",   "center"),
            "mr": (0.97, 0.50, "right",  "center"),
        }

        # Scale exclusion zone based on content size
        base_w_frac = 0.22
        base_h_frac = 0.28
        if n_text_lines > 4:
            # Taller box for many lines — scale height up to 50% more
            h_scale = min(1.5, 1.0 + (n_text_lines - 4) * 0.06)
            base_h_frac *= h_scale
        # Wider zone for equations (long text)
        w_frac = min(0.35, base_w_frac + max(0, n_text_lines - 2) * 0.015)
        h_frac = base_h_frac

        def _count_pts(key):
            xa, ya, hal, val = candidates[key]
            w = (x1 - x0) * w_frac
            h = (y1 - y0) * h_frac
            x_abs = x0 + xa * (x1 - x0)
            if hal == "left":
                x_min = x_abs
            elif hal == "right":
                x_min = x_abs - w
            else:  # center
                x_min = x_abs - w / 2
            x_max = x_min + w
            if val == "top":
                y_min = y0 + ya * (y1 - y0) - h
                y_max = y0 + ya * (y1 - y0)
            elif val == "bottom":
                y_min = y0 + ya * (y1 - y0)
                y_max = y0 + ya * (y1 - y0) + h
            else:  # center (mid-edge)
                y_min = y0 + ya * (y1 - y0) - h / 2
                y_max = y0 + ya * (y1 - y0) + h / 2
            return ((xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)).sum()

        best = min(candidates.keys(), key=_count_pts)

        # If the best candidate clashes with the legend corner, pick the next
        # least-crowded one that doesn't clash.
        if avoid_corner is not None:
            avoid_hal, avoid_val = avoid_corner
            ordered = sorted(candidates.keys(), key=_count_pts)
            for key in ordered:
                _, _, hal, val = candidates[key]
                if not (hal == avoid_hal and val == avoid_val):
                    best = key
                    break

        return candidates[best]

    # ------------------------------------------------------------------
    # Fit info display — per-segment boxes (Proposal A)
    # ------------------------------------------------------------------

    # Segments above this threshold use a single compact box instead of
    # one box per segment, to avoid consuming too much of the plot area.
    COMPACT_SEGMENT_THRESHOLD = 3

    @staticmethod
    def _fmt_coeff(v):
        """Format a fit coefficient cleanly, preferring fixed-point for readable magnitudes."""
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

    def _parse_eq_list_to_segments(self, eq_list, fit_labels=None):
        """Restructure eq_list (per-run) into per-segment dicts.

        Each dict: {'condition': str, 'runs': [(label, color, eq_part, slope), ...]}
        The eq lines from datafunctions use '   y = ' (3 spaces) as the
        separator between the condition prefix and the equation.
        """
        if not eq_list:
            return []

        n_segments = max(
            len([l for l in str(entry[1]).splitlines() if l.strip()]) if entry[1] else 1
            for entry in eq_list
        )

        segments = []
        for seg_idx in range(n_segments):
            # Derive the condition label for this segment
            if fit_labels and seg_idx < len(fit_labels):
                condition = fit_labels[seg_idx]
            else:
                condition = ""
                for _, eq_text, _, _, _, _ in eq_list:
                    if not eq_text:
                        continue
                    lines = [l.strip() for l in str(eq_text).splitlines() if l.strip()]
                    if seg_idx < len(lines):
                        line = lines[seg_idx]
                        if "   y = " in line:
                            condition = line.split("   y = ")[0].strip()
                        break

            runs_in_seg = []
            for run_label, eq_text, color, x_vals, y_vals, slopes in eq_list:
                lines = (
                    [l.strip() for l in str(eq_text).splitlines() if l.strip()]
                    if eq_text else []
                )
                if not lines or seg_idx >= len(lines):
                    continue
                line = lines[seg_idx]
                eq_part = (
                    "y = " + line.split("   y = ")[1].strip()
                    if "   y = " in line
                    else line  # single-fit: line IS the equation
                )
                slope_val = (
                    slopes[seg_idx]
                    if isinstance(slopes, tuple) and seg_idx < len(slopes)
                    else slopes
                )
                runs_in_seg.append((run_label, color, eq_part, slope_val))

            segments.append({"condition": condition, "runs": runs_in_seg})

        return segments

    def _compute_segment_pct_errors(self, runs_in_seg, baseline_label):
        """Compute % correction needed on each run's slope to match baseline.

        Returns the percentage increase/decrease that must be applied to the
        run's gradient to obtain the baseline gradient:
            δm = (m_baseline - m_run) / m_run × 100%
        """
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

    def _display_fit_info(self, ax, eq_list, show_equations, show_error,
                           fit_labels=None, avoid_corner=None):
        """Unified renderer: per-segment boxes (≤ threshold segs) or compact box.

        Returns anchor tuple (x, halign, valign, [y_positions]) for downstream
        use by gate-info placement.
        """
        segments = self._parse_eq_list_to_segments(eq_list, fit_labels)
        if not segments:
            return None

        baseline_label = (
            self.runs[0]["name"].upper() if self.runs
            else (eq_list[0][0] if eq_list else "")
        )

        # Estimate total text lines for sizing the exclusion zone
        n_text_lines = 0
        for seg in segments:
            has_cond = bool(seg["condition"])
            n_r = len(seg["runs"])
            n_c = sum(1 for lbl, _, _, _ in seg["runs"] if lbl.upper() != baseline_label.upper())
            n_text_lines += (1 if has_cond else 0) + (n_r if show_equations else n_c)

        x_anchor, y_anchor, halign, valign = self._select_trendline_anchor(
            ax, eq_list, avoid_corner=avoid_corner, n_text_lines=n_text_lines
        )

        if len(segments) > self.COMPACT_SEGMENT_THRESHOLD:
            return self._display_compact_fit_box(
                ax, segments, show_equations, show_error, baseline_label,
                x_anchor, y_anchor, halign, valign,
            )
        return self._display_segment_boxes(
            ax, segments, show_equations, show_error, baseline_label,
            x_anchor, y_anchor, halign, valign,
        )

    def _display_segment_boxes(
        self, ax, segments, show_equations, show_error, baseline_label,
        x_anchor, y_anchor, halign, valign,
    ):
        """One rounded box per segment. Each run equation in its run color.

        Uses AnnotationBbox + VPacker + TextArea so matplotlib sizes the box
        exactly from the rendered text — no manual line-height estimates needed.
        pad in AnnotationBbox is in points (DPI-independent).

        After initial placement, a renderer-measured pass detects overlaps and
        repositions boxes precisely, eliminating line-height estimation drift.
        """
        from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

        fig = ax.get_figure()
        fig_h = fig.get_size_inches()[1]

        # Adaptive fontsize
        total_lines = 0
        for seg in segments:
            has_cond = bool(seg["condition"])
            n_r = len(seg["runs"])
            n_c = sum(1 for lbl, _, _, _ in seg["runs"] if lbl.upper() != baseline_label.upper())
            total_lines += (1 if has_cond else 0) + (n_r if show_equations else n_c)

        fontsize = 11 if total_lines <= 6 else (10 if total_lines <= 12 else 9)

        # box_alignment: (left/right=0/1, bottom/top=0/1) — maps corner to xy anchor
        _ha_map = {"left": 0.0, "center": 0.5, "right": 1.0}
        _va_map = {"top": 1.0, "center": 0.5, "bottom": 0.0}
        ha_val = _ha_map.get(halign, 0.0)
        va_val = _va_map.get(valign, 1.0)

        ab_pad   = 0    # padding handled by VPacker instead (points-based)
        sep_pts  = 3    # separation between TextArea children in points
        vpk_pad  = 8    # boundary padding inside VPacker, in points
        box_gap_frac = 0.015

        cursor = y_anchor
        all_ypos = []
        annotation_boxes = []

        for seg in segments:
            condition = seg["condition"]
            runs_in_seg = seg["runs"]

            pct_errors = {}
            if show_error and len(runs_in_seg) > 1:
                pct_errors = self._compute_segment_pct_errors(runs_in_seg, baseline_label)

            line_items = []
            if condition:
                line_items.append((f"{condition}:", "#2A2A2A"))

            for run_label, run_color, eq_part, _ in runs_in_seg:
                is_baseline = run_label.upper() == baseline_label.upper()
                if show_equations:
                    text = eq_part
                    if show_error and not is_baseline and run_label in pct_errors:
                        pct = pct_errors[run_label]
                        pct_str = "n/a" if pct is None else f"{pct:+.1f}%"
                        text += f" $\\rightarrow$ $\\delta$ = {pct_str}"
                    line_items.append((text, run_color))
                elif show_error and not is_baseline:
                    pct = pct_errors.get(run_label)
                    pct_str = "n/a" if pct is None else f"{pct:+.1f}%"
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

            # Initial estimate for cursor advancement (refined below with renderer)
            n = len(line_items)
            box_h_pts = n * fontsize * 1.3 + (n - 1) * sep_pts + 2 * vpk_pad
            box_h_frac = box_h_pts / (fig_h * 72)
            if valign == "top":
                cursor -= box_h_frac + box_gap_frac
            else:
                cursor += box_h_frac + box_gap_frac

        # ── Renderer-measured repositioning pass ──────────────────────────
        # Draw once to compute actual extents, then fix any overlap.
        if len(annotation_boxes) > 1:
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                ax_bbox = ax.get_window_extent(renderer)
                ax_height = ax_bbox.height  # display pixels

                # Get actual box extents in display coords
                extents = []
                for ab in annotation_boxes:
                    bb = ab.get_window_extent(renderer)
                    extents.append(bb)

                # Reposition boxes to eliminate overlap
                gap_px = box_gap_frac * ax_height
                new_positions = [y_anchor]

                for i in range(1, len(extents)):
                    prev_bb = extents[i - 1]
                    curr_bb = extents[i]
                    # Actual box heights in axes fraction
                    prev_h_frac = prev_bb.height / ax_height
                    curr_h_frac = curr_bb.height / ax_height
                    gap_frac = gap_px / ax_height

                    if valign == "top":
                        new_pos = new_positions[i - 1] - prev_h_frac - gap_frac
                    else:
                        new_pos = new_positions[i - 1] + prev_h_frac + gap_frac
                    new_positions.append(new_pos)

                # Update positions
                for ab, new_y in zip(annotation_boxes, new_positions):
                    ab.xy = (x_anchor, new_y)
                    ab.xybox = (x_anchor, new_y)

                all_ypos = new_positions
            except Exception:
                # Fallback: keep initial estimated positions
                pass

        return (x_anchor, halign, valign, all_ypos) if all_ypos else None

    def _display_compact_fit_box(
        self, ax, segments, show_equations, show_error, baseline_label,
        x_anchor, y_anchor, halign, valign,
    ):
        """Single compact box for plots with many segments (> COMPACT_SEGMENT_THRESHOLD).

        One line per segment:
          show_equations=True:   COND  RUN m=val  RUN m=val (Δ%)
          show_equations=False:  COND  RUN Δ%  RUN Δ%
        """
        fontsize = 9
        lines = []

        for seg in segments:
            condition = seg["condition"]
            runs_in_seg = seg["runs"]

            pct_errors = {}
            if show_error and len(runs_in_seg) > 1:
                pct_errors = self._compute_segment_pct_errors(runs_in_seg, baseline_label)

            parts = [f"{condition}:"] if condition else []
            for run_label, _, eq_part, slope in runs_in_seg:
                is_baseline = run_label.upper() == baseline_label.upper()
                if show_equations:
                    part = f"{run_label} m={self._fmt_coeff(slope)}"
                    if show_error and not is_baseline and run_label in pct_errors:
                        pct = pct_errors[run_label]
                        pct_str = "n/a" if pct is None else f"{pct:+.1f}%"
                        part += f" ({pct_str})"
                    parts.append(part)
                elif show_error and not is_baseline:
                    pct = pct_errors.get(run_label)
                    pct_str = "n/a" if pct is None else f"{pct:+.1f}%"
                    parts.append(f"{run_label} {pct_str}")

            min_parts = 2 if condition else 1
            if len(parts) >= min_parts:
                lines.append("  ".join(parts))

        if not lines:
            return None

        if not show_equations and show_error:
            lines = [f"$\\delta$m vs {baseline_label}"] + [f"  {l}" for l in lines]

        ax.text(
            x_anchor, y_anchor, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment=valign,
            horizontalalignment=halign,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white", alpha=0.92,
                edgecolor="#3C3C3C", linewidth=1.4,
            ),
            color="#1A1A1A", fontweight="bold", family="Montserrat", zorder=10,
        )
        return x_anchor, halign, valign, [y_anchor]

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
            condition_data_bounds = {}  # axis_name -> (min, max) for segment labels

            for run in self.runs:
                rn = run["name"].lower()
                if rn not in self.run_data:
                    continue

                df = self._get_filtered_run_dataframe(rn, gate_spec)
                if df is None:
                    continue

                if x_var not in df.columns or y_var not in df.columns:
                    missing = [ch for ch in (x_var, y_var) if ch not in df.columns]
                    msg = (
                        f"[WARNING][DataPlotter] Scatter '{plot_name}': "
                        f"missing {missing} in run '{rn}'. Skipping."
                    )
                    for ch in missing:
                        hint = self._format_missing_channel_hint(rn, ch)
                        if hint:
                            msg += f"\n{hint}"
                    print(msg)
                    continue

                xy_index, x_values, y_values = self._prepare_scatter_xy(df, x_var, y_var)
                if x_values is None:
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
                    # Accumulate condition channel bounds for segment labels
                    if fit_condition_data:
                        for ch_name, ch_arr in fit_condition_data.items():
                            finite_vals = ch_arr[np.isfinite(ch_arr)] if hasattr(ch_arr, '__len__') else np.array([])
                            if len(finite_vals) > 0:
                                ch_min, ch_max = float(np.min(finite_vals)), float(np.max(finite_vals))
                                if ch_name in condition_data_bounds:
                                    prev_min, prev_max = condition_data_bounds[ch_name]
                                    condition_data_bounds[ch_name] = (min(prev_min, ch_min), max(prev_max, ch_max))
                                else:
                                    condition_data_bounds[ch_name] = (ch_min, ch_max)
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

            self._apply_grid(ax, which="major")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Fit-info first so we know its corner, then legend avoids it
            anchor = None
            if eq_list and (show_equations or show_error):
                fit_labels = None
                if (
                    isinstance(best_fit, (list, tuple))
                    and best_fit
                    and isinstance(best_fit[0], (list, tuple))
                ):
                    # Build data_bounds for segment labels (replace inf with real range)
                    data_bounds = dict(condition_data_bounds)
                    all_x = np.concatenate([e[3] for e in eq_list if e[3] is not None])
                    all_y = np.concatenate([e[4] for e in eq_list if e[4] is not None])
                    if len(all_x) > 0:
                        data_bounds[x_var] = (float(np.nanmin(all_x)), float(np.nanmax(all_x)))
                    if len(all_y) > 0:
                        data_bounds[y_var] = (float(np.nanmin(all_y)), float(np.nanmax(all_y)))
                    fit_labels = self._build_gradient_segment_labels(
                        best_fit, x_var=x_var, y_var=y_var, data_bounds=data_bounds
                    )
                anchor = self._display_fit_info(
                    ax, eq_list, show_equations, show_error, fit_labels=fit_labels
                )

            fit_corner = (anchor[1], anchor[2]) if anchor else None
            legend = self._add_standard_legend(ax, avoid_corner=fit_corner)

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
                rebuilt = ax.legend(
                    handles=list(existing) + [patch],
                    labels=existing_labels + [f"Gate: {cg_label}"],
                    fancybox=True, framealpha=0.92,
                    edgecolor="#3C3C3C", borderpad=0.55, handlelength=1.8,
                    prop={"family": "Montserrat", "weight": "bold", "size": 12},
                )
                rebuilt.get_frame().set_linewidth(1.4)
                rebuilt.set_zorder(10)
                self._colorize_legend_labels(rebuilt)

            # ── annotate_fit_at ───────────────────────────────────────────
            if annotate_fit_at is not None and fit_line_params:
                # Support single value or tuple/list of values
                if isinstance(annotate_fit_at, (list, tuple)):
                    x_at_values = [float(v) for v in annotate_fit_at]
                else:
                    x_at_values = [float(annotate_fit_at)]

                xl, xr = ax.get_xlim()
                for x_at in x_at_values:
                    if not (xl <= x_at <= xr):
                        continue
                    ax.axvline(x_at, color="#5E5E5E", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

                    # Collect all annotation candidates first
                    ann_items = []
                    for entry in eq_list:
                        label_name, _, color_e = entry[0], entry[1], entry[2]
                        if label_name not in fit_line_params:
                            continue
                        slopes_p, intercepts_p = fit_line_params[label_name]
                        # Multi-segment fits: use the first valid segment
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
                        # Sort by y-value so we can resolve collisions bottom-up
                        ann_items.sort(key=lambda t: t[0])

                        # Convert y data values to display points to detect overlap
                        trans = ax.transData
                        display_ys = [trans.transform((x_at, item[0]))[1] for item in ann_items]

                        # Minimum separation between label centres (points)
                        min_sep = 18
                        # Push labels apart in display space when they collide
                        adjusted_display_ys = list(display_ys)
                        for i in range(1, len(adjusted_display_ys)):
                            gap = adjusted_display_ys[i] - adjusted_display_ys[i - 1]
                            if gap < min_sep:
                                adjusted_display_ys[i] = adjusted_display_ys[i - 1] + min_sep

                        # Determine label side: flip to left when x_at is in
                        # the right 20% of the axis range, or alternate sides
                        # when multiple x_at values are close together.
                        x_frac = (x_at - xl) / (xr - xl) if (xr - xl) > 0 else 0.5
                        place_left = x_frac > 0.80

                        # Render each annotation with collision-aware offset
                        for i, (y_at, color_e, _) in enumerate(ann_items):
                            base_display_y = display_ys[i]
                            target_display_y = adjusted_display_ys[i]
                            # Extra offset in points needed beyond the natural position
                            nudge_pts = target_display_y - base_display_y
                            y_offset = 8 + nudge_pts

                            # Alternate sides for stacked labels to reduce overlap
                            if place_left or (len(ann_items) > 2 and i % 2 == 1):
                                x_text_offset = -10
                                ha = "right"
                            else:
                                x_text_offset = 10
                                ha = "left"

                            ax.scatter([x_at], [y_at], color=color_e, s=50, zorder=10,
                                       edgecolors="white", linewidths=1.2)
                            ax.annotate(
                                f"{y_at:.3g}",
                                xy=(x_at, y_at), xytext=(x_text_offset, y_offset),
                                textcoords="offset points",
                                fontsize=9, fontweight="bold", color=color_e,
                                ha=ha,
                                zorder=11,
                                arrowprops=dict(arrowstyle="-", color=color_e,
                                                lw=0.8, alpha=0.6),
                                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                          alpha=0.92, edgecolor=color_e, linewidth=0.8),
                            )

            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")
