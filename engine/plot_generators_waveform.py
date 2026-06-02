"""Waveform plot generator mixin for DataPlotter."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from . import datafunctions
from .datafunctions import _tqdm
from .logger import log


class WaveformMixin:
    """Waveform plot generation methods. Mixed into DataPlotter."""

    # ------------------------------------------------------------------
    # Waveform helpers
    # ------------------------------------------------------------------

    def _normalize_waveform_row_spec(self, row_spec):
        """Normalize a waveform row spec to (primary_channel, secondary_channel_or_None)."""
        if isinstance(row_spec, str):
            return row_spec, None

        if isinstance(row_spec, (list, tuple)):
            if len(row_spec) == 1 and isinstance(row_spec[0], str):
                return row_spec[0], None
            if len(row_spec) == 2 and all(isinstance(v, str) for v in row_spec):
                return row_spec[0], row_spec[1]

        raise ValueError(
            "Waveform channel row must be 'channel' or ('primary_channel', 'secondary_channel')."
        )

    def _normalize_waveform_axis_limits(self, raw_limits, has_secondary, row_name):
        """Normalize waveform y-limit config for one row."""
        if raw_limits is None:
            return None, None

        if not has_secondary:
            return raw_limits, None

        if (
            isinstance(raw_limits, (list, tuple))
            and len(raw_limits) == 2
            and all(isinstance(v, (list, tuple)) or v is None for v in raw_limits)
        ):
            return raw_limits[0], raw_limits[1]

        if self.verbose:
            log.debug(
                "Waveform row '%s': dual-channel row expects axis limits as "
                "((y1_min,y1_max),(y2_min,y2_max)). Applying provided limits to primary channel only.",
                row_name,
            )
        return raw_limits, None

    def _normalize_waveform_reference_lines(self, raw_refs, has_secondary):
        """Normalize waveform reference-line config for one row."""
        if raw_refs is None:
            return None, None

        if not has_secondary:
            return raw_refs, None

        if isinstance(raw_refs, (list, tuple)) and len(raw_refs) == 2:
            return raw_refs[0], raw_refs[1]

        return raw_refs, None

    def _prepare_waveform_channels(self, channels, axis_limits, reference_lines, subplot_heights):
        """Build validated waveform rows with optional two-channel overlays."""
        prepared_rows = []
        row_heights = []

        for i, row_spec in enumerate(channels):
            primary, secondary = self._normalize_waveform_row_spec(row_spec)

            p_count = sum(primary in self.run_data[r["name"].lower()].columns for r in self.runs)
            s_count = (
                sum(secondary in self.run_data[r["name"].lower()].columns for r in self.runs)
                if secondary is not None
                else 0
            )

            if p_count == 0 and (secondary is None or s_count == 0):
                missing_name = (
                    f"'{primary}' and '{secondary}'" if secondary is not None else f"'{primary}'"
                )
                # Show suggestions from the first available run
                hints = []
                for r in self.runs:
                    rn = r["name"].lower()
                    if rn in self.run_data:
                        for ch in ([primary] + ([secondary] if secondary else [])):
                            h = self._format_missing_channel_hint(rn, ch)
                            if h:
                                hints.append(h)
                        break
                log.warning(
                    "Waveform row %s missing from all runs. Skipping row.%s",
                    missing_name, f"\n{''.join(hints)}" if hints else "",
                )
                continue

            if p_count == 0 and secondary is not None and s_count > 0:
                log.warning(
                    "Waveform row primary channel '%s' missing in all runs; using '%s' as single-channel row.",
                    primary, secondary,
                )
                primary, secondary = secondary, None
                p_count = s_count
                s_count = 0

            if p_count < len(self.runs):
                log.warning(
                    "Waveform channel '%s' present in %d/%d runs. Plotting available runs only.",
                    primary, p_count, len(self.runs),
                )

            if secondary is not None:
                if s_count == 0:
                    log.warning(
                        "Waveform secondary channel '%s' missing from all runs; rendering row as single-channel.",
                        secondary,
                    )
                    secondary = None
                elif s_count < len(self.runs):
                    log.warning(
                        "Waveform secondary channel '%s' present in %d/%d runs. Plotting available runs only.",
                        secondary, s_count, len(self.runs),
                    )

            raw_lim = axis_limits[i] if axis_limits and i < len(axis_limits) else None
            raw_ref = reference_lines[i] if reference_lines and i < len(reference_lines) else None
            y1_lim, y2_lim = self._normalize_waveform_axis_limits(raw_lim, secondary is not None, primary)
            y1_refs, y2_refs = self._normalize_waveform_reference_lines(raw_ref, secondary is not None)

            prepared_rows.append({
                "primary": primary,
                "secondary": secondary,
                "y1_lim": y1_lim,
                "y2_lim": y2_lim,
                "y1_refs": y1_refs,
                "y2_refs": y2_refs,
            })
            row_heights.append(
                subplot_heights[i] if subplot_heights and i < len(subplot_heights) else 1.0
            )

        return prepared_rows, row_heights

    def _format_waveform_channel_label(self, channel, *, secondary=False, show_style_hint=False):
        """Format waveform channel label and optional line-style hint."""
        base = datafunctions.add_units_to_label(channel, units_map=self.units_map)
        if not show_style_hint:
            return base
        style_hint = "- - - - -" if secondary else "_______"
        return f"{base}\n{style_hint}"

    # ------------------------------------------------------------------
    # Waveform generator
    # ------------------------------------------------------------------

    def generate_waveform_plots(self):
        """Generate all configured waveform subplot figures."""
        self._ensure_preprocessed()
        plots = self._get_plot_group(0)
        if not plots:
            return

        plot_iter = plots if self.verbose else _tqdm(plots, desc="Waveform", unit="plot", leave=True)
        for plot_def in plot_iter:
            # Typed dataclass access (#9/#24).
            plot_name       = plot_def.name
            channels        = plot_def.channels
            axis_limits     = plot_def.axis_limits
            ref_lines       = plot_def.reference_lines
            subplot_heights = plot_def.subplot_heights
            x_limits        = plot_def.x_limits
            x_channel       = plot_def.x_channel or "sLap"
            highlight_zones = plot_def.highlight_zones
            normalise       = plot_def.normalise
            legend_position = plot_def.legend_position
            show_delta      = plot_def.show_delta
            markers         = plot_def.markers

            if self.verbose:
                log.debug("Creating waveform plot: %s", plot_name)

            prepared_rows, avail_heights = self._prepare_waveform_channels(
                channels, axis_limits, ref_lines, subplot_heights
            )

            if not prepared_rows:
                log.warning("No valid channels for '%s' -- skipping.", plot_name)
                continue

            filename = self._sanitize_plot_filename("waveform", plot_name)
            min_height = 1.6 * sum(avail_heights)

            # Resolve which runs actually have data loaded — needed for delta gating.
            loaded_run_names = [r["name"].lower() for r in self.runs if r["name"].lower() in self.run_data]
            delta_active = any(show_delta) and len(loaded_run_names) == 2

            if delta_active:
                # Each prepared row with show_delta=True is followed by a half-height delta row.
                expanded_heights = []
                for i, h in enumerate(avail_heights):
                    expanded_heights.append(h)
                    if show_delta[i]:
                        expanded_heights.append(h * 0.45)
                n_axes = len(expanded_heights)
                min_height = 1.4 * sum(expanded_heights)
            else:
                expanded_heights = list(avail_heights)
                n_axes = len(prepared_rows)

            figsize = self._resolve_plot_figsize(filename, self.waveform_figsize, min_height=min_height)

            fig, axes = plt.subplots(
                n_axes,
                1,
                figsize=figsize,
                sharex=True,
                squeeze=False,
                gridspec_kw={"height_ratios": expanded_heights},
            )
            axes = axes.flatten()
            plotted_runs = set()

            # Resolve the x-axis channel, falling back gracefully if unavailable
            x_channel_available = x_channel and all(
                x_channel in self.run_data[rn].columns for rn in loaded_run_names
            )
            if not x_channel_available:
                fallback = "sLap" if x_channel != "sLap" else None
                if fallback and all(fallback in self.run_data[rn].columns for rn in loaded_run_names):
                    log.warning(
                        "Waveform '%s': x_channel '%s' not available in all runs. Falling back to 'sLap'.",
                        plot_name, x_channel,
                    )
                    x_channel = "sLap"
                    x_channel_available = True
                else:
                    log.warning(
                        "Waveform '%s': x_channel '%s' not available. Using row index.",
                        plot_name, x_channel,
                    )
                    x_channel = None

            # Build xlabel using units_map if available
            if x_channel:
                unit = (self.units_map or {}).get(x_channel, (self.units_map or {}).get(x_channel.lower(), ""))
                xlabel = f"{x_channel} ({unit})" if unit else x_channel
            else:
                xlabel = "Sample"

            # ── normalise: pre-compute per-channel global min/max ─────────
            channel_ranges = {}
            if normalise:
                all_channels = set()
                for row in prepared_rows:
                    all_channels.add(row["primary"])
                    if row["secondary"]:
                        all_channels.add(row["secondary"])
                for ch in all_channels:
                    vals = []
                    for run in self.runs:
                        df = self.run_data.get(run["name"].lower())
                        if df is not None and ch in df.columns:
                            v = df[ch].dropna().to_numpy(dtype=float)
                            if len(v):
                                vals.extend(v)
                    if vals:
                        lo, hi = float(np.min(vals)), float(np.max(vals))
                        channel_ranges[ch] = (lo, hi - lo) if hi != lo else (lo, 1.0)

            for idx, row in enumerate(prepared_rows):
                # Compute axis index: count preceding rows + their delta subplots
                ax_idx = idx + sum(1 for i in range(idx) if show_delta[i]) if delta_active else idx
                ax = axes[ax_idx]
                ax_delta = axes[ax_idx + 1] if (delta_active and show_delta[idx]) else None
                ch_primary = row["primary"]
                ch_secondary = row["secondary"]
                ax_right = (ax.twinx() if ch_secondary is not None else None) if not normalise else None

                # Collect per-run primary traces for delta computation
                delta_traces = {}  # rn -> (x_arr, y_arr)

                for run in self.runs:
                    rn = run["name"].lower()
                    if rn not in self.run_data:
                        continue
                    df = self.run_data[rn]
                    if ch_primary not in df.columns:
                        continue

                    x_vals = df[x_channel] if (x_channel and x_channel in df.columns) else df.index
                    x_plot, y_plot = datafunctions.mask_waveform_discontinuities(x_vals, df[ch_primary])

                    if normalise and ch_primary in channel_ranges:
                        lo, rng = channel_ranges[ch_primary]
                        y_plot = (np.array(y_plot, dtype=float) - lo) / rng

                    ax.plot(
                        x_plot, y_plot,
                        linewidth=1.6, color=run["color"],
                        label=run["name"].upper(), alpha=0.85,
                    )
                    plotted_runs.add(rn)

                    if ax_delta is not None:
                        delta_traces[rn] = (
                            np.asarray(x_plot, dtype=float),
                            np.asarray(y_plot, dtype=float),
                        )

                    if ax_right is not None and ch_secondary in df.columns:
                        x2_plot, y2_plot = datafunctions.mask_waveform_discontinuities(
                            x_vals, df[ch_secondary]
                        )
                        ax_right.plot(
                            x2_plot, y2_plot,
                            linewidth=1.45, linestyle="--",
                            color=run["color"], label="_nolegend_", alpha=0.85,
                        )
                        plotted_runs.add(rn)
                    elif normalise and ch_secondary and ch_secondary in df.columns:
                        # normalise mode: secondary on same axis with dashed line
                        x2_plot, y2_plot = datafunctions.mask_waveform_discontinuities(
                            x_vals, df[ch_secondary]
                        )
                        if ch_secondary in channel_ranges:
                            lo2, rng2 = channel_ranges[ch_secondary]
                            y2_plot = (np.array(y2_plot, dtype=float) - lo2) / rng2
                        ax.plot(
                            x2_plot, y2_plot,
                            linewidth=1.45, linestyle="--",
                            color=run["color"], label="_nolegend_", alpha=0.85,
                        )
                        plotted_runs.add(rn)

                if normalise:
                    ch_label = ch_primary
                    if ch_secondary:
                        ch_label = f"{ch_primary} / {ch_secondary}"
                    ax.set_ylabel(
                        f"{ch_label}\n(norm.)",
                        fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center",
                    )
                    ax.set_ylim(-0.05, 1.05)
                else:
                    ax.set_ylabel(
                        self._format_waveform_channel_label(
                            ch_primary, secondary=False, show_style_hint=(ch_secondary is not None)
                        ),
                        fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center",
                    )
                ax.yaxis.set_label_coords(-0.035, 0.5)
                self._apply_grid(ax, which="major", axis="y")

                if not normalise and row["y1_lim"] is not None:
                    yl, yh = row["y1_lim"]
                    yl = yl if (yl is None or np.isscalar(yl)) else None
                    yh = yh if (yh is None or np.isscalar(yh)) else None
                    if yl is not None or yh is not None:
                        ax.set_ylim(bottom=yl, top=yh)

                if not normalise and row["y1_refs"] is not None:
                    vals = [row["y1_refs"]] if np.isscalar(row["y1_refs"]) else row["y1_refs"]
                    for vv in vals:
                        ax.axhline(vv, linestyle="--", linewidth=0.8, color="#4A4A4A", alpha=0.65, zorder=1)

                if ax_right is not None:
                    ax_right.set_ylabel(
                        self._format_waveform_channel_label(
                            ch_secondary, secondary=True, show_style_hint=True
                        ),
                        fontsize=9.5, fontweight="bold", rotation=0, ha="left", va="center",
                    )
                    ax_right.yaxis.set_label_coords(1.03, 0.5)
                    ax_right.spines["top"].set_visible(False)
                    ax_right.grid(False)
                    ax_right.tick_params(axis="y", labelsize=8.5)

                    if row["y2_lim"] is not None:
                        yl2, yh2 = row["y2_lim"]
                        yl2 = yl2 if (yl2 is None or np.isscalar(yl2)) else None
                        yh2 = yh2 if (yh2 is None or np.isscalar(yh2)) else None
                        if yl2 is not None or yh2 is not None:
                            ax_right.set_ylim(bottom=yl2, top=yh2)

                    if row["y2_refs"] is not None:
                        vals2 = [row["y2_refs"]] if np.isscalar(row["y2_refs"]) else row["y2_refs"]
                        for vv2 in vals2:
                            ax_right.axhline(vv2, linestyle="--", linewidth=0.8, color="#4A4A4A", alpha=0.55, zorder=1)

                # ── highlight_zones ───────────────────────────────────────
                # Gate spec: ('ch', 'op', val) or ('ch', 'op', val, '#hexcolor')
                # or list of such tuples. Optional color overrides the run color.
                if highlight_zones is not None:
                    # Normalise to a 3-element spec tuple and extract optional color
                    if isinstance(highlight_zones[0], (list, tuple)):
                        z_spec = list(highlight_zones)
                        z_override_color = None
                    else:
                        z_spec = highlight_zones[:3]
                        z_override_color = (
                            highlight_zones[3]
                            if len(highlight_zones) >= 4 and isinstance(highlight_zones[3], str)
                            else None
                        )

                    for run in self.runs:
                        df_z = self.run_data.get(run["name"].lower())
                        if df_z is None:
                            continue
                        # Check the gate channel(s) exist
                        spec_list = z_spec if isinstance(z_spec[0], (list, tuple)) else [z_spec]
                        if not all(s[0] in df_z.columns for s in spec_list):
                            continue
                        x_z = df_z[x_channel] if (x_channel and x_channel in df_z.columns) else df_z.index
                        df_cond = datafunctions.apply_gate_to_dataframe(df_z, z_spec)
                        if df_cond is None or df_cond.empty:
                            continue
                        mask_z = df_z.index.isin(df_cond.index)
                        x_arr = x_z.to_numpy() if hasattr(x_z, "to_numpy") else np.array(x_z)
                        m_arr = mask_z.to_numpy() if hasattr(mask_z, "to_numpy") else np.array(mask_z, dtype=bool)
                        shade_color = z_override_color if z_override_color else run["color"]
                        # Draw one axvspan per contiguous True segment
                        padded = np.concatenate([[False], m_arr, [False]])
                        starts = np.where(~padded[:-1] & padded[1:])[0]
                        ends   = np.where( padded[:-1] & ~padded[1:])[0]
                        for s, e in zip(starts, ends):
                            xe = min(e, len(x_arr) - 1)
                            ax.axvspan(
                                x_arr[s], x_arr[xe],
                                alpha=0.15, color=shade_color, zorder=0, linewidth=0,
                            )

                if idx < len(prepared_rows) - 1:
                    ax.tick_params(labelbottom=False)
                elif ax_delta is not None:
                    # Last data row: hide its x labels because the delta row owns them.
                    ax.tick_params(labelbottom=False)

                # ── delta subplot: plot (run_B − run_A) for the primary channel ──
                if ax_delta is not None and len(delta_traces) == 2:
                    # Preserve run order from self.runs
                    ordered = [r["name"].lower() for r in self.runs if r["name"].lower() in delta_traces]
                    rn_a, rn_b = ordered[0], ordered[1]
                    xa, ya = delta_traces[rn_a]
                    xb, yb = delta_traces[rn_b]
                    # Align on rn_a's x by linear interpolation of yb.
                    finite_a = np.isfinite(xa) & np.isfinite(ya)
                    finite_b = np.isfinite(xb) & np.isfinite(yb)
                    if finite_a.sum() >= 2 and finite_b.sum() >= 2:
                        xa_f, ya_f = xa[finite_a], ya[finite_a]
                        xb_f, yb_f = xb[finite_b], yb[finite_b]
                        order_b = np.argsort(xb_f)
                        yb_on_a = np.interp(xa_f, xb_f[order_b], yb_f[order_b],
                                            left=np.nan, right=np.nan)
                        delta = yb_on_a - ya_f
                        ax_delta.plot(
                            xa_f, delta,
                            linewidth=1.2, color="#3C3C3C", alpha=0.95,
                        )
                        ax_delta.axhline(0, color="#9A9A9A", linewidth=0.8, alpha=0.7, zorder=1)
                        ax_delta.set_ylabel(
                            f"Δ {ch_primary}\n({rn_b.upper()}−{rn_a.upper()})",
                            fontsize=8.5, fontweight="bold", rotation=0, ha="right", va="center",
                        )
                        ax_delta.yaxis.set_label_coords(-0.035, 0.5)
                        ax_delta.tick_params(axis="y", labelsize=8)
                        self._apply_grid(ax_delta, which="major", axis="y")
                        if idx < len(prepared_rows) - 1:
                            ax_delta.tick_params(labelbottom=False)

            # X-axis styling
            axes[-1].set_xlabel(xlabel, fontweight="bold")
            axes[-1].tick_params(axis="x", labelsize=10)

            if x_limits is not None:
                xmin, xmax = x_limits
                if xmin is not None or xmax is not None:
                    for ax in axes:
                        ax.set_xlim(left=xmin, right=xmax)
            else:
                if x_channel == "sLap":
                    # Round up to nearest 100 m and start from 0
                    xmaxs = [xm for ax in axes for _, xm in [ax.get_xlim()] if xm > 0]
                    if xmaxs:
                        xv = np.ceil(max(xmaxs) / 100) * 100
                        for ax in axes:
                            ax.set_xlim(0, xv)

                for ax in axes:
                    ax.xaxis.set_major_locator(
                        ticker.MaxNLocator(nbins=8, min_n_ticks=5, steps=[1, 2, 2.5, 5, 10])
                    )
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                    self._apply_grid(ax, which="both", axis="x")

            # ── Markers (#6): vertical reference lines ────────────────────────
            # Static markers (concrete x): drawn once per figure with a boxed
            # label at the top edge (well-separated horizontally by x).
            # Condition markers: resolved per-run (rising/falling edges of a
            # gate condition), drawn in the run's colour at each transition. The
            # label is placed rotated 90° INSIDE the plot, attached to the line
            # at its first hit, so labels stay near their lines instead of
            # piling up at the top corner. Vertical slot is keyed off the
            # condition-marker index (not the run) so DRY/WET share a slot —
            # colour disambiguates them, and hits at different x rarely overlap.
            if markers:
                static_markers = [m for m in markers if m.x is not None]
                cond_markers   = [m for m in markers if m.condition is not None]

                def _draw_line(ax_m, x_val, color, linestyle):
                    ax_m.axvline(x_val, color=color, linestyle=linestyle,
                                 linewidth=1.2, alpha=0.7, zorder=2)

                def _draw_static_label(ax_m, x_val, color, label):
                    ax_m.text(
                        x_val, 1.01, label,
                        transform=ax_m.get_xaxis_transform(),
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=color, linewidth=0.8, alpha=0.9),
                        zorder=12,
                    )

                def _draw_inline_label(ax_m, x_val, color, label, y_frac):
                    # Horizontal label offset a few points to the right of the
                    # vertical line so the line itself stays unobscured. Anchor
                    # x in data coords + y in axes-fraction via xaxis_transform,
                    # then nudge by a pixel offset using ``offset points``.
                    ax_m.annotate(
                        label,
                        xy=(x_val, y_frac),
                        xycoords=ax_m.get_xaxis_transform(),
                        xytext=(6, 0),
                        textcoords="offset points",
                        ha="left", va="center",
                        fontsize=8, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=color, linewidth=0.6, alpha=0.9),
                        zorder=12,
                    )

                # Static markers ─ same x for every run.
                for m in static_markers:
                    color = m.color or "#5E5E5E"
                    rows_to_draw = (
                        [m.row]
                        if m.row is not None and 0 <= m.row < len(axes)
                        else list(range(len(axes)))
                    )
                    for ridx in rows_to_draw:
                        _draw_line(axes[ridx], m.x, color, m.linestyle)
                    # Label only on the topmost drawn row.
                    if m.label and m.show_label:
                        _draw_static_label(axes[rows_to_draw[0]], m.x, color, m.label)

                # Condition markers ─ resolved per run on the original (un-clipped)
                # dataframe so transitions outside x_limits are still detected.
                if cond_markers:
                    # Stagger vertical slot by marker index inside the topmost
                    # row so multiple condition markers don't collide. Top slot
                    # at 0.96, stepping down by 0.12 of the axes height (tight
                    # enough to keep labels visually grouped near the top).
                    n_slots   = max(len(cond_markers), 1)
                    slot_step = 0.12 if n_slots > 1 else 0.0
                    slot_y    = [0.96 - i * slot_step for i in range(n_slots)]

                    plotted_run_names = [
                        r["name"] for r in self.runs if r["name"].lower() in plotted_runs
                    ]

                    # First pass: draw every line in run colour.
                    # Track the earliest x-hit per condition marker so we can
                    # place a single neutral label per marker afterwards.
                    earliest_hit = [None] * len(cond_markers)
                    marker_rows  = [None] * len(cond_markers)
                    for run_name in plotted_run_names:
                        run_df = self.run_data.get(run_name.lower())
                        if run_df is None or x_channel not in run_df.columns:
                            continue
                        run_color = next(
                            (r["color"] for r in self.runs if r["name"] == run_name),
                            "#5E5E5E",
                        )
                        for m_idx, m in enumerate(cond_markers):
                            x_hits = datafunctions.resolve_condition_marker(
                                m, run_df, x_channel,
                            )
                            if not x_hits:
                                continue
                            color = m.color or run_color
                            rows_to_draw = (
                                [m.row]
                                if m.row is not None and 0 <= m.row < len(axes)
                                else list(range(len(axes)))
                            )
                            marker_rows[m_idx] = rows_to_draw
                            for x_val in x_hits:
                                for ridx in rows_to_draw:
                                    _draw_line(axes[ridx], x_val, color, m.linestyle)
                            if (earliest_hit[m_idx] is None
                                    or x_hits[0] < earliest_hit[m_idx]):
                                earliest_hit[m_idx] = x_hits[0]

                    # Second pass: one inline label per condition marker, placed
                    # at the earliest hit across runs. Colour is the marker's
                    # explicit colour if set, otherwise a neutral grey so it
                    # doesn't favour one run over the other.
                    for m_idx, m in enumerate(cond_markers):
                        if (not m.label
                                or not m.show_label
                                or earliest_hit[m_idx] is None
                                or marker_rows[m_idx] is None):
                            continue
                        label_color = m.color or "#3A3A3A"
                        _draw_inline_label(
                            axes[marker_rows[m_idx][0]],
                            earliest_hit[m_idx], label_color, m.label,
                            slot_y[m_idx],
                        )

            # Legend above (default) or to the right of subplots
            run_handles = [
                Line2D([0], [0], color=run["color"], linewidth=2.0)
                for run in self.runs if run["name"].lower() in plotted_runs
            ]
            run_labels = [
                run["name"].upper()
                for run in self.runs if run["name"].lower() in plotted_runs
            ]
            self._add_waveform_figure_legend(fig, run_handles, run_labels, position=legend_position)

            if legend_position == "right":
                # Reserve space on the right for the side legend.
                plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 0.88, 1.0))
            else:
                plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.95))
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)
