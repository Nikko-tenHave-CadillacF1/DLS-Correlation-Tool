"""Module-level generator functions extracted from ``WaveformMixin``.

Extracted 2026-07 (Prompt 12 Phase 2). Function bodies are the class's
original method bodies with a mechanical ``self`` -> ``plotter`` rename;
the old file at ``engine/plot_generators_*.py`` keeps a thin ``class ..Mixin``
shim whose methods delegate here so ``DataPlotter``'s multiple inheritance
keeps working unchanged.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

from .. import datafunctions
from ..datafunctions import _tqdm
from ..logger import log


def _normalize_waveform_row_spec(plotter, row_spec):
    if isinstance(row_spec, str):
        return row_spec, None
    if isinstance(row_spec, (list, tuple)):
        if len(row_spec) == 1 and isinstance(row_spec[0], str):
            return row_spec[0], None
        if len(row_spec) == 2 and all(isinstance(v, str) for v in row_spec):
            return row_spec[0], row_spec[1]
    raise ValueError("Waveform channel row must be 'channel' or ('primary_channel', 'secondary_channel').")


def _normalize_waveform_axis_limits(plotter, raw_limits, has_secondary, row_name):
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
    if plotter.verbose:
        log.debug(
            "Waveform row '%s': dual-channel row expects axis limits as "
            "((y1_min,y1_max),(y2_min,y2_max)). Applying provided limits to primary channel only.",
            row_name,
        )
    return raw_limits, None


def _normalize_waveform_reference_lines(plotter, raw_refs, has_secondary):
    if raw_refs is None:
        return None, None
    if not has_secondary:
        return raw_refs, None
    if isinstance(raw_refs, (list, tuple)) and len(raw_refs) == 2:
        return raw_refs[0], raw_refs[1]
    return raw_refs, None


def _prepare_waveform_channels(plotter, channels, axis_limits, reference_lines, subplot_heights):
    prepared_rows = []
    row_heights = []
    for i, row_spec in enumerate(channels):
        primary, secondary = plotter._normalize_waveform_row_spec(row_spec)
        p_count = sum(primary in plotter.run_data[r["name"].lower()].columns for r in plotter.runs)
        s_count = (
            sum(secondary in plotter.run_data[r["name"].lower()].columns for r in plotter.runs)
            if secondary is not None
            else 0
        )
        if p_count == 0 and (secondary is None or s_count == 0):
            missing_name = f"'{primary}' and '{secondary}'" if secondary is not None else f"'{primary}'"
            hints = []
            for r in plotter.runs:
                rn = r["name"].lower()
                if rn in plotter.run_data:
                    for ch in [primary] + ([secondary] if secondary else []):
                        h = plotter._format_missing_channel_hint(rn, ch)
                        if h:
                            hints.append(h)
                    break
            log.warning(
                "Waveform row %s missing from all runs. Skipping row.%s",
                missing_name,
                f"\n{''.join(hints)}" if hints else "",
            )
            continue
        if p_count == 0 and secondary is not None and s_count > 0:
            log.warning(
                "Waveform row primary channel '%s' missing in all runs; using '%s' as single-channel row.",
                primary,
                secondary,
            )
            primary, secondary = secondary, None
            p_count = s_count
            s_count = 0
        if p_count < len(plotter.runs):
            log.warning(
                "Waveform channel '%s' present in %d/%d runs. Plotting available runs only.",
                primary,
                p_count,
                len(plotter.runs),
            )
        if secondary is not None:
            if s_count == 0:
                log.warning(
                    "Waveform secondary channel '%s' missing from all runs; rendering row as single-channel.",
                    secondary,
                )
                secondary = None
            elif s_count < len(plotter.runs):
                log.warning(
                    "Waveform secondary channel '%s' present in %d/%d runs. Plotting available runs only.",
                    secondary,
                    s_count,
                    len(plotter.runs),
                )
        raw_lim = axis_limits[i] if axis_limits and i < len(axis_limits) else None
        raw_ref = reference_lines[i] if reference_lines and i < len(reference_lines) else None
        y1_lim, y2_lim = plotter._normalize_waveform_axis_limits(raw_lim, secondary is not None, primary)
        y1_refs, y2_refs = plotter._normalize_waveform_reference_lines(raw_ref, secondary is not None)
        prepared_rows.append(
            {
                "primary": primary,
                "secondary": secondary,
                "y1_lim": y1_lim,
                "y2_lim": y2_lim,
                "y1_refs": y1_refs,
                "y2_refs": y2_refs,
            }
        )
        row_heights.append(subplot_heights[i] if subplot_heights and i < len(subplot_heights) else 1.0)
    return prepared_rows, row_heights


def _format_waveform_channel_label(plotter, channel, *, secondary=False, show_style_hint=False):
    base = datafunctions.add_units_to_label(channel, units_map=plotter.units_map)
    if not show_style_hint:
        return base
    style_hint = "- - - - -" if secondary else "_______"
    return f"{base}\n{style_hint}"


def generate_waveform_plots(plotter):
    plotter._ensure_preprocessed()
    plots = plotter._get_plot_group(0)
    if not plots:
        return
    plot_iter = plots if plotter.verbose else _tqdm(plots, desc="Waveform", unit="plot", leave=True)
    for plot_def in plot_iter:
        plot_name = plot_def.name
        channels = plot_def.channels
        axis_limits = plot_def.axis_limits
        ref_lines = plot_def.reference_lines
        subplot_heights = plot_def.subplot_heights
        x_limits = plot_def.x_limits
        x_channel = plot_def.x_channel or "sLap"
        highlight_zones = plot_def.highlight_zones
        normalise = plot_def.normalise
        legend_position = plot_def.legend_position
        show_delta = plot_def.show_delta
        markers = plot_def.markers
        annotate_at = plot_def.annotate_at
        if plotter.verbose:
            log.debug("Creating waveform plot: %s", plot_name)
        prepared_rows, avail_heights = plotter._prepare_waveform_channels(
            channels, axis_limits, ref_lines, subplot_heights
        )
        if not prepared_rows:
            log.warning("No valid channels for '%s' -- skipping.", plot_name)
            continue
        filename = plotter._sanitize_plot_filename("waveform", plot_name)
        min_height = 1.6 * sum(avail_heights)
        loaded_run_names = [r["name"].lower() for r in plotter.runs if r["name"].lower() in plotter.run_data]
        delta_active = any(show_delta) and len(loaded_run_names) >= 2
        ref_run_name = plotter.reference_run_name() if delta_active else None
        if delta_active:
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
        figsize = plotter._resolve_plot_figsize(filename, plotter.waveform_figsize, min_height=min_height)
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
        x_channel_available = x_channel and all(x_channel in plotter.run_data[rn].columns for rn in loaded_run_names)
        if not x_channel_available:
            fallback = "sLap" if x_channel != "sLap" else None
            if fallback and all(fallback in plotter.run_data[rn].columns for rn in loaded_run_names):
                log.warning(
                    "Waveform '%s': x_channel '%s' not available in all runs. Falling back to 'sLap'.",
                    plot_name,
                    x_channel,
                )
                x_channel = "sLap"
                x_channel_available = True
            else:
                log.warning(
                    "Waveform '%s': x_channel '%s' not available. Using row index.",
                    plot_name,
                    x_channel,
                )
                x_channel = None
        if x_channel:
            unit = (plotter.units_map or {}).get(x_channel, (plotter.units_map or {}).get(x_channel.lower(), ""))
            xlabel = f"{x_channel} ({unit})" if unit else x_channel
        else:
            xlabel = "Sample"
        channel_ranges = {}
        if normalise:
            all_channels = set()
            for row in prepared_rows:
                all_channels.add(row["primary"])
                if row["secondary"]:
                    all_channels.add(row["secondary"])
            for ch in all_channels:
                vals = []
                for run in plotter.runs:
                    df = plotter.run_data.get(run["name"].lower())
                    if df is not None and ch in df.columns:
                        v = df[ch].dropna().to_numpy(dtype=float)
                        if len(v):
                            vals.extend(v)
                if vals:
                    lo, hi = float(np.min(vals)), float(np.max(vals))
                    channel_ranges[ch] = (lo, hi - lo) if hi != lo else (lo, 1.0)
        annotate_row_data = {} if annotate_at else None
        for idx, row in enumerate(prepared_rows):
            ax_idx = idx + sum(1 for i in range(idx) if show_delta[i]) if delta_active else idx
            ax = axes[ax_idx]
            ax_delta = axes[ax_idx + 1] if (delta_active and show_delta[idx]) else None
            ch_primary = row["primary"]
            ch_secondary = row["secondary"]
            ax_right = (ax.twinx() if ch_secondary is not None else None) if not normalise else None
            delta_traces = {}
            delta_traces_secondary = {}
            for run in plotter.runs:
                rn = run["name"].lower()
                if rn not in plotter.run_data:
                    continue
                df = plotter.run_data[rn]
                if ch_primary not in df.columns:
                    continue
                x_vals = df[x_channel] if (x_channel and x_channel in df.columns) else df.index
                x_plot, y_plot = datafunctions.mask_waveform_discontinuities(x_vals, df[ch_primary])
                if normalise and ch_primary in channel_ranges:
                    lo, rng = channel_ranges[ch_primary]
                    y_plot = (np.array(y_plot, dtype=float) - lo) / rng
                ax.plot(
                    x_plot,
                    y_plot,
                    linewidth=1.6,
                    color=run["color"],
                    label=run["name"].upper(),
                    alpha=0.85,
                )
                plotted_runs.add(rn)
                if annotate_row_data is not None:
                    if idx not in annotate_row_data:
                        annotate_row_data[idx] = {}
                    y_sec = None
                    if ch_secondary and ch_secondary in df.columns:
                        _, y_sec_raw = datafunctions.mask_waveform_discontinuities(x_vals, df[ch_secondary])
                        if normalise and ch_secondary in channel_ranges:
                            lo2n, rng2n = channel_ranges[ch_secondary]
                            y_sec_raw = (np.array(y_sec_raw, dtype=float) - lo2n) / rng2n
                        y_sec = np.asarray(y_sec_raw, dtype=float)
                    annotate_row_data[idx][rn] = (
                        np.asarray(x_plot, dtype=float),
                        np.asarray(y_plot, dtype=float),
                        y_sec,
                    )
                if ax_delta is not None:
                    delta_traces[rn] = (
                        np.asarray(x_plot, dtype=float),
                        np.asarray(y_plot, dtype=float),
                    )
                if ax_delta is not None and ch_secondary and ch_secondary in df.columns:
                    x2_d, y2_d = datafunctions.mask_waveform_discontinuities(x_vals, df[ch_secondary])
                    if normalise and ch_secondary in channel_ranges:
                        lo2, rng2 = channel_ranges[ch_secondary]
                        y2_d = (np.array(y2_d, dtype=float) - lo2) / rng2
                    delta_traces_secondary[rn] = (
                        np.asarray(x2_d, dtype=float),
                        np.asarray(y2_d, dtype=float),
                    )
                if ax_right is not None and ch_secondary in df.columns:
                    x2_plot, y2_plot = datafunctions.mask_waveform_discontinuities(x_vals, df[ch_secondary])
                    ax_right.plot(
                        x2_plot,
                        y2_plot,
                        linewidth=1.45,
                        linestyle="--",
                        color=run["color"],
                        label="_nolegend_",
                        alpha=0.85,
                    )
                    plotted_runs.add(rn)
                elif normalise and ch_secondary and ch_secondary in df.columns:
                    x2_plot, y2_plot = datafunctions.mask_waveform_discontinuities(x_vals, df[ch_secondary])
                    if ch_secondary in channel_ranges:
                        lo2, rng2 = channel_ranges[ch_secondary]
                        y2_plot = (np.array(y2_plot, dtype=float) - lo2) / rng2
                    ax.plot(
                        x2_plot,
                        y2_plot,
                        linewidth=1.45,
                        linestyle="--",
                        color=run["color"],
                        label="_nolegend_",
                        alpha=0.85,
                    )
                    plotted_runs.add(rn)
            if normalise:
                ch_label = ch_primary
                if ch_secondary:
                    ch_label = f"{ch_primary} / {ch_secondary}"
                ax.set_ylabel(
                    f"{ch_label}\n(norm.)",
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                )
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylabel(
                    plotter._format_waveform_channel_label(
                        ch_primary, secondary=False, show_style_hint=(ch_secondary is not None)
                    ),
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                )
            ax.yaxis.set_label_coords(-0.035, 0.5)
            plotter._apply_grid(ax, which="both", axis="y")
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
                    plotter._format_waveform_channel_label(ch_secondary, secondary=True, show_style_hint=True),
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    ha="left",
                    va="center",
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
                if delta_active and show_delta[idx] and row["y1_lim"] is None and row["y2_lim"] is None:
                    lim_l = ax.get_ylim()
                    lim_r = ax_right.get_ylim()
                    unified_lo = min(lim_l[0], lim_r[0])
                    unified_hi = max(lim_l[1], lim_r[1])
                    ax.set_ylim(unified_lo, unified_hi)
                    ax_right.set_ylim(unified_lo, unified_hi)
                if row["y2_refs"] is not None:
                    vals2 = [row["y2_refs"]] if np.isscalar(row["y2_refs"]) else row["y2_refs"]
                    for vv2 in vals2:
                        ax_right.axhline(vv2, linestyle="--", linewidth=0.8, color="#4A4A4A", alpha=0.55, zorder=1)
            if highlight_zones is not None:
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
                for run in plotter.runs:
                    df_z = plotter.run_data.get(run["name"].lower())
                    if df_z is None:
                        continue
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
                    padded = np.concatenate([[False], m_arr, [False]])
                    starts = np.where(~padded[:-1] & padded[1:])[0]
                    ends = np.where(padded[:-1] & ~padded[1:])[0]
                    for s, e in zip(starts, ends):
                        xe = min(e, len(x_arr) - 1)
                        ax.axvspan(
                            x_arr[s],
                            x_arr[xe],
                            alpha=0.15,
                            color=shade_color,
                            zorder=0,
                            linewidth=0,
                        )
            if idx < len(prepared_rows) - 1 or ax_delta is not None:
                ax.tick_params(labelbottom=False)
            if ax_delta is not None and ref_run_name in delta_traces and len(delta_traces) >= 2:
                xa, ya = delta_traces[ref_run_name]
                finite_a = np.isfinite(xa) & np.isfinite(ya)
                if finite_a.sum() >= 2:
                    xa_f, ya_f = xa[finite_a], ya[finite_a]
                    order_a = np.argsort(xa_f)
                    xa_s, ya_s = xa_f[order_a], ya_f[order_a]
                    for run in plotter.runs:
                        rn = run["name"].lower()
                        if rn == ref_run_name or rn not in delta_traces:
                            continue
                        xb, yb = delta_traces[rn]
                        finite_b = np.isfinite(xb) & np.isfinite(yb)
                        if finite_b.sum() < 2:
                            continue
                        xb_f, yb_f = xb[finite_b], yb[finite_b]
                        order_b = np.argsort(xb_f)
                        yb_on_a = np.interp(
                            xa_s,
                            xb_f[order_b],
                            yb_f[order_b],
                            left=np.nan,
                            right=np.nan,
                        )
                        delta = yb_on_a - ya_s
                        ax_delta.plot(
                            xa_s,
                            delta,
                            linewidth=1.2,
                            color=run["color"],
                            alpha=0.95,
                            label="_nolegend_",
                        )
                ax_delta_right = None
                if ch_secondary and ref_run_name in delta_traces_secondary and len(delta_traces_secondary) >= 2:
                    ax_delta_right = ax_delta.twinx()
                    xa2, ya2 = delta_traces_secondary[ref_run_name]
                    finite_a2 = np.isfinite(xa2) & np.isfinite(ya2)
                    if finite_a2.sum() >= 2:
                        xa2_f, ya2_f = xa2[finite_a2], ya2[finite_a2]
                        order_a2 = np.argsort(xa2_f)
                        xa2_s, ya2_s = xa2_f[order_a2], ya2_f[order_a2]
                        for run in plotter.runs:
                            rn = run["name"].lower()
                            if rn == ref_run_name or rn not in delta_traces_secondary:
                                continue
                            xb2, yb2 = delta_traces_secondary[rn]
                            finite_b2 = np.isfinite(xb2) & np.isfinite(yb2)
                            if finite_b2.sum() < 2:
                                continue
                            xb2_f, yb2_f = xb2[finite_b2], yb2[finite_b2]
                            order_b2 = np.argsort(xb2_f)
                            yb2_on_a = np.interp(
                                xa2_s,
                                xb2_f[order_b2],
                                yb2_f[order_b2],
                                left=np.nan,
                                right=np.nan,
                            )
                            delta2 = yb2_on_a - ya2_s
                            ax_delta_right.plot(
                                xa2_s,
                                delta2,
                                linewidth=1.1,
                                linestyle="--",
                                color=run["color"],
                                alpha=0.85,
                                label="_nolegend_",
                            )
                    ax_delta_right.set_ylabel(
                        plotter._format_waveform_channel_label(
                            f"Δ {ch_secondary}", secondary=True, show_style_hint=True
                        ),
                        fontsize=8.5,
                        fontweight="bold",
                        rotation=0,
                        ha="left",
                        va="center",
                    )
                    ax_delta_right.yaxis.set_label_coords(1.03, 0.5)
                    ax_delta_right.spines["top"].set_visible(False)
                    ax_delta_right.grid(False)
                    ax_delta_right.tick_params(axis="y", labelsize=8)
                ax_delta.axhline(0, linestyle="--", color="#4A4A4A", linewidth=0.9, alpha=0.75, zorder=2)
                ylim = ax_delta.get_ylim()
                yabs = max(abs(ylim[0]), abs(ylim[1]))
                if yabs > 0:
                    ax_delta.set_ylim(-yabs, yabs)
                if ax_delta_right is not None:
                    ylim_r = ax_delta_right.get_ylim()
                    yabs_r = max(abs(ylim_r[0]), abs(ylim_r[1]))
                    if yabs_r > 0:
                        ax_delta_right.set_ylim(-yabs_r, yabs_r)
                ax_delta.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))
                ax_delta.set_ylabel(
                    plotter._format_waveform_channel_label(
                        f"Δ {ch_primary}",
                        secondary=False,
                        show_style_hint=(ch_secondary is not None and ax_delta_right is not None),
                    ),
                    fontsize=8.5,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                )
                ax_delta.yaxis.set_label_coords(-0.035, 0.5)
                ax_delta.tick_params(axis="y", labelsize=8)
                plotter._apply_grid(ax_delta, which="major", axis="y")
                if idx < len(prepared_rows) - 1:
                    ax_delta.tick_params(labelbottom=False)
        axes[-1].set_xlabel(xlabel, fontweight="bold")
        axes[-1].tick_params(axis="x", labelsize=10)
        if x_limits is not None:
            xmin, xmax = x_limits
            if xmin is not None or xmax is not None:
                for ax in axes:
                    ax.set_xlim(left=xmin, right=xmax)
        else:
            if x_channel == "sLap":
                xmaxs = [xm for ax in axes for _, xm in [ax.get_xlim()] if xm > 0]
                if xmaxs:
                    xv = max(xmaxs)
                    for ax in axes:
                        ax.set_xlim(0, xv)
            for ax in axes:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=5, steps=[1, 2, 2.5, 5, 10]))
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                plotter._apply_grid(ax, which="both", axis="x")
        if annotate_at and annotate_row_data:
            xl_ann, xr_ann = axes[0].get_xlim()

            def _fmt_val(v):
                if abs(v) >= 1000:
                    return f"{v:.0f}"
                if abs(v) >= 100:
                    return f"{v:.1f}"
                return f"{v:.2f}"

            def _render_annotations(ax_ann, x_at, ann_items, x_frac):
                if not ann_items:
                    return
                place_left = x_frac > 0.75
                ha = "right" if place_left else "left"
                x_offset = -6 if place_left else 6
                ann_items.sort(key=lambda t: t[0])
                fig.canvas.draw_idle()
                trans = ax_ann.transData
                display_ys = [trans.transform((x_at, item[0]))[1] for item in ann_items]
                min_sep = 14
                adjusted = list(display_ys)
                for i in range(1, len(adjusted)):
                    gap = adjusted[i] - adjusted[i - 1]
                    if gap < min_sep:
                        adjusted[i] = adjusted[i - 1] + min_sep
                for i, (y_val, color, mstyle) in enumerate(ann_items):
                    msize = 4 if mstyle == "o" else 3.5
                    ax_ann.plot(
                        x_at,
                        y_val,
                        marker=mstyle,
                        markersize=msize,
                        color=color,
                        zorder=10,
                        alpha=0.9,
                    )
                    dy_pts = adjusted[i] - display_ys[i]
                    ax_ann.annotate(
                        _fmt_val(y_val),
                        xy=(x_at, y_val),
                        xytext=(x_offset, dy_pts),
                        textcoords="offset points",
                        ha=ha,
                        va="center",
                        fontsize=7.5,
                        color=color,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="white",
                            edgecolor=color,
                            linewidth=0.5,
                            alpha=0.85,
                        ),
                        zorder=11,
                    )

            for x_at in annotate_at:
                if not (xl_ann <= x_at <= xr_ann):
                    continue
                for ax in axes:
                    ax.axvline(
                        x_at,
                        color="#5E5E5E",
                        linestyle=":",
                        linewidth=0.9,
                        alpha=0.6,
                        zorder=2,
                    )
                x_frac = (x_at - xl_ann) / (xr_ann - xl_ann) if (xr_ann - xl_ann) > 0 else 0.5
                for row_idx, run_traces in annotate_row_data.items():
                    ax_idx_a = row_idx + sum(1 for i in range(row_idx) if show_delta[i]) if delta_active else row_idx
                    ax_row = axes[ax_idx_a]
                    ann_items = []
                    for run in plotter.runs:
                        rn = run["name"].lower()
                        if rn not in run_traces:
                            continue
                        x_arr, y_arr, y_sec_arr = run_traces[rn]
                        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
                        if finite.sum() < 2:
                            continue
                        xf, yf = x_arr[finite], y_arr[finite]
                        order = np.argsort(xf)
                        y_interp = np.interp(x_at, xf[order], yf[order], left=np.nan, right=np.nan)
                        if np.isfinite(y_interp):
                            ann_items.append((y_interp, run["color"], "o"))
                        if y_sec_arr is not None:
                            finite2 = np.isfinite(x_arr) & np.isfinite(y_sec_arr)
                            if finite2.sum() >= 2:
                                xf2, yf2 = x_arr[finite2], y_sec_arr[finite2]
                                order2 = np.argsort(xf2)
                                y_interp2 = np.interp(x_at, xf2[order2], yf2[order2], left=np.nan, right=np.nan)
                                if np.isfinite(y_interp2):
                                    ann_items.append((y_interp2, run["color"], "s"))
                    _render_annotations(ax_row, x_at, ann_items, x_frac)
                    if delta_active and show_delta[row_idx]:
                        ax_delta_a = axes[ax_idx_a + 1]
                        delta_ann_items = []
                        ref_data = run_traces.get(ref_run_name)
                        if ref_data is not None:
                            x_ref, y_ref, y_sec_ref = ref_data
                            finite_ref = np.isfinite(x_ref) & np.isfinite(y_ref)
                            if finite_ref.sum() >= 2:
                                xr_f, yr_f = x_ref[finite_ref], y_ref[finite_ref]
                                order_r = np.argsort(xr_f)
                                y_ref_at = np.interp(x_at, xr_f[order_r], yr_f[order_r], left=np.nan, right=np.nan)
                            else:
                                y_ref_at = np.nan
                            y_sec_ref_at = np.nan
                            if y_sec_ref is not None:
                                finite_ref2 = np.isfinite(x_ref) & np.isfinite(y_sec_ref)
                                if finite_ref2.sum() >= 2:
                                    xr2_f, yr2_f = x_ref[finite_ref2], y_sec_ref[finite_ref2]
                                    order_r2 = np.argsort(xr2_f)
                                    y_sec_ref_at = np.interp(
                                        x_at, xr2_f[order_r2], yr2_f[order_r2], left=np.nan, right=np.nan
                                    )
                            for run in plotter.runs:
                                rn = run["name"].lower()
                                if rn == ref_run_name or rn not in run_traces:
                                    continue
                                x_arr, y_arr, y_sec_arr = run_traces[rn]
                                finite_d = np.isfinite(x_arr) & np.isfinite(y_arr)
                                if finite_d.sum() >= 2 and np.isfinite(y_ref_at):
                                    xd_f, yd_f = x_arr[finite_d], y_arr[finite_d]
                                    order_d = np.argsort(xd_f)
                                    y_d_at = np.interp(x_at, xd_f[order_d], yd_f[order_d], left=np.nan, right=np.nan)
                                    if np.isfinite(y_d_at):
                                        delta_ann_items.append((y_d_at - y_ref_at, run["color"], "o"))
                                if y_sec_arr is not None and np.isfinite(y_sec_ref_at):
                                    finite_d2 = np.isfinite(x_arr) & np.isfinite(y_sec_arr)
                                    if finite_d2.sum() >= 2:
                                        xd2_f, yd2_f = x_arr[finite_d2], y_sec_arr[finite_d2]
                                        order_d2 = np.argsort(xd2_f)
                                        y_d2_at = np.interp(
                                            x_at, xd2_f[order_d2], yd2_f[order_d2], left=np.nan, right=np.nan
                                        )
                                        if np.isfinite(y_d2_at):
                                            delta_ann_items.append((y_d2_at - y_sec_ref_at, run["color"], "s"))
                        _render_annotations(ax_delta_a, x_at, delta_ann_items, x_frac)
        if markers:
            static_markers = [m for m in markers if m.x is not None]
            cond_markers = [m for m in markers if m.condition is not None]

            def _draw_line(ax_m, x_val, color, linestyle):
                ax_m.axvline(x_val, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.7, zorder=2)

            def _draw_static_label(ax_m, x_val, color, label):
                ax_m.text(
                    x_val,
                    1.01,
                    label,
                    transform=ax_m.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9),
                    zorder=12,
                )

            def _draw_inline_label(ax_m, x_val, color, label, y_frac):
                ax_m.annotate(
                    label,
                    xy=(x_val, y_frac),
                    xycoords=ax_m.get_xaxis_transform(),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.6, alpha=0.9),
                    zorder=12,
                )

            for m in static_markers:
                color = m.color or "#5E5E5E"
                rows_to_draw = [m.row] if m.row is not None and 0 <= m.row < len(axes) else list(range(len(axes)))
                for ridx in rows_to_draw:
                    _draw_line(axes[ridx], m.x, color, m.linestyle)
                if m.label and m.show_label:
                    _draw_static_label(axes[rows_to_draw[0]], m.x, color, m.label)
            if cond_markers:
                n_slots = max(len(cond_markers), 1)
                slot_step = 0.12 if n_slots > 1 else 0.0
                slot_y = [0.96 - i * slot_step for i in range(n_slots)]
                plotted_run_names = [r["name"] for r in plotter.runs if r["name"].lower() in plotted_runs]
                earliest_hit = [None] * len(cond_markers)
                marker_rows = [None] * len(cond_markers)
                for run_name in plotted_run_names:
                    run_df = plotter.run_data.get(run_name.lower())
                    if run_df is None or x_channel not in run_df.columns:
                        continue
                    run_color = next(
                        (r["color"] for r in plotter.runs if r["name"] == run_name),
                        "#5E5E5E",
                    )
                    for m_idx, m in enumerate(cond_markers):
                        x_hits = datafunctions.resolve_condition_marker(
                            m,
                            run_df,
                            x_channel,
                        )
                        if not x_hits:
                            continue
                        color = m.color or run_color
                        rows_to_draw = (
                            [m.row] if m.row is not None and 0 <= m.row < len(axes) else list(range(len(axes)))
                        )
                        marker_rows[m_idx] = rows_to_draw
                        for x_val in x_hits:
                            for ridx in rows_to_draw:
                                _draw_line(axes[ridx], x_val, color, m.linestyle)
                        if earliest_hit[m_idx] is None or x_hits[0] < earliest_hit[m_idx]:
                            earliest_hit[m_idx] = x_hits[0]
                for m_idx, m in enumerate(cond_markers):
                    if not m.label or not m.show_label or earliest_hit[m_idx] is None or marker_rows[m_idx] is None:
                        continue
                    label_color = m.color or "#3A3A3A"
                    _draw_inline_label(
                        axes[marker_rows[m_idx][0]],
                        earliest_hit[m_idx],
                        label_color,
                        m.label,
                        slot_y[m_idx],
                    )
        run_handles = [
            Line2D([0], [0], color=run["color"], linewidth=2.0)
            for run in plotter.runs
            if run["name"].lower() in plotted_runs
        ]
        run_labels = [run["name"].upper() for run in plotter.runs if run["name"].lower() in plotted_runs]
        plotter._add_waveform_figure_legend(fig, run_handles, run_labels, position=legend_position)
        if legend_position == "right":
            plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 0.88, 1.0))
        else:
            plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.95))
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
