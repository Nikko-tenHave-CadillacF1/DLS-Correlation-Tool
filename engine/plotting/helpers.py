"""Module-level helper functions extracted from :class:`DataPlotter`.

Prompt 10, phase 1 (2026-07): 17 helper methods lifted out of ``DataPlotter``
so the plot-generator mixins can be converted to module functions in a
follow-up phase without having to reach back through ``self.<method>`` to
DataPlotter. Function bodies are byte-identical to the originals aside from
the ``self`` argument being renamed to ``plotter``; the DataPlotter methods
now delegate one-line to the functions here, so every existing call site
keeps working unchanged.

Signature convention:
* Fully pure helpers keep their original signature (no plotter arg).
* Helpers that read plotter config (``PLOT_FONT``, ``GRID_STYLE``,
  ``plots_dir``, ``FILTER_SAMPLE_RATE``, ``plot_aspect_ratios``,
  ``PLOT_DEFINITIONS``, ``_INFO_CORNER_XY``/``_CORNER_TO_LOC``/
  ``_LOC_TO_CORNER``) or per-run state (``ctx.run_data``, ``ctx.psd_cache``,
  ``ctx.gated_data_cache``, ``ctx.run_sample_rates``) take ``plotter`` as
  the first positional argument.
* Helpers that call *other* DataPlotter methods that were NOT extracted this
  phase (e.g. ``_rank_info_corners``, ``_colorize_legend_labels``,
  ``_legend_corner``, ``_suggest_similar_channels``) still dispatch through
  ``plotter.<method>(...)``.
"""

from __future__ import annotations

import numpy as np

from .. import datafunctions
from ..logger import log

# ---------------------------------------------------------------------------
# Pure/static helpers
# ---------------------------------------------------------------------------


def _apply_2d_axis_limits(ax, axis_limits, *, log_scale_y=False, y_floor=1e-4):
    if not axis_limits:
        return False, False
    (xmin, xmax), (ymin, ymax) = axis_limits
    has_x = xmin is not None or xmax is not None
    has_y = ymin is not None or ymax is not None
    if has_x:
        ax.set_xlim(left=xmin, right=xmax)
    if has_y:
        if log_scale_y and ymin is not None:
            ymin = max(ymin, y_floor)
        ax.set_ylim(bottom=ymin, top=ymax)
    return has_x, has_y


def _draw_horizontal_reference_lines(ax, refs, *, label=True):
    if not refs:
        return
    y0, y1 = ax.get_ylim()
    new_y0, new_y1 = y0, y1
    for v in refs:
        if not np.isfinite(v):
            continue
        pad = (y1 - y0) * 0.05 if (y1 > y0) else 0.0
        new_y0 = min(new_y0, v - pad)
        new_y1 = max(new_y1, v + pad)
    if (new_y0, new_y1) != (y0, y1):
        ax.set_ylim(new_y0, new_y1)
    for v in refs:
        if not np.isfinite(v):
            continue
        ax.axhline(v, color="#4A4A4A", linestyle="--", linewidth=0.8, alpha=0.65, zorder=1)
        if label:
            ax.text(
                0.995,
                v,
                f" {v:g}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#333333",
            )


def _draw_static_markers(axes, markers, *, label_y=1.01, x_clip=True):
    if not markers:
        return
    try:
        ax_iter = list(axes)
    except TypeError:
        ax_iter = [axes]
    for ax in ax_iter:
        xl, xr = ax.get_xlim() if x_clip else (None, None)
        for m in markers:
            if m.condition is not None:
                continue
            if x_clip and not (xl <= m.x <= xr):
                continue
            color = m.color or "#5E5E5E"
            ax.axvline(m.x, color=color, linestyle=m.linestyle, linewidth=1.2, alpha=0.7, zorder=2)
            if m.label and m.show_label:
                ax.text(
                    m.x,
                    label_y,
                    m.label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9
                    ),
                    zorder=12,
                )


def _count_points_in_region(xs, ys, x0, x1, y0, y1, halign, valign, w_frac=0.18, h_frac=0.20):
    if xs.size == 0:
        return 0
    w = (x1 - x0) * w_frac
    h = (y1 - y0) * h_frac
    xa = 0.97 if halign == "right" else (0.50 if halign == "center" else 0.03)
    ya = 0.97 if valign == "top" else (0.50 if valign == "center" else 0.03)
    x_abs = x0 + xa * (x1 - x0)
    x_min = x_abs if halign == "left" else (x_abs - w if halign == "right" else x_abs - w / 2)
    x_max = x_min + w
    y_abs = y0 + ya * (y1 - y0)
    if valign == "top":
        y_min, y_max = y_abs - h, y_abs
    elif valign == "bottom":
        y_min, y_max = y_abs, y_abs + h
    else:
        y_min, y_max = y_abs - h / 2, y_abs + h / 2
    return int(((xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)).sum())


def _add_axis_edge_padding(ax, x_pad_ratio=0.02, y_pad_ratio=0.03):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmax > xmin:
        pad = (xmax - xmin) * x_pad_ratio
        ax.set_xlim(xmin - pad, xmax + pad)
    if ymax > ymin:
        pad = (ymax - ymin) * y_pad_ratio
        ax.set_ylim(ymin - pad, ymax + pad)


# ---------------------------------------------------------------------------
# Config-reading helpers (take ``plotter`` as first arg)
# ---------------------------------------------------------------------------


def _apply_grid(plotter, ax, which="both", axis="both"):
    if which in ("major", "both"):
        ax.grid(True, which="major", axis=axis, **plotter.GRID_STYLE["major"])
    if which in ("minor", "both"):
        ax.grid(True, which="minor", axis=axis, **plotter.GRID_STYLE["minor"])
    ax.set_axisbelow(True)


def _run_fs(plotter, run_name: str) -> float:
    """Sample rate for a run: detected per-run value, else the global default.

    After ``_preprocess_data`` the per-run rate equals ``RESAMPLE_RATE`` when
    resampling is enabled, otherwise the rate inferred from the run's time
    column. ``FILTER_SAMPLE_RATE`` is only used as a fallback before detection
    has run (e.g. inside ``_clean_data`` on the very first call).
    """
    pair = plotter.run_sample_rates.get(run_name)
    if pair and pair[0]:
        return float(pair[0])
    return float(plotter.FILTER_SAMPLE_RATE)


def _ensure_preprocessed(plotter):
    if not plotter._loaded:
        raise RuntimeError("Data has not been loaded.")
    if not plotter._preprocessed:
        raise RuntimeError("Data has not been preprocessed.")


def _format_missing_channel_hint(plotter, run_name, missing_channel):
    suggestions = plotter._suggest_similar_channels(run_name, missing_channel)
    if suggestions:
        return f"  Similar available: {', '.join(suggestions)}"
    return ""


def _get_plot_group(plotter, index):
    if not plotter.PLOT_DEFINITIONS or len(plotter.PLOT_DEFINITIONS) <= index:
        return []
    return plotter.PLOT_DEFINITIONS[index] or []


def _sanitize_plot_filename(plotter, prefix, plot_name, suffix=""):
    safe = (
        plot_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
    )
    subdir = plotter.plots_dir / prefix
    try:
        subdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return f"{prefix}/{prefix}_{safe}{suffix}.png"


def _resolve_plot_figsize(plotter, filename, default_size, *, min_height=None):
    w0, h0 = default_size
    target_aspect = plotter.plot_aspect_ratios.get(filename)
    if isinstance(target_aspect, (list, tuple)):
        target_aspect = sum(target_aspect) / len(target_aspect)
    if target_aspect is None:
        w, h = w0, h0
    else:
        h = h0
        w = h * target_aspect
    if min_height:
        h = max(h, min_height)
        if target_aspect:
            w = h * target_aspect
        elif h > h0:
            # Height was forced above the default; scale width to preserve
            # the default aspect ratio so all waveforms look consistent.
            w = h * (w0 / h0)
    return (w, h)


# ---------------------------------------------------------------------------
# State-reading helpers (read/write plotter.ctx)
# ---------------------------------------------------------------------------


def _get_filtered_run_dataframe(plotter, run_name, gate_spec=None):
    if gate_spec is None:
        return plotter.run_data.get(run_name)
    cache_key = (run_name, repr(gate_spec))
    cached = plotter._gated_data_cache.get(cache_key)
    if cached is not None:
        return cached
    df = plotter.run_data.get(run_name)
    if df is None:
        return None
    filtered = datafunctions.apply_gate_to_dataframe(df, gate_spec)
    plotter._gated_data_cache[cache_key] = filtered
    return filtered


def _cached_psd_with_segments(plotter, run_name, channel, nperseg, gate_spec=None):
    gate_key = repr(gate_spec) if gate_spec is not None else None
    key = (run_name, channel, nperseg, gate_key)
    cached = plotter._psd_cache.get(key)
    if cached is not None:
        return cached
    df = plotter.run_data.get(run_name)
    if df is None or channel not in df.columns:
        return None, None, 0
    signal = np.asarray(df[channel], dtype=float)
    rate = plotter.run_sample_rates.get(run_name, (plotter.FILTER_SAMPLE_RATE, "default"))[0]
    if gate_spec is None:
        finite_n = int(np.isfinite(signal).sum())
        if finite_n < nperseg and finite_n >= 8:
            effective = min(nperseg, finite_n)
            if effective < max(64, nperseg // 4):
                log.warning(
                    "PSD '%s'/'%s': only %d finite samples — nperseg capped from %d to %d "
                    "(coarse frequency resolution, low averaging).",
                    run_name,
                    channel,
                    finite_n,
                    nperseg,
                    effective,
                )
        freq, power = datafunctions.calculate_psd(signal, rate, nperseg=nperseg)
        if freq is not None:
            eff_n = min(nperseg, int(np.isfinite(signal).sum()))
            step = max(1, eff_n // 2)
            n_segs = max(1, 1 + (int(np.isfinite(signal).sum()) - eff_n) // step)
        else:
            n_segs = 0
    else:
        try:
            mask = datafunctions.compute_gate_mask(df, gate_spec).to_numpy()
        except Exception as exc:
            log.warning(
                "PSD '%s'/'%s': gate evaluation failed (%s). Skipping.",
                run_name,
                channel,
                exc,
            )
            plotter._psd_cache[key] = (None, None, 0)
            return None, None, 0
        freq, power, n_segs = datafunctions.calculate_segmented_psd(
            signal,
            mask,
            rate,
            nperseg=nperseg,
        )
        if freq is None:
            log.warning(
                "PSD '%s'/'%s': no gated segment >= nperseg (%d). Skipping.",
                run_name,
                channel,
                nperseg,
            )
    plotter._psd_cache[key] = (freq, power, n_segs)
    return freq, power, n_segs


# ---------------------------------------------------------------------------
# Legend / info-box helpers (read config + call unextracted plotter methods)
# ---------------------------------------------------------------------------


def _add_standard_legend(
    plotter, ax, handles=None, labels=None, loc="best", bbox_to_anchor=None, ncol=1, avoid_corner=None
):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if avoid_corner is not None and bbox_to_anchor is None:
        ranked = [c for c in plotter._rank_info_corners(ax) if c != avoid_corner]
        corner = ranked[0] if ranked else None
        loc = plotter._CORNER_TO_LOC.get(corner, "best") if corner else "best"
    legend_kwargs = dict(
        fancybox=True,
        framealpha=0.92,
        edgecolor="#3C3C3C",
        borderpad=0.55,
        handlelength=1.8,
        ncol=ncol,
        prop={"family": plotter.PLOT_FONT["family"], "weight": "bold", "size": plotter.PLOT_FONT["legend_size"]},
    )
    corner = plotter._LOC_TO_CORNER.get(loc) if isinstance(loc, str) else None
    if corner is not None and bbox_to_anchor is None:
        legend_kwargs["bbox_to_anchor"] = plotter._INFO_CORNER_XY[corner]
        legend_kwargs["bbox_transform"] = ax.transAxes
        legend_kwargs["borderaxespad"] = 0
    else:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = ax.legend(handles, labels, loc=loc, **legend_kwargs)
    legend.get_frame().set_linewidth(1.4)
    legend.set_zorder(10)
    plotter._colorize_legend_labels(legend)
    return legend


def _add_waveform_figure_legend(plotter, fig, handles, labels, position="top"):
    if not handles:
        return None
    if position == "right":
        legend = fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            fancybox=True,
            framealpha=0.92,
            edgecolor="#3C3C3C",
            borderpad=0.4,
            handlelength=1.8,
            prop={"family": plotter.PLOT_FONT["family"], "weight": "bold", "size": plotter.PLOT_FONT["legend_size"]},
        )
    else:
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=max(1, min(len(handles), 5)),
            fancybox=True,
            framealpha=0.92,
            edgecolor="#3C3C3C",
            borderpad=0.3,
            handlelength=1.8,
            prop={"family": plotter.PLOT_FONT["family"], "weight": "bold", "size": plotter.PLOT_FONT["legend_size"]},
        )
    legend.get_frame().set_linewidth(1.4)
    legend.set_zorder(10)
    plotter._colorize_legend_labels(legend)
    return legend


def _display_gate_info(plotter, ax, text, legend=None, trend_anchor=None):
    occupied = set()
    if trend_anchor is not None:
        _, trend_halign, trend_valign, _ = trend_anchor
        occupied.add((trend_halign, trend_valign))
    legend_corner = plotter._legend_corner(legend)
    if legend_corner is not None:
        occupied.add(legend_corner)
    ranked = plotter._rank_info_corners(ax)
    free = [c for c in ranked if c not in occupied]
    halign, valign = free[0] if free else ranked[0]
    x_anchor, y_anchor = plotter._INFO_CORNER_XY[(halign, valign)]
    ax.text(
        x_anchor,
        y_anchor,
        text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment=valign,
        horizontalalignment=halign,
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            alpha=0.92,
            edgecolor="#3C3C3C",
            linewidth=1.4,
        ),
        color="#1A1A1A",
        fontweight="bold",
        family=plotter.PLOT_FONT["family"],
    )
