"""Modal evolution plots — visualise Lorentz/4DOF fit results across runs.

These plots are produced after `run_workflow()` returns; they consume
`plotter.modal_results` (populated by `_run_modal_fits()`) and render line
plots with confidence bands (sigma error envelope) plus matching bar charts
with error bars.

Use from a Run_*.py:

    from engine.modal_plots import plot_modal_evolution, get_plotter
    plot_modal_evolution(get_plotter(), modes=("Heave", "Pitch", "Roll", "Warp"))

Or capture the plotter via `run_from_config` returning it (currently the
public API runs a full pipeline and does not return the plotter, so the
recommended pattern is to call this directly from the script after the
workflow has run via a slim helper that re-creates the plotter).
"""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .logger import log


_FIG_FONT = {"family": "DejaVu Sans", "size": 11}
_INK = "#1A1A1A"
_PARAM_LABELS = {
    "f0":         r"$f_0$ [Hz]",
    "zeta":       r"$\zeta$ [-]",
    "amp_front":  "Front amplitude [norm.]",
    "amp_rear":   "Rear amplitude [norm.]",
}
# Mode-agnostic params iterated by the generic line/bar generators.
_PARAMS_ALL = ("f0", "zeta")
# Axle series rendered together on per-mode amplitude plots.
_AMP_AXLES = (
    ("amp_front", "Front", "#1f77b4"),
    ("amp_rear",  "Rear",  "#d62728"),
)
_AMP_AXIS_LABEL = "Amplitude [norm.]"
# Hatch patterns used to distinguish groups (cars) in compare-mode bar plots
# when no per-group colour mapping is in effect (legacy fallback).
_GROUP_HATCHES = ("", "///", "...", "xxx", "\\\\\\")
# In compare-mode amplitude plots, group is encoded by colour and axle by
# hatch (bars) or linestyle (lines).
_AXLE_HATCHES = {"amp_front": "", "amp_rear": "///"}
_AXLE_LINESTYLES = {"amp_front": "-", "amp_rear": "--"}
# Named-group → colour presets, applied case-insensitively. Anything else
# falls back to the palette below indexed by group order.
_GROUP_COLOR_PRESETS = {
    "RED":    "#d62728",
    "BLUE":   "#1f77b4",
    "GREEN":  "#2ca02c",
    "ORANGE": "#ff7f0e",
    "PURPLE": "#9467bd",
    "YELLOW": "#bcbd22",
    "CYAN":   "#17becf",
    "PINK":   "#e377c2",
}
_GROUP_FALLBACK_PALETTE = (
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
)


def _resolve_group_colors(groups, runs):
    """Return `{gkey: "#hex"}` for compare-mode rendering.

    Priority: (1) named-group preset (case-insensitive on str(gkey)), so a
    group literally called "RED" always paints red regardless of auto-
    assigned tab10 run colours; (2) palette fallback by group order.
    """
    out: dict = {}
    for gi, gkey in enumerate(groups.keys()):
        chosen = None
        if gkey is not None:
            chosen = _GROUP_COLOR_PRESETS.get(str(gkey).upper())
        if chosen is None:
            chosen = _GROUP_FALLBACK_PALETTE[gi % len(_GROUP_FALLBACK_PALETTE)]
        out[gkey] = chosen
    return out


def _fmt_value(param: str, value: float) -> str:
    if not np.isfinite(value):
        return ""
    if param == "f0":
        return f"{value:.2f}"
    if param == "zeta":
        return f"{value:.3f}"
    # Amplitudes can span many decades — use compact scientific form.
    if value == 0:
        return "0"
    a = abs(value)
    if 1e-2 <= a < 1e3:
        return f"{value:.2f}"
    return f"{value:.1e}"


def _modal_records(modal_results: dict, runs: list) -> list[dict]:
    """Flatten modal_results into per-(run, mode) records preserving runs order."""

    def _as_list(v):
        if v is None:
            return []
        try:
            return list(v)
        except TypeError:
            return [v]

    rows: list[dict] = []
    for run in runs:
        name = run["name"]
        key = name.lower()
        res = modal_results.get(key)
        if not res:
            continue
        labels = _as_list(res.get("mode_labels"))
        fn = _as_list(res.get("fn"))
        zeta = _as_list(res.get("zeta"))
        sigma_fn_raw = res.get("sigma_fn")
        sigma_zeta_raw = res.get("sigma_zeta")
        sigma_fn = _as_list(sigma_fn_raw) if sigma_fn_raw is not None else None
        sigma_zeta = _as_list(sigma_zeta_raw) if sigma_zeta_raw is not None else None
        amp_f_raw = res.get("amp_front")
        amp_r_raw = res.get("amp_rear")
        amp_f = _as_list(amp_f_raw) if amp_f_raw is not None else None
        amp_r = _as_list(amp_r_raw) if amp_r_raw is not None else None
        sig_amp_f_raw = res.get("sigma_amp_front")
        sig_amp_r_raw = res.get("sigma_amp_rear")
        sig_amp_f = _as_list(sig_amp_f_raw) if sig_amp_f_raw is not None else None
        sig_amp_r = _as_list(sig_amp_r_raw) if sig_amp_r_raw is not None else None
        for i, m in enumerate(labels):
            rows.append({
                "run": name,
                "color": run.get("color", "#1f77b4"),
                "group": run.get("group"),
                "session_label": run.get("session_label") or name,
                "mode": str(m),
                "f0": float(fn[i]) if i < len(fn) else float("nan"),
                "zeta": float(zeta[i]) if i < len(zeta) else float("nan"),
                "f0_sigma": (float(sigma_fn[i]) if sigma_fn is not None
                             and i < len(sigma_fn) else float("nan")),
                "zeta_sigma": (float(sigma_zeta[i]) if sigma_zeta is not None
                               and i < len(sigma_zeta) else float("nan")),
                "amp_front": (float(amp_f[i]) if amp_f is not None
                              and i < len(amp_f) else float("nan")),
                "amp_rear": (float(amp_r[i]) if amp_r is not None
                             and i < len(amp_r) else float("nan")),
                "amp_front_sigma": (float(sig_amp_f[i]) if sig_amp_f is not None
                                    and i < len(sig_amp_f) else float("nan")),
                "amp_rear_sigma": (float(sig_amp_r[i]) if sig_amp_r is not None
                                   and i < len(sig_amp_r) else float("nan")),
            })
    return rows


def _mode_color(mode: str) -> str:
    return {
        "Heave": "#1f77b4", "Pitch": "#ff7f0e",
        "Roll":  "#2ca02c", "Warp":  "#d62728",
    }.get(mode, "#7f7f7f")


def _ensure_output_dir(plotter, subdir: str = "modal") -> Path:
    out = Path(plotter.plots_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_modal_evolution(
    plotter,
    *,
    modes: tuple[str, ...] = ("Heave", "Pitch", "Roll", "Warp"),
    group_by: str | None = None,
    compare_by=None,
    include_consolidated: bool = True,
    output_subdir: str = "modal",
    name_suffix: str = "",
    line_ci: bool = True,
    bars: bool = True,
) -> list[Path]:
    """Render session-over-session modal evolution plots.

    Parameters
    ----------
    plotter : DataPlotter
        After `run_from_config` has finished; provides `runs`, `modal_results`,
        `plots_dir`, and `output_dpi`.
    modes : tuple of mode labels to include (must match `mode_labels` in fit).
    group_by : optional run-dict key (e.g. "group" or "type") — when set, runs
        are split into separate series by that key and shown with distinct
        line styles. Useful for RED-vs-BLUE car or event-to-event compare.
    compare_by : optional run-name → session-token extractor. When set ALL
        groups are aligned on a shared x-axis: runs that map to the same
        session token share an x position so groups (cars) can be compared
        side-by-side per session. Accepts either a callable `(name) -> str`
        or the string preset `"session"` (= last `_`-separated token of the
        run name).
    include_consolidated : if False, drop runs whose name contains
        "consolidated" from the per-session plots.
    name_suffix : appended to output filenames (e.g. "heave_pitch") so the
        function can be called multiple times with disjoint mode subsets
        without filename collisions. Empty → legacy filenames.
    line_ci : draw the line+CI-band variant.
    bars : draw the bar+errorbar variant.

    Returns the list of saved figure paths.
    """
    if not getattr(plotter, "modal_results", None):
        log.info("plot_modal_evolution: no modal_results on plotter; nothing to do.")
        return []
    runs = [
        r for r in plotter.runs
        if r["name"].lower() in plotter.modal_results
        and (include_consolidated or "consolidated" not in r["name"].lower())
    ]
    if not runs:
        log.info("plot_modal_evolution: no qualifying runs.")
        return []
    rows = _modal_records(plotter.modal_results, runs)
    if not rows:
        log.info("plot_modal_evolution: no records to plot.")
        return []
    out_dir = _ensure_output_dir(plotter, output_subdir)
    saved: list[Path] = []
    dpi = getattr(plotter, "output_dpi", 300)
    # ── group runs (preserving run order) ───────────────────────────────────
    groups: "OrderedDict[str | None, list[str]]" = OrderedDict()
    for r in runs:
        gkey = r.get(group_by) if group_by else None
        groups.setdefault(gkey, []).append(r["name"])
    # ── optional shared-x layout for group comparison ───────────────────────
    compare_layout = _build_compare_layout(runs, compare_by)
    group_colors = _resolve_group_colors(groups, runs) if compare_layout else None
    if line_ci:
        for param in _PARAMS_ALL:
            saved.extend(_line_ci_figure(
                rows, modes, param, groups, out_dir, dpi, name_suffix,
                compare_layout, group_colors,
            ))
        for mode in modes:
            saved.extend(_line_ci_amp_figure(
                rows, mode, groups, out_dir, dpi, compare_layout, group_colors,
            ))
    if bars:
        for param in _PARAMS_ALL:
            saved.extend(_bar_figure(
                rows, modes, param, groups, out_dir, dpi, name_suffix,
                compare_layout, group_colors,
            ))
        for mode in modes:
            saved.extend(_bar_amp_figure(
                rows, mode, groups, out_dir, dpi, compare_layout, group_colors,
            ))
    return saved


def _build_compare_layout(runs, compare_by):
    """Return `(session_order, run_session)` if compare_by is set, else None.

    `session_order` is the ordered list of unique session tokens (preserving
    first appearance). `run_session` maps each run name to its token.
    """
    if compare_by is None:
        return None
    if isinstance(compare_by, str):
        if compare_by == "session":
            def extract(n):
                return str(n).split("_")[-1]
        else:
            token = compare_by
            def extract(n):  # noqa: E306 — closure on plain string is not a preset
                return token
    elif callable(compare_by):
        extract = compare_by
    else:
        return None
    session_order: list[str] = []
    run_session: dict[str, str] = {}
    for r in runs:
        n = r["name"]
        tok = str(extract(n))
        run_session[n] = tok
        if tok not in session_order:
            session_order.append(tok)
    return session_order, run_session


def _line_ci_figure(rows, modes, param, groups, out_dir, dpi, name_suffix="",
                    compare_layout=None, group_colors=None) -> list[Path]:
    sigma_key = f"{param}_sigma"
    label_text = _PARAM_LABELS[param]
    # Build x positions per (group, run) and label list
    x_pos: list[float] = []
    x_labels: list[str] = []
    run_to_x: dict[tuple, float] = {}
    if compare_layout is not None:
        session_order, run_session = compare_layout
        x_pos = list(range(len(session_order)))
        x_labels = list(session_order)
        session_x = {tok: i for i, tok in enumerate(session_order)}
        for gkey, run_names in groups.items():
            for rn in run_names:
                tok = run_session.get(rn)
                if tok is not None:
                    run_to_x[(gkey, rn)] = float(session_x[tok])
        n_x = len(x_pos)
    else:
        cursor = 0
        for gkey, run_names in groups.items():
            for rn in run_names:
                x_pos.append(cursor)
                x_labels.append(_short_label(rn))
                run_to_x[(gkey, rn)] = cursor
                cursor += 1
            cursor += 0.5  # small visual gap between groups
        n_x = sum(len(g) for g in groups.values())
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * n_x + 3), 5.2))
    linestyles = ["-", "--", ":", "-."]
    n_modes = max(len(modes), 1)
    n_groups = max(len(groups), 1)
    label_gap_pt = 11
    base_dy_pt = 6
    # Collect labels first, then anchor all labels at the same x to a common
    # top (max y+err across series at that x) and stack upward by series.
    label_records: list[tuple[float, float, int, str, str]] = []
    for mi, mode in enumerate(modes):
        for gi, (gkey, run_names) in enumerate(groups.items()):
            # Compare-mode: colour identifies the GROUP (car); linestyle
            # remains solid because the colour already carries the identity.
            # Non-compare: keep legacy mode-colour + per-group linestyle.
            if compare_layout is not None and group_colors is not None:
                color = group_colors.get(gkey, _mode_color(mode))
                ls = "-"
            else:
                color = _mode_color(mode)
                ls = linestyles[gi % len(linestyles)]
            xs = []
            ys = []
            errs = []
            for rn in run_names:
                rec = next(
                    (r for r in rows if r["run"] == rn and r["mode"].lower() == mode.lower()),
                    None,
                )
                if rec is None or not np.isfinite(rec[param]):
                    continue
                if (gkey, rn) not in run_to_x:
                    continue
                xs.append(run_to_x[(gkey, rn)])
                ys.append(rec[param])
                err = rec.get(sigma_key)
                errs.append(err if (err is not None and np.isfinite(err)) else 0.0)
            if not xs:
                continue
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            errs = np.asarray(errs)
            if compare_layout is not None and group_colors is not None:
                glabel = group_by_label(gkey) or f"{mode}"
            else:
                glabel = f"{mode}" + (f" [{gkey}]" if group_by_label(gkey) else "")
            ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                    linewidth=2.0, markersize=5.5, label=glabel)
            if np.any(errs > 0):
                ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.15)
                ax.errorbar(xs, ys, yerr=errs, fmt="none",
                            ecolor=color, elinewidth=1.0,
                            capsize=3, capthick=1.0, alpha=0.55)
            stack_idx = mi * n_groups + gi
            for x, y, e in zip(xs, ys, errs):
                label_records.append((
                    float(x), float(y) + float(e), stack_idx, float(y),
                    color, _fmt_value(param, float(y)),
                ))
    # Anchor every label at the SAME top per x then stack upward by VALUE
    # (largest at the top of the stack) so the visual order matches magnitude.
    tops_by_x: dict[float, float] = {}
    by_x: dict[float, list] = {}
    for rec in label_records:
        xc, top = rec[0], rec[1]
        tops_by_x[xc] = max(tops_by_x.get(xc, top), top)
        by_x.setdefault(xc, []).append(rec)
    for xc, recs in by_x.items():
        recs_sorted = sorted(recs, key=lambda r: r[3])  # ascending by y
        for slot, (_xc, _top, _orig_idx, _y, color, txt) in enumerate(recs_sorted):
            if not txt:
                continue
            ax.annotate(
                txt,
                xy=(xc, tops_by_x[xc]),
                xytext=(0, base_dy_pt + slot * label_gap_pt),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold",
                clip_on=False,
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(label_text, fontweight="bold")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=":")
    ax.legend(loc="best", ncol=2, fontsize=9, framealpha=0.9,
              handlelength=3.2, handletextpad=0.6)
    title_suffix = f" — {modes[0]}" if len(modes) == 1 else ""
    ax.set_title(f"Modal evolution — {label_text}{title_suffix}",
                 fontweight="bold", color=_INK)
    # Y-headroom so stacked labels stay inside the axes.
    y_lo, y_hi = ax.get_ylim()
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        head = 0.08 + 0.05 * (n_modes * n_groups)
        ax.set_ylim(y_lo, y_hi + head * (y_hi - y_lo))
    fig.tight_layout()
    fname = f"modal_evolution_line_{param}{('_' + name_suffix) if name_suffix else ''}.png"
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("Modal evolution line plot saved: %s", path.name)
    return [path]


def _bar_figure(rows, modes, param, groups, out_dir, dpi, name_suffix="",
                compare_layout=None, group_colors=None) -> list[Path]:
    sigma_key = f"{param}_sigma"
    label_text = _PARAM_LABELS[param]
    n_modes = len(modes)
    n_groups = max(len(groups), 1)
    if compare_layout is not None:
        # x-axis = sessions; one bar per (mode, group) per session
        session_order, run_session = compare_layout
        x_base = np.arange(len(session_order))
        n_x = len(session_order)
        n_series = n_modes * n_groups
        # Resolve a record for (mode, group, session) via the run that has
        # that group + session token.
        run_lookup: dict[tuple, str] = {}
        for gkey, run_names in groups.items():
            for rn in run_names:
                tok = run_session.get(rn)
                if tok is not None:
                    run_lookup[(gkey, tok)] = rn
        x_labels = list(session_order)
    else:
        all_runs = [rn for run_names in groups.values() for rn in run_names]
        n_x = len(all_runs)
        x_base = np.arange(n_x)
        n_series = n_modes
        x_labels = [_short_label(rn) for rn in all_runs]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n_x * n_series + 3), 5.2))
    bar_w = 0.8 / max(n_series, 1)
    many_bars = n_x * n_series >= 8
    label_rotation = 90 if many_bars else 0
    label_fontsize = 8 if many_bars else 9
    # (x_center, top, err, color) for label placement after y-headroom is known
    label_pts: list[tuple[float, float, float, str]] = []
    if compare_layout is not None:
        session_order, run_session = compare_layout
        series_idx = 0
        for mi, mode in enumerate(modes):
            for gi, (gkey, _run_names) in enumerate(groups.items()):
                # Compare-mode: colour = group (car); no hatch (colour
                # already carries identity). Fall back to mode-colour+hatch
                # if no group_colors were resolved.
                if group_colors is not None:
                    fill = group_colors.get(gkey, _mode_color(mode))
                    hatch = ""
                else:
                    fill = _mode_color(mode)
                    hatch = _GROUP_HATCHES[gi % len(_GROUP_HATCHES)]
                ys = []
                errs = []
                for tok in session_order:
                    rn = run_lookup.get((gkey, tok))
                    rec = None
                    if rn is not None:
                        rec = next(
                            (r for r in rows
                             if r["run"] == rn and r["mode"].lower() == mode.lower()),
                            None,
                        )
                    ys.append(rec[param] if (rec and np.isfinite(rec[param])) else np.nan)
                    err = rec.get(sigma_key) if rec else None
                    errs.append(err if (err is not None and np.isfinite(err)) else np.nan)
                xs = x_base + (series_idx - (n_series - 1) / 2) * bar_w
                label = f"{mode} [{gkey}]" if n_modes > 1 else str(gkey)
                ax.bar(xs, ys, width=bar_w, color=fill, label=label,
                       edgecolor="white", linewidth=0.6, alpha=0.9,
                       hatch=hatch)
                finite_err = [e if np.isfinite(e) else 0.0 for e in errs]
                if any(e > 0 for e in finite_err):
                    ax.errorbar(xs, ys, yerr=finite_err, fmt="none",
                                ecolor=_INK, elinewidth=1.2,
                                capsize=3, capthick=1.0, alpha=0.85)
                for xc, yt, e in zip(xs, ys, finite_err):
                    if np.isfinite(yt):
                        label_pts.append((float(xc), float(yt), float(e), fill))
                series_idx += 1
    else:
        all_runs = [rn for run_names in groups.values() for rn in run_names]
        for mi, mode in enumerate(modes):
            color = _mode_color(mode)
            ys = []
            errs = []
            for rn in all_runs:
                rec = next(
                    (r for r in rows if r["run"] == rn and r["mode"].lower() == mode.lower()),
                    None,
                )
                ys.append(rec[param] if (rec and np.isfinite(rec[param])) else np.nan)
                err = rec.get(sigma_key) if rec else None
                errs.append(err if (err is not None and np.isfinite(err)) else np.nan)
            xs = x_base + (mi - (n_modes - 1) / 2) * bar_w
            ax.bar(xs, ys, width=bar_w, color=color, label=mode,
                   edgecolor="white", linewidth=0.6, alpha=0.9)
            finite_err = [e if np.isfinite(e) else 0.0 for e in errs]
            if any(e > 0 for e in finite_err):
                ax.errorbar(xs, ys, yerr=finite_err, fmt="none",
                            ecolor=_INK, elinewidth=1.2,
                            capsize=3, capthick=1.0, alpha=0.85)
            for xc, yt, e in zip(xs, ys, finite_err):
                if np.isfinite(yt):
                    label_pts.append((float(xc), float(yt), float(e), color))
    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(label_text, fontweight="bold")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=":")
    ax.legend(loc="best", ncol=min(n_series, 4), fontsize=9, framealpha=0.9,
              handlelength=2.4, handleheight=1.4, handletextpad=0.6)
    title_suffix = f" — {modes[0]}" if len(modes) == 1 else ""
    ax.set_title(f"Modal parameters per session — {label_text}{title_suffix}",
                 fontweight="bold", color=_INK)
    # Y-headroom: ensure the highest label clears the top of the axes.
    y_lo, y_hi = ax.get_ylim()
    if label_pts and np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        max_top = max(t + e for _, t, e, _ in label_pts)
        if max_top > y_hi:
            y_hi = max_top
        head = 0.20 if label_rotation == 90 else 0.12
        ax.set_ylim(y_lo, y_hi + head * (y_hi - y_lo))
    # Value readouts above each bar's errorbar tip.
    pad_pt = 3
    for xc, yt, e, color in label_pts:
        label = _fmt_value(param, yt)
        if not label:
            continue
        ax.annotate(
            label,
            xy=(xc, yt + e),
            xytext=(0, pad_pt),
            textcoords="offset points",
            ha="center", va="bottom",
            rotation=label_rotation, fontsize=label_fontsize,
            color=color, fontweight="bold",
            clip_on=False,
        )
    fig.tight_layout()
    fname = f"modal_evolution_bar_{param}{('_' + name_suffix) if name_suffix else ''}.png"
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("Modal evolution bar plot saved: %s", path.name)
    return [path]


def _short_label(run_name: str) -> str:
    """Compact run label for x-tick rendering."""
    s = str(run_name)
    if len(s) <= 22:
        return s
    return s[:10] + "…" + s[-10:]


def group_by_label(gkey) -> str:
    if gkey is None:
        return ""
    return str(gkey)


def _line_ci_amp_figure(rows, mode, groups, out_dir, dpi,
                        compare_layout=None, group_colors=None) -> list[Path]:
    """Per-mode line+CI plot with front and rear amplitudes overlaid."""
    mode_l = mode.lower()
    x_pos: list[float] = []
    x_labels: list[str] = []
    run_to_x: dict[tuple, float] = {}
    if compare_layout is not None:
        session_order, run_session = compare_layout
        x_pos = list(range(len(session_order)))
        x_labels = list(session_order)
        session_x = {tok: i for i, tok in enumerate(session_order)}
        for gkey, run_names in groups.items():
            for rn in run_names:
                tok = run_session.get(rn)
                if tok is not None:
                    run_to_x[(gkey, rn)] = float(session_x[tok])
        n_x = len(x_pos)
    else:
        cursor = 0
        for gkey, run_names in groups.items():
            for rn in run_names:
                x_pos.append(cursor)
                x_labels.append(_short_label(rn))
                run_to_x[(gkey, rn)] = cursor
                cursor += 1
            cursor += 0.5
        n_x = sum(len(g) for g in groups.values())
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * n_x + 3), 5.2))
    linestyles = ["-", "--", ":", "-."]
    n_groups = max(len(groups), 1)
    label_gap_pt = 11
    base_dy_pt = 6
    # Collect labels first, then anchor all labels at the same x to a common
    # top (max y+err across series at that x) and stack upward by series.
    label_records: list[tuple[float, float, int, str, str]] = []
    for ai, (param, axle_label, axle_color) in enumerate(_AMP_AXLES):
        sigma_key = f"{param}_sigma"
        for gi, (gkey, run_names) in enumerate(groups.items()):
            # Compare-mode: colour = group (car); linestyle = axle (front/rear).
            # Non-compare: colour = axle; linestyle = group (legacy).
            if compare_layout is not None and group_colors is not None:
                color = group_colors.get(gkey, axle_color)
                ls = _AXLE_LINESTYLES.get(param, "-")
                glabel = f"{gkey} {axle_label}" if group_by_label(gkey) else axle_label
            else:
                color = axle_color
                ls = linestyles[gi % len(linestyles)]
                glabel = f"{axle_label}" + (f" [{gkey}]" if group_by_label(gkey) else "")
            xs = []
            ys = []
            errs = []
            for rn in run_names:
                rec = next(
                    (r for r in rows if r["run"] == rn and r["mode"].lower() == mode_l),
                    None,
                )
                if rec is None or not np.isfinite(rec.get(param, np.nan)):
                    continue
                if (gkey, rn) not in run_to_x:
                    continue
                xs.append(run_to_x[(gkey, rn)])
                ys.append(rec[param])
                err = rec.get(sigma_key)
                errs.append(err if (err is not None and np.isfinite(err)) else 0.0)
            if not xs:
                continue
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            errs = np.asarray(errs)
            ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                    linewidth=2.0, markersize=5.5, label=glabel)
            if np.any(errs > 0):
                ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.15)
                ax.errorbar(xs, ys, yerr=errs, fmt="none",
                            ecolor=color, elinewidth=1.0,
                            capsize=3, capthick=1.0, alpha=0.55)
            stack_idx = ai * n_groups + gi
            for x, y, e in zip(xs, ys, errs):
                label_records.append((
                    float(x), float(y) + float(e), stack_idx, float(y),
                    color, _fmt_value(param, float(y)),
                ))
    tops_by_x: dict[float, float] = {}
    by_x: dict[float, list] = {}
    for rec in label_records:
        xc, top = rec[0], rec[1]
        tops_by_x[xc] = max(tops_by_x.get(xc, top), top)
        by_x.setdefault(xc, []).append(rec)
    for xc, recs in by_x.items():
        recs_sorted = sorted(recs, key=lambda r: r[3])  # ascending by y
        for slot, (_xc, _top, _orig_idx, _y, color, txt) in enumerate(recs_sorted):
            if not txt:
                continue
            ax.annotate(
                txt,
                xy=(xc, tops_by_x[xc]),
                xytext=(0, base_dy_pt + slot * label_gap_pt),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold",
                clip_on=False,
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(_AMP_AXIS_LABEL, fontweight="bold")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=":")
    ax.legend(loc="best", ncol=2, fontsize=9, framealpha=0.9,
              handlelength=3.2, handletextpad=0.6)
    ax.set_title(f"Modal evolution — {mode} amplitude (front vs rear)",
                 fontweight="bold", color=_INK)
    y_lo, y_hi = ax.get_ylim()
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        head = 0.10 + 0.05 * (len(_AMP_AXLES) * n_groups)
        ax.set_ylim(y_lo, y_hi + head * (y_hi - y_lo))
    fig.tight_layout()
    fname = f"modal_evolution_line_amp_{mode_l}.png"
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("Modal evolution line plot saved: %s", path.name)
    return [path]


def _bar_amp_figure(rows, mode, groups, out_dir, dpi,
                    compare_layout=None, group_colors=None) -> list[Path]:
    """Per-mode bar plot with front and rear amplitudes side-by-side per run."""
    mode_l = mode.lower()
    n_axles = len(_AMP_AXLES)
    n_groups = max(len(groups), 1)
    if compare_layout is not None:
        session_order, run_session = compare_layout
        n_x = len(session_order)
        x_base = np.arange(n_x)
        n_series = n_axles * n_groups
        x_labels = list(session_order)
        run_lookup: dict[tuple, str] = {}
        for gkey, run_names in groups.items():
            for rn in run_names:
                tok = run_session.get(rn)
                if tok is not None:
                    run_lookup[(gkey, tok)] = rn
    else:
        all_runs = [rn for run_names in groups.values() for rn in run_names]
        n_x = len(all_runs)
        x_base = np.arange(n_x)
        n_series = n_axles
        x_labels = [_short_label(rn) for rn in all_runs]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n_x * n_series + 3), 5.2))
    bar_w = 0.8 / max(n_series, 1)
    many_bars = n_x * n_series >= 8
    label_rotation = 90 if many_bars else 0
    label_fontsize = 8 if many_bars else 9
    label_pts: list[tuple[float, float, float, str, str]] = []
    if compare_layout is not None:
        series_idx = 0
        for ai, (param, axle_label, axle_color) in enumerate(_AMP_AXLES):
            sigma_key = f"{param}_sigma"
            for gi, (gkey, _run_names) in enumerate(groups.items()):
                # Compare-mode: colour = group (car); hatch = axle (front/rear).
                # Fall back to axle-colour + group-hatch if no group_colors.
                if group_colors is not None:
                    fill = group_colors.get(gkey, axle_color)
                    hatch = _AXLE_HATCHES.get(param, "")
                    label = f"{gkey} {axle_label}" if n_groups > 1 else axle_label
                else:
                    fill = axle_color
                    hatch = _GROUP_HATCHES[gi % len(_GROUP_HATCHES)]
                    label = f"{axle_label} [{gkey}]" if n_groups > 1 else axle_label
                ys = []
                errs = []
                for tok in session_order:
                    rn = run_lookup.get((gkey, tok))
                    rec = None
                    if rn is not None:
                        rec = next(
                            (r for r in rows
                             if r["run"] == rn and r["mode"].lower() == mode_l),
                            None,
                        )
                    ys.append(rec[param] if (rec and np.isfinite(rec.get(param, np.nan))) else np.nan)
                    err = rec.get(sigma_key) if rec else None
                    errs.append(err if (err is not None and np.isfinite(err)) else np.nan)
                xs = x_base + (series_idx - (n_series - 1) / 2) * bar_w
                ax.bar(xs, ys, width=bar_w, color=fill, label=label,
                       edgecolor="white", linewidth=0.6, alpha=0.9,
                       hatch=hatch)
                finite_err = [e if np.isfinite(e) else 0.0 for e in errs]
                if any(e > 0 for e in finite_err):
                    ax.errorbar(xs, ys, yerr=finite_err, fmt="none",
                                ecolor=_INK, elinewidth=1.2,
                                capsize=3, capthick=1.0, alpha=0.85)
                for xc, yt, e in zip(xs, ys, finite_err):
                    if np.isfinite(yt):
                        label_pts.append((float(xc), float(yt), float(e), fill, param))
                series_idx += 1
    else:
        all_runs = [rn for run_names in groups.values() for rn in run_names]
        for ai, (param, axle_label, color) in enumerate(_AMP_AXLES):
            sigma_key = f"{param}_sigma"
            ys = []
            errs = []
            for rn in all_runs:
                rec = next(
                    (r for r in rows if r["run"] == rn and r["mode"].lower() == mode_l),
                    None,
                )
                ys.append(rec[param] if (rec and np.isfinite(rec.get(param, np.nan))) else np.nan)
                err = rec.get(sigma_key) if rec else None
                errs.append(err if (err is not None and np.isfinite(err)) else np.nan)
            xs = x_base + (ai - (n_series - 1) / 2) * bar_w
            ax.bar(xs, ys, width=bar_w, color=color, label=axle_label,
                   edgecolor="white", linewidth=0.6, alpha=0.9)
            finite_err = [e if np.isfinite(e) else 0.0 for e in errs]
            if any(e > 0 for e in finite_err):
                ax.errorbar(xs, ys, yerr=finite_err, fmt="none",
                            ecolor=_INK, elinewidth=1.2,
                            capsize=3, capthick=1.0, alpha=0.85)
            for xc, yt, e in zip(xs, ys, finite_err):
                if np.isfinite(yt):
                    label_pts.append((float(xc), float(yt), float(e), color, param))
    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(_AMP_AXIS_LABEL, fontweight="bold")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=":")
    ax.legend(loc="best", ncol=min(n_series, 4), fontsize=9, framealpha=0.9,
              handlelength=2.4, handleheight=1.4, handletextpad=0.6)
    ax.set_title(f"Modal parameters per session — {mode} amplitude (front vs rear)",
                 fontweight="bold", color=_INK)
    y_lo, y_hi = ax.get_ylim()
    if label_pts and np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        max_top = max(t + e for _, t, e, _, _ in label_pts)
        if max_top > y_hi:
            y_hi = max_top
        head = 0.20 if label_rotation == 90 else 0.12
        ax.set_ylim(y_lo, y_hi + head * (y_hi - y_lo))
    pad_pt = 3
    for xc, yt, e, color, param in label_pts:
        label = _fmt_value(param, yt)
        if not label:
            continue
        ax.annotate(
            label,
            xy=(xc, yt + e),
            xytext=(0, pad_pt),
            textcoords="offset points",
            ha="center", va="bottom",
            rotation=label_rotation, fontsize=label_fontsize,
            color=color, fontweight="bold",
            clip_on=False,
        )
    fig.tight_layout()
    fname = f"modal_evolution_bar_amp_{mode_l}.png"
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("Modal evolution bar plot saved: %s", path.name)
    return [path]
