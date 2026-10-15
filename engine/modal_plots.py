"""Modal evolution plots: fitted modal parameters across runs or sessions.

`plot_modal_evolution(plotter, ...)` reads `plotter.modal_results` (filled by
`run_workflow(vibrations_fit=...)`) and writes into `plots/<output_subdir>/`:

    modal_evolution_{line,bar}_{f0,zeta}[_<name_suffix>].png   one series per mode
    modal_evolution_{line,bar}_amp_<mode>.png                   front vs rear peak PSD

Error bands and bars show the bootstrap CI when available, else +/- sigma.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .dataplotter import DataPlotter
from .logger import log
from .plot_runtime import _session_sort_key

_INK = "#1A1A1A"
_GRID = DataPlotter.GRID_STYLE
_MODE_COLORS = {"Heave": "#1f77b4", "Pitch": "#ff7f0e", "Roll": "#2ca02c", "Warp": "#d62728"}
_AXLES = (("Front", "amp_front", "#1f77b4"), ("Rear", "amp_rear", "#d62728"))
_GROUP_COLOR_PRESETS = {
    "RED": "#d62728",
    "BLUE": "#1f77b4",
    "GREEN": "#2ca02c",
    "ORANGE": "#ff7f0e",
    "PURPLE": "#9467bd",
    "YELLOW": "#bcbd22",
    "CYAN": "#17becf",
    "PINK": "#e377c2",
}
_GROUP_PALETTE = ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
_LINESTYLES = ("-", "--", ":", "-.")
_HATCHES = ("", "///", "...", "xxx")
# Fit-result keys (value, sigma, bootstrap CI) per plotted parameter.
_PARAM_KEYS = {
    "f0": ("fn", "sigma_fn", "fn_ci"),
    "zeta": ("zeta", "sigma_zeta", "zeta_ci"),
    "amp_front": ("amp_front", "sigma_amp_front", "amp_front_ci"),
    "amp_rear": ("amp_rear", "sigma_amp_rear", "amp_rear_ci"),
}
_PARAM_LABELS = {"f0": r"$f_0$ [Hz]", "zeta": r"$\zeta$ [-]"}


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
    """Render modal parameters (f0, zeta, front/rear amplitude) across runs.

    Parameters
    ----------
    plotter : DataPlotter
        Returned by ``run_workflow``; provides ``runs``, ``modal_results``,
        ``plots_dir`` and ``output_dpi``.
    modes : tuple of str
        Mode labels to include.
    group_by : str, optional
        Run-dict key (e.g. ``"group"``) splitting runs into series, e.g. RED
        vs BLUE car.
    compare_by : "session" or callable, optional
        Maps a run name to an x-axis token so groups share x positions:
        ``"session"`` takes the last ``_``-separated token, in session order.
        Groups are then coloured by name (RED, BLUE, ...).
    include_consolidated : bool
        Keep runs whose name contains "consolidated".
    output_subdir : str
        Folder under ``plotter.plots_dir``.
    name_suffix : str
        Appended to the f0/zeta filenames so disjoint ``modes`` calls don't collide.
    line_ci, bars : bool
        Draw the line + error band and/or the bar + error bar variants.

    Returns
    -------
    list of Path
        Saved figures; figures with no finite values are skipped.
    """
    results = getattr(plotter, "modal_results", None) or {}
    runs = [
        r
        for r in plotter.runs
        if r["name"].lower() in results and (include_consolidated or "consolidated" not in r["name"].lower())
    ]
    if not runs:
        log.info("plot_modal_evolution: no modal results to plot.")
        return []

    records = _records(results, runs)
    groups: OrderedDict = OrderedDict()
    for r in runs:
        groups.setdefault(r.get(group_by) if group_by else None, []).append(r["name"])
    compare = _compare_tokens(runs, compare_by)
    layout = _layout(groups, compare)
    group_colors = _group_colors(groups) if compare else None
    out_dir = Path(plotter.plots_dir) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = getattr(plotter, "output_dpi", 300)

    # (filename stem, items = (series name, mode, param, colour), y label, title)
    one_mode = f" - {modes[0]}" if len(modes) == 1 else ""
    suffix = f"_{name_suffix}" if name_suffix else ""
    figures = [
        (f"{p}{suffix}", [(m, m, p, _MODE_COLORS.get(m, "#7f7f7f")) for m in modes], _PARAM_LABELS[p], one_mode)
        for p in ("f0", "zeta")
    ]
    figures += [
        (f"amp_{m.lower()}", [(axle, m, p, c) for axle, p, c in _AXLES], "Peak PSD", f" - {m} front vs rear")
        for m in modes
    ]
    saved = []
    for kind in [k for k, on in (("line", line_ci), ("bar", bars)) if on]:
        for stem, items, ylabel, title in figures:
            series, n_slots = _series(items, groups, group_colors)
            path = out_dir / f"modal_evolution_{kind}_{stem}.png"
            if _figure(kind, path, series, n_slots, records, layout, ylabel, title, dpi):
                saved.append(path)
    return saved


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def _at(values, i: int) -> float:
    try:
        return float(values[i])
    except (TypeError, IndexError, ValueError):
        return float("nan")


def _records(results: dict, runs: list) -> dict:
    """``{(run, mode): {param: (value, lo, hi)}}``; lo/hi = bootstrap CI, else value -/+ sigma, else NaN."""
    out = {}
    for run in runs:
        res = results[run["name"].lower()]
        for i, mode in enumerate(res.get("mode_labels", [])):
            rec = {}
            for param, (key, sigma_key, ci_key) in _PARAM_KEYS.items():
                value = _at(res.get(key), i)
                ci = res.get(ci_key)
                lo, hi = (_at(ci[0], i), _at(ci[1], i)) if ci is not None else (np.nan, np.nan)
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    sigma = _at(res.get(sigma_key), i)
                    lo, hi = value - sigma, value + sigma
                rec[param] = (value, lo, hi)
            out[(run["name"], str(mode).lower())] = rec
    return out


def _compare_tokens(runs: list, compare_by) -> tuple[dict, list] | None:
    """``(run -> token, ordered tokens)`` for a shared x-axis, or None for one x position per run."""
    if compare_by is None:
        return None
    if compare_by == "session":

        def extract(name):
            return str(name).split("_")[-1]

    elif callable(compare_by):
        extract = compare_by
    else:
        raise ValueError(f"compare_by must be 'session' or a callable, got {compare_by!r}")
    tokens = {r["name"]: str(extract(r["name"])) for r in runs}
    order = list(dict.fromkeys(tokens.values()))
    if compare_by == "session":
        order.sort(key=_session_sort_key)
    return tokens, order


def _layout(groups: OrderedDict, compare) -> tuple[dict, list, list]:
    """``(run -> x, tick positions, tick labels)``: one tick per token, else per run with a gap between groups."""
    if compare is not None:
        tokens, order = compare
        return {rn: float(order.index(tok)) for rn, tok in tokens.items()}, list(range(len(order))), order
    run_x, x = {}, 0.0
    for run_names in groups.values():
        for rn in run_names:
            run_x[rn] = x
            x += 1.0
        x += 0.5
    return run_x, list(run_x.values()), [_short_label(rn) for rn in run_x]


def _group_colors(groups: OrderedDict) -> dict:
    """Named preset (e.g. "RED") per group, else the palette by group order."""
    return {
        g: _GROUP_COLOR_PRESETS.get(str(g).upper(), _GROUP_PALETTE[i % len(_GROUP_PALETTE)])
        for i, g in enumerate(groups)
    }


def _series(items: list, groups: OrderedDict, group_colors: dict | None) -> tuple[list[dict], int]:
    """One series per (item, group) and the number of bar slots per x position.

    When comparing, colour encodes the group and line style / hatch the item;
    otherwise colour encodes the item and style the group.
    """
    compare = group_colors is not None
    n_items, n_groups = len(items), len(groups)
    out = []
    for ii, (name, mode, param, item_color) in enumerate(items):
        for gi, (gkey, run_names) in enumerate(groups.items()):
            parts = (None if gkey is None else str(gkey), name if n_items > 1 or gkey is None else None)
            out.append(
                {
                    "label": " ".join(p for p in parts if p),
                    "mode": mode.lower(),
                    "param": param,
                    "runs": run_names,
                    "color": group_colors[gkey] if compare else item_color,
                    "style": ii if compare else gi,
                    "slot": ii * n_groups + gi if compare else ii,
                }
            )
    return out, n_items * n_groups if compare else n_items


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------


def _short_label(run_name: str) -> str:
    s = str(run_name)
    return s if len(s) <= 22 else f"{s[:10]}\u2026{s[-10:]}"


def _fmt(param: str, value: float) -> str:
    if param == "f0":
        return f"{value:.2f}"
    if param == "zeta":
        return f"{value:.3f}"
    return f"{value:.2f}" if 1e-2 <= abs(value) < 1e3 else f"{value:.1e}"


def _figure(kind, path, series, n_slots, records, layout, ylabel, title, dpi) -> bool:
    """Draw one line or bar figure; returns False (nothing saved) when no series has data."""
    run_x, ticks, tick_labels = layout
    width_per_x = 0.6 * (n_slots if kind == "bar" else 1)
    fig, ax = plt.subplots(figsize=(max(9.0, width_per_x * len(ticks) + 3), 5.2))
    bar_w = 0.8 / n_slots
    readouts = []  # (x, top of error bar, value, colour, text)
    for s in series:
        pts = sorted((run_x[rn], *records[(rn, s["mode"])][s["param"]]) for rn in s["runs"] if (rn, s["mode"]) in records)
        pts = [p for p in pts if np.isfinite(p[1])]
        if not pts:
            continue
        x, y, lo, hi = (np.array(v) for v in zip(*pts))
        has_err = np.isfinite(lo) & np.isfinite(hi)
        if kind == "line":
            ls = _LINESTYLES[s["style"] % len(_LINESTYLES)]
            ax.plot(x, y, color=s["color"], linestyle=ls, marker="o", linewidth=2.0, markersize=5.5, label=s["label"])
            ax.fill_between(x, lo, hi, color=s["color"], alpha=0.15, linewidth=0)
            ecolor = s["color"]
        else:
            x = x + (s["slot"] - (n_slots - 1) / 2) * bar_w
            hatch = _HATCHES[s["style"] % len(_HATCHES)]
            ax.bar(x, y, width=bar_w, color=s["color"], hatch=hatch, edgecolor="white", linewidth=0.6, label=s["label"])
            ecolor = _INK
        if has_err.any():
            yerr = np.vstack([y - lo, hi - y])[:, has_err]
            ax.errorbar(x[has_err], y[has_err], yerr=yerr, fmt="none", ecolor=ecolor, elinewidth=1.0, capsize=3)
        tops = np.where(has_err, hi, y)
        readouts += [(xi, ti, yi, s["color"], _fmt(s["param"], yi)) for xi, ti, yi in zip(x, tops, y)]
    if not readouts:
        plt.close(fig)
        return False

    headroom = _annotate(ax, readouts, kind)
    y_lo, y_hi = ax.get_ylim()
    y_hi = max(y_hi, max(r[1] for r in readouts))
    ax.set_ylim(y_lo, y_hi + headroom * (y_hi - y_lo))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(f"Modal evolution - {ylabel}{title}", fontweight="bold", color=_INK)
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", **_GRID["major"])
    ax.grid(True, which="minor", axis="y", linestyle=":", **_GRID["minor"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", ncol=min(len(series), 4), fontsize=9, framealpha=0.9, handlelength=2.8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("Modal evolution plot saved: %s", path.name)
    return True


def _annotate(ax, readouts: list, kind: str) -> float:
    """Write value readouts and return the y headroom (fraction of range) they need.

    Lines: readouts sharing an x are stacked above the highest error bar, in
    order of value. Bars: one readout above each bar's error bar, rotated
    when bars are crowded.
    """
    if kind == "line":
        by_x: dict = {}
        for r in readouts:
            by_x.setdefault(r[0], []).append(r)
        placed = []
        for x, group in by_x.items():
            top = max(r[1] for r in group)
            placed += [(x, top, 6 + 11 * slot, 0, r[3], r[4]) for slot, r in enumerate(sorted(group, key=lambda r: r[2]))]
        headroom = 0.08 + 0.05 * max(len(g) for g in by_x.values())
    else:
        rotation = 90 if len(readouts) >= 8 else 0
        placed = [(x, top, 3, rotation, color, text) for x, top, _, color, text in readouts]
        headroom = 0.20 if rotation else 0.12
    for x, top, dy, rotation, color, text in placed:
        ax.annotate(
            text,
            xy=(x, top),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=8,
            color=color,
            fontweight="bold",
            clip_on=False,
        )
    return headroom
