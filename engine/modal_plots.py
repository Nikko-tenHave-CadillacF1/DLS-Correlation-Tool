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
    "f0": r"$f_0$ [Hz]",
    "zeta": r"$\zeta$ [-]",
}


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
    if line_ci:
        for param in ("f0", "zeta"):
            saved.extend(_line_ci_figure(
                rows, modes, param, groups, out_dir, dpi, name_suffix,
            ))
    if bars:
        for param in ("f0", "zeta"):
            saved.extend(_bar_figure(
                rows, modes, param, groups, out_dir, dpi, name_suffix,
            ))
    return saved


def _line_ci_figure(rows, modes, param, groups, out_dir, dpi, name_suffix="") -> list[Path]:
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * sum(len(g) for g in groups.values()) + 3), 5.2))
    sigma_key = f"{param}_sigma"
    label_text = _PARAM_LABELS[param]
    # Build x positions per (group, run) and label list
    x_pos = []
    x_labels = []
    run_to_x: dict[tuple, int] = {}
    cursor = 0
    for gkey, run_names in groups.items():
        for rn in run_names:
            x_pos.append(cursor)
            x_labels.append(_short_label(rn))
            run_to_x[(gkey, rn)] = cursor
            cursor += 1
        cursor += 0.5  # small visual gap between groups
    linestyles = ["-", "--", ":", "-."]
    for mi, mode in enumerate(modes):
        color = _mode_color(mode)
        for gi, (gkey, run_names) in enumerate(groups.items()):
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
                xs.append(run_to_x[(gkey, rn)])
                ys.append(rec[param])
                err = rec.get(sigma_key)
                errs.append(err if (err is not None and np.isfinite(err)) else 0.0)
            if not xs:
                continue
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            errs = np.asarray(errs)
            ls = linestyles[gi % len(linestyles)]
            glabel = f"{mode}" + (f" [{gkey}]" if group_by_label(gkey) else "")
            ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                    linewidth=2.0, markersize=5.5, label=glabel)
            if np.any(errs > 0):
                ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.15)
                ax.errorbar(xs, ys, yerr=errs, fmt="none",
                            ecolor=color, elinewidth=1.0,
                            capsize=3, capthick=1.0, alpha=0.55)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(label_text, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=9, framealpha=0.9)
    title = f"Modal evolution — {label_text}"
    ax.set_title(title, fontweight="bold", color=_INK)
    fig.tight_layout()
    fname = f"modal_evolution_line_{param}{('_' + name_suffix) if name_suffix else ''}.png"
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log.info("Modal evolution line plot saved: %s", path.name)
    return [path]


def _bar_figure(rows, modes, param, groups, out_dir, dpi, name_suffix="") -> list[Path]:
    sigma_key = f"{param}_sigma"
    label_text = _PARAM_LABELS[param]
    all_runs = [rn for run_names in groups.values() for rn in run_names]
    n_runs = len(all_runs)
    n_modes = len(modes)
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n_runs * n_modes + 3), 5.2))
    bar_w = 0.8 / max(n_modes, 1)
    x_base = np.arange(n_runs)
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
    ax.set_xticks(x_base)
    ax.set_xticklabels([_short_label(rn) for rn in all_runs],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(label_text, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", ncol=min(n_modes, 4), fontsize=9, framealpha=0.9)
    ax.set_title(f"Modal parameters per session — {label_text}",
                 fontweight="bold", color=_INK)
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
