"""Module-level generator functions extracted from ``HeatmapMixin``.

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


def _lorentz_peak_model(f, f0, zeta, amp, baseline):
    denom = (f0**2 - f**2) ** 2 + (2.0 * zeta * f0 * f) ** 2
    return amp * f0**4 / np.maximum(denom, 1e-30) + baseline


_ZETA_LO_PSD, _ZETA_HI_PSD = 1e-3, 1.0
_SATURATION_MARGIN = 0.05


def _fit_lorentz_peak(freq, power, f_lo, f_hi, min_points=8):
    from scipy.optimize import least_squares

    mask = (freq >= f_lo) & (freq <= f_hi)
    if int(mask.sum()) < min_points:
        return None
    f_fit = np.asarray(freq[mask], dtype=float)
    p_fit = np.asarray(power[mask], dtype=float)
    if not np.all(np.isfinite(p_fit)) or float(np.max(p_fit)) <= 0.0:
        return None
    p_peak = float(np.max(p_fit))
    p_base = float(np.percentile(p_fit, 10))
    f0_init = float(f_fit[int(np.argmax(p_fit))])
    amp_hi = max(p_peak * 1e3, 1.0)
    base_hi = max(p_peak, 1e-9)
    p0 = [f0_init, 0.05, max(p_peak - p_base, 1e-12), p_base]
    lo_bounds = [f_lo, _ZETA_LO_PSD, 0.0, 0.0]
    hi_bounds = [f_hi, _ZETA_HI_PSD, amp_hi, base_hi]
    log_p = np.log(np.maximum(p_fit, 1e-30))

    def residual(params):
        model = _lorentz_peak_model(f_fit, *params)
        return np.log(np.maximum(model, 1e-30)) - log_p

    try:
        res = least_squares(residual, p0, bounds=(lo_bounds, hi_bounds), max_nfev=2000)
    except Exception:
        return None
    f0_fit, zeta_fit, amp_fit, base_fit = (float(v) for v in res.x)
    log_p_pred = np.log(np.maximum(_lorentz_peak_model(f_fit, f0_fit, zeta_fit, amp_fit, base_fit), 1e-30))
    ss_res = float(np.sum((log_p - log_p_pred) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_obs = len(p_fit)
    dof = max(n_obs - 4, 1)
    mse = 2.0 * float(res.cost) / dof
    sigma_zeta = float("nan")
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * mse
        var_zeta = float(cov[1, 1])
        if var_zeta >= 0.0 and np.isfinite(var_zeta):
            sigma_zeta = float(np.sqrt(var_zeta))
    except np.linalg.LinAlgError:
        pass
    saturated = False
    for val, b_lo, b_hi in (
        (f0_fit, f_lo, f_hi),
        (zeta_fit, _ZETA_LO_PSD, _ZETA_HI_PSD),
    ):
        span = b_hi - b_lo
        if span > 0 and (val - b_lo < _SATURATION_MARGIN * span or b_hi - val < _SATURATION_MARGIN * span):
            saturated = True
            break
    return f0_fit, zeta_fit, amp_fit, base_fit, f_lo, f_hi, sigma_zeta, r_squared, saturated


def generate_heatmap_plots(plotter):
    plotter._ensure_preprocessed()
    plots = plotter._get_plot_group(6)
    if not plots:
        return
    plot_iter = plots if plotter.verbose else _tqdm(plots, desc="Heatmap", unit="plot", leave=True)
    for plot_def in plot_iter:
        plot_name = plot_def.name
        x_channel = plot_def.x_channel
        y_channel = plot_def.y_channel
        z_channel = plot_def.z_channel
        agg = plot_def.aggregation
        bins = plot_def.bins
        axis_limits = plot_def.axis_limits
        cmap = plot_def.cmap
        z_limits = plot_def.z_limits
        gate_spec = plot_def.gate
        markers = plot_def.markers
        min_count = plot_def.min_count
        if plotter.verbose:
            log.debug("Creating heatmap plot: %s", plot_name)
        run_frames = []
        for run in plotter.runs:
            rn = run["name"].lower()
            df = plotter._get_filtered_run_dataframe(rn, gate_spec)
            if df is None or x_channel not in df.columns or y_channel not in df.columns:
                continue
            if z_channel is not None and z_channel not in df.columns:
                continue
            run_frames.append((run, df))
        if not run_frames:
            log.warning("Heatmap '%s': no usable runs. Skipping.", plot_name)
            continue
        ncols = len(run_frames)
        base_w = plotter.histogram_FIGSIZE[0]
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(base_w * 0.9 * ncols, plotter.histogram_FIGSIZE[1]),
            squeeze=False,
        )
        axes = axes[0]
        xs_all = np.concatenate(
            [pd.to_numeric(df[x_channel], errors="coerce").dropna().to_numpy() for _, df in run_frames]
        )
        ys_all = np.concatenate(
            [pd.to_numeric(df[y_channel], errors="coerce").dropna().to_numpy() for _, df in run_frames]
        )
        if axis_limits and axis_limits[0]:
            x_lo = axis_limits[0][0] if axis_limits[0][0] is not None else float(np.nanmin(xs_all))
            x_hi = axis_limits[0][1] if axis_limits[0][1] is not None else float(np.nanmax(xs_all))
        else:
            x_lo, x_hi = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
        if axis_limits and axis_limits[1]:
            y_lo = axis_limits[1][0] if axis_limits[1][0] is not None else float(np.nanmin(ys_all))
            y_hi = axis_limits[1][1] if axis_limits[1][1] is not None else float(np.nanmax(ys_all))
        else:
            y_lo, y_hi = float(np.nanmin(ys_all)), float(np.nanmax(ys_all))
        if isinstance(bins, tuple):
            nx, ny = bins
        else:
            nx = ny = int(bins)
        x_edges = np.linspace(x_lo, x_hi, nx + 1)
        y_edges = np.linspace(y_lo, y_hi, ny + 1)
        all_grids = []
        for run, df in run_frames:
            xv = pd.to_numeric(df[x_channel], errors="coerce").to_numpy()
            yv = pd.to_numeric(df[y_channel], errors="coerce").to_numpy()
            mask = np.isfinite(xv) & np.isfinite(yv)
            if z_channel is None:
                grid, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[x_edges, y_edges])
                counts = grid
            else:
                zv = pd.to_numeric(df[z_channel], errors="coerce").to_numpy()
                mask = mask & np.isfinite(zv)
                counts, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[x_edges, y_edges])
                sums, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[x_edges, y_edges], weights=zv[mask])
                with np.errstate(invalid="ignore", divide="ignore"):
                    if agg == "mean":
                        grid = sums / counts
                    elif agg == "sum":
                        grid = sums
                    elif agg == "count":
                        grid = counts
                    else:
                        xi = np.clip(np.digitize(xv[mask], x_edges) - 1, 0, nx - 1)
                        yi = np.clip(np.digitize(yv[mask], y_edges) - 1, 0, ny - 1)
                        zfin = zv[mask]
                        grid = np.full((nx, ny), np.nan)
                        from collections import defaultdict

                        buckets = defaultdict(list)
                        for i_, j_, z_ in zip(xi, yi, zfin):
                            buckets[(i_, j_)].append(z_)
                        reducer = {
                            "median": np.median,
                            "std": np.std,
                            "max": np.max,
                            "min": np.min,
                        }[agg]
                        for (i_, j_), values in buckets.items():
                            if len(values) >= min_count:
                                grid[i_, j_] = reducer(values)
                grid = np.where(counts >= min_count, grid, np.nan)
            all_grids.append((run, grid, counts))
        zs_combined = np.concatenate([g.ravel() for _, g, _ in all_grids])
        zs_combined = zs_combined[np.isfinite(zs_combined)]
        if z_limits and (z_limits[0] is not None or z_limits[1] is not None):
            z_min = (
                z_limits[0] if z_limits[0] is not None else float(np.nanmin(zs_combined)) if len(zs_combined) else 0.0
            )
            z_max = (
                z_limits[1] if z_limits[1] is not None else float(np.nanmax(zs_combined)) if len(zs_combined) else 1.0
            )
        elif len(zs_combined):
            z_min, z_max = float(np.nanmin(zs_combined)), float(np.nanmax(zs_combined))
        else:
            z_min, z_max = 0.0, 1.0
        for ax, (run, grid, counts) in zip(axes, all_grids):
            im = ax.imshow(
                grid.T,
                origin="lower",
                aspect="auto",
                extent=(x_lo, x_hi, y_lo, y_hi),
                cmap=cmap,
                vmin=z_min,
                vmax=z_max,
                interpolation="nearest",
            )
            ax.set_title(run["name"].upper(), fontsize=11, fontweight="bold", color=run["color"])
            ax.set_xlabel(datafunctions.add_units_to_label(x_channel, plotter.units_map), fontweight="bold")
            ax.set_ylabel(datafunctions.add_units_to_label(y_channel, plotter.units_map), fontweight="bold")
            plotter._apply_grid(ax, which="major")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plotter._draw_static_markers(ax, markers, x_clip=False)
        cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
        if z_channel is None:
            cbar.set_label("Count", fontweight="bold")
        else:
            cbar.set_label(
                f"{agg}({datafunctions.add_units_to_label(z_channel, plotter.units_map)})",
                fontweight="bold",
            )
        filename = plotter._sanitize_plot_filename("heatmap", plot_name)
        fig.suptitle(plot_name, fontweight="bold", fontsize=13)
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
