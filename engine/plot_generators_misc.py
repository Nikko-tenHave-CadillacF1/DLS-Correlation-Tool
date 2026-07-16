
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from . import datafunctions
from .datafunctions import _tqdm
from .logger import log


def _lorentz_peak_model(f, f0, zeta, amp, baseline):
    denom = (f0 ** 2 - f ** 2) ** 2 + (2.0 * zeta * f0 * f) ** 2
    return amp * f0 ** 4 / np.maximum(denom, 1e-30) + baseline

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
    log_p_pred = np.log(np.maximum(
        _lorentz_peak_model(f_fit, f0_fit, zeta_fit, amp_fit, base_fit), 1e-30))
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
        if span > 0 and (val - b_lo < _SATURATION_MARGIN * span
                         or b_hi - val < _SATURATION_MARGIN * span):
            saturated = True
            break
    return f0_fit, zeta_fit, amp_fit, base_fit, f_lo, f_hi, sigma_zeta, r_squared, saturated

class PsdHistMixin:

    LORENTZ_COMPACT_THRESHOLD = 5

    def generate_psd_plots(self):
        self._ensure_preprocessed()
        plots = self._get_plot_group(2)
        if not plots:
            return
        self._lorentz_fit_records = []
        plot_iter = plots if self.verbose else _tqdm(plots, desc="PSD", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name     = plot_def.name
            channel       = plot_def.channel
            axis_limits   = plot_def.axis_limits
            log_scale     = plot_def.log_scale
            annotate_at   = plot_def.annotate_at
            markers       = plot_def.markers
            gate_spec     = getattr(plot_def, "gate", None)
            show_envelope = bool(getattr(plot_def, "show_envelope", False))
            reference_lines = getattr(plot_def, "reference_lines", None)
            lorentz_fit_windows = getattr(plot_def, "lorentz_fit", None) or []
            channels_list = [channel] if isinstance(channel, str) else list(channel)
            if plot_def.nperseg is not None:
                nperseg = plot_def.nperseg
                _nperseg_ref_rate = None  # user-specified; no per-run scaling
            else:
                sample_counts = []
                sample_rates = []
                for run in self.runs:
                    df = self.run_data.get(run["name"].lower())
                    if df is None:
                        continue
                    run_rate = self.run_sample_rates.get(
                        run["name"].lower(),
                        (self.FILTER_SAMPLE_RATE, "default"),
                    )[0]
                    if run_rate and np.isfinite(run_rate):
                        sample_rates.append(float(run_rate))
                    for ch in channels_list:
                        if ch in df.columns:
                            sample_counts.append(int(np.isfinite(df[ch].to_numpy(dtype=float)).sum()))
                n_min = min(sample_counts) if sample_counts else 0
                fs_ref = min(sample_rates) if sample_rates else float(self.FILTER_SAMPLE_RATE)
                nperseg = datafunctions.auto_nperseg(
                    n_min, sample_rate=fs_ref,
                    min_averages_target=self.PSD_MIN_AVERAGES_TARGET,
                ) if n_min >= 16 else 512
                _nperseg_ref_rate = fs_ref  # remember reference rate for per-run scaling
            line_styles = ["-", "--", ":", "-."]
            if self.verbose:
                log.debug("Creating PSD plot: %s (%s)", plot_name, ', '.join(channels_list))
            filename = self._sanitize_plot_filename("psd", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.psd_FIGSIZE)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel("Frequency (Hz)", fontweight="bold")
            primary_ch = channels_list[0]
            ax.set_ylabel(
                datafunctions.format_psd_ylabel(primary_ch, self.units_map),
                fontweight="bold",
            )
            run_groups = {}
            for run in self.runs:
                run_groups[run["name"]] = [run]
            plotted_any = False
            multi = len(channels_list) > 1
            psd_curves = []
            lorentz_results = []
            nyquist_lines = {}
            for group_name, group_runs in run_groups.items():
                loaded_runs = [r for r in group_runs if r["name"].lower() in self.run_data]
                if not loaded_runs:
                    log.warning(
                        "PSD '%s': group '%s' has no loaded runs. Skipping.",
                        plot_name, group_name,
                    )
                    continue
                primary_run = loaded_runs[0]
                group_color = primary_run["color"]
                rate_info = self.run_sample_rates.get(
                    primary_run["name"].lower(),
                    (self.FILTER_SAMPLE_RATE, "default"),
                )
                fs = float(rate_info[0]) if rate_info and rate_info[0] else 0.0
                if fs > 0:
                    nyq = round(fs / 2.0, 3)
                    nyquist_lines.setdefault(nyq, group_color)
                for ch_idx, ch in enumerate(channels_list):
                    per_run_psds = []
                    for run in loaded_runs:
                        run_name = run["name"].lower()
                        df = self.run_data.get(run_name)
                        if df is None or ch not in df.columns:
                            hint = self._format_missing_channel_hint(run_name, ch)
                            log.warning(
                                "PSD '%s': channel '%s' missing in run '%s'. Skipping.%s",
                                plot_name, ch, run_name, f"\n{hint}" if hint else "",
                            )
                            continue
                        # Scale nperseg by sample-rate ratio so all runs share
                        # the same frequency resolution (Δf) regardless of
                        # native logging rate.
                        if _nperseg_ref_rate and _nperseg_ref_rate > 0:
                            run_rate = self.run_sample_rates.get(
                                run_name, (self.FILTER_SAMPLE_RATE, "default"))[0]
                            run_nperseg = max(64, int(nperseg * run_rate / _nperseg_ref_rate))
                        else:
                            run_nperseg = nperseg
                        freq, power, n_segs = self._cached_psd_with_segments(
                            run_name, ch, run_nperseg, gate_spec=gate_spec,
                        )
                        if freq is None or power is None or n_segs <= 0:
                            continue
                        per_run_psds.append((freq, power, n_segs))
                    if not per_run_psds:
                        continue
                    freq = per_run_psds[0][0]
                    if len(per_run_psds) == 1:
                        power_mean = per_run_psds[0][1]
                        power_std = None
                    else:
                        powers = []
                        weights = []
                        for f_i, p_i, n_i in per_run_psds:
                            if not np.array_equal(f_i, freq):
                                p_i = np.interp(freq, f_i, p_i)
                            powers.append(np.asarray(p_i, dtype=float))
                            weights.append(float(n_i))
                        stacked = np.vstack(powers)
                        w = np.asarray(weights, dtype=float)
                        power_mean = (stacked * w[:, None]).sum(axis=0) / w.sum()
                        if show_envelope:
                            var = (((stacked - power_mean) ** 2) * w[:, None]).sum(axis=0) / w.sum()
                            power_std = np.sqrt(var)
                        else:
                            power_std = None
                    lstyle = line_styles[ch_idx % len(line_styles)]
                    lbl = f"{group_name.upper()} — {ch}" if multi else group_name.upper()
                    plot_func = ax.semilogy if log_scale else ax.plot
                    plot_func(
                        freq, power_mean,
                        linewidth=1.8, color=group_color,
                        linestyle=lstyle, alpha=0.9, label=lbl,
                    )
                    if power_std is not None:
                        lower = np.clip(power_mean - power_std, 1e-30 if log_scale else None, None)
                        upper = power_mean + power_std
                        ax.fill_between(
                            freq, lower, upper,
                            color=group_color, alpha=0.15, linewidth=0, zorder=1,
                        )
                    psd_curves.append((group_color, freq, power_mean))
                    plotted_any = True
                    for f_lo_u, f_hi_u in lorentz_fit_windows:
                        fit_res = _fit_lorentz_peak(
                            freq, power_mean, f_lo_u, f_hi_u)
                        if fit_res is None:
                            self._record_lorentz_fit(
                                plot_name, group_name, ch, f_lo_u, f_hi_u,
                                None, None, None, None, None, failed=True)
                            continue
                        (f0_fit, zeta_fit, amp_fit, base_fit, f_lo, f_hi,
                         sigma_zeta, r_squared, saturated) = fit_res
                        f_dense = np.linspace(f_lo, f_hi, 200)
                        y_dense = _lorentz_peak_model(
                            f_dense, f0_fit, zeta_fit, amp_fit, base_fit)
                        plot_func(
                            f_dense, y_dense, linewidth=1.6, color=group_color,
                            linestyle=":", alpha=0.95, label=None, zorder=5,
                        )
                        peak_idx = int(np.argmax(y_dense))
                        f_peak_vis = float(f_dense[peak_idx])
                        model_peak = float(y_dense[peak_idx])
                        lorentz_results.append(
                            (f_lo, f_hi, model_peak, f_peak_vis, group_color,
                             zeta_fit, f0_fit, sigma_zeta, group_name))
                        self._record_lorentz_fit(
                            plot_name, group_name, ch, f_lo, f_hi,
                            f0_fit, zeta_fit, sigma_zeta, r_squared, saturated)
            if not plotted_any:
                log.warning(
                    "PSD '%s': no valid data for '%s'. Plot not saved.",
                    plot_name, ', '.join(channels_list),
                )
                plt.close(fig)
                continue
            has_x_limits, has_y_limits = self._apply_2d_axis_limits(
                ax, axis_limits, log_scale_y=log_scale,
            )
            default_y_pad = 0 if log_scale else 0.04
            self._add_axis_edge_padding(
                ax,
                x_pad_ratio=(0 if has_x_limits else 0.02),
                y_pad_ratio=(0 if has_y_limits else default_y_pad),
            )
            self._apply_grid(ax, which="both")
            if log_scale:
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=24))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.grid(True, which="minor", axis="y", alpha=0.20, linewidth=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            self._add_standard_legend(ax, loc="best")
            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    legend_obj = ax.get_legend()
                    self._display_gate_info(ax, gate_text, legend=legend_obj)
            lorentz_by_color = {}
            for (lf_lo, lf_hi, _lmp, _lfpv, lcol, lzf, lff, lsz, _lgn) in lorentz_results:
                lorentz_by_color.setdefault(lcol, []).append((lf_lo, lf_hi, lzf, lff, lsz))
            def _find_lorentz_for(color_e, f_at):
                for entry in lorentz_by_color.get(color_e, []):
                    lo_, hi_, _zf, _ff, _sz = entry
                    if lo_ <= f_at <= hi_:
                        return entry
                return None
            consumed_lorentz_keys = set()
            if annotate_at is not None and psd_curves:
                if isinstance(annotate_at, (list, tuple)):
                    freq_targets = [float(v) for v in annotate_at]
                else:
                    freq_targets = [float(annotate_at)]
                xl, xr = ax.get_xlim()
                for f_at in freq_targets:
                    if not (xl <= f_at <= xr):
                        continue
                    ax.axvline(f_at, color="#5E5E5E", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
                    ann_items = []
                    for (run_color, freq_arr, power_arr) in psd_curves:
                        if len(freq_arr) == 0 or f_at > freq_arr[-1]:
                            continue
                        idx = np.argmin(np.abs(freq_arr - f_at))
                        p_at = power_arr[idx]
                        ann_items.append((p_at, run_color))
                    if ann_items:
                        ann_items.sort(key=lambda t: t[0])
                        trans = ax.transData
                        display_ys = [trans.transform((f_at, item[0]))[1] for item in ann_items]
                        has_two_line = any(
                            _find_lorentz_for(c, f_at) is not None for _, c in ann_items
                        )
                        min_sep = 30 if has_two_line else 16
                        adjusted_display_ys = list(display_ys)
                        for i in range(1, len(adjusted_display_ys)):
                            gap = adjusted_display_ys[i] - adjusted_display_ys[i - 1]
                            if gap < min_sep:
                                adjusted_display_ys[i] = adjusted_display_ys[i - 1] + min_sep
                        n_items = len(ann_items)
                        for i in reversed(range(n_items)):
                            p_at, color_e = ann_items[i]
                            nudge_pts = adjusted_display_ys[i] - display_ys[i]
                            y_offset = 8 + nudge_pts
                            ax.scatter([f_at], [p_at], color=color_e, s=50, zorder=10,
                                       edgecolors="white", linewidths=1.2)
                            extra = _find_lorentz_for(color_e, f_at)
                            if extra is not None:
                                lo_, hi_, zf, ff, sz = extra
                                if sz is not None and np.isfinite(sz):
                                    label = f"{p_at:.3g}\n$f_0$={ff:.2f}Hz $\\zeta$={zf:.3f}\u00b1{sz:.3f}"
                                else:
                                    label = f"{p_at:.3g}\n$f_0$={ff:.2f}Hz $\\zeta$={zf:.3f}"
                                consumed_lorentz_keys.add((round(lo_, 4), round(hi_, 4), color_e))
                            else:
                                label = f"{p_at:.3g}"
                            ax.annotate(
                                label,
                                xy=(f_at, p_at), xytext=(10, y_offset),
                                textcoords="offset points",
                                fontsize=9, fontweight="bold", color=color_e,
                                zorder=11 + (n_items - 1 - i),
                                arrowprops=dict(arrowstyle="-", color=color_e,
                                                lw=0.8, alpha=0.6),
                                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                          alpha=0.92, edgecolor=color_e, linewidth=0.8),
                            )
            remaining_lorentz = [
                t for t in lorentz_results
                if (round(float(t[0]), 4), round(float(t[1]), 4), t[4]) not in consumed_lorentz_keys
            ]
            if remaining_lorentz:
                xl, xr = ax.get_xlim()
                trans = ax.transData
                by_window = {}
                for f_lo, f_hi, model_pk, f_peak_vis, col, zf, ff, sz, gname in remaining_lorentz:
                    by_window.setdefault((f_lo, f_hi), []).append(
                        (model_pk, f_peak_vis, col, zf, ff, sz, gname))
                for _win_key, items in by_window.items():
                    items.sort(key=lambda t: t[0])
                    if len(items) >= self.LORENTZ_COMPACT_THRESHOLD:
                        for model_pk, f_peak_vis, color_e, _zf, _ff, _sz, _gn in items:
                            if xl <= f_peak_vis <= xr:
                                ax.scatter([f_peak_vis], [model_pk],
                                           color=color_e, s=40, zorder=10,
                                           edgecolors="white", linewidths=1.2)
                        self._draw_lorentz_compact_box(ax, _win_key, items, psd_curves)
                        continue
                    display_ys = [trans.transform((it[1], it[0]))[1] for it in items]
                    min_sep = 30
                    adjusted = list(display_ys)
                    for i in range(1, len(adjusted)):
                        gap = adjusted[i] - adjusted[i - 1]
                        if gap < min_sep:
                            adjusted[i] = adjusted[i - 1] + min_sep
                    n_items = len(items)
                    for i in reversed(range(n_items)):
                        model_pk, f_peak_vis, color_e, zf, f0_fit, sz, _gn = items[i]
                        if not (xl <= f_peak_vis <= xr):
                            continue
                        nudge_pts = adjusted[i] - display_ys[i]
                        y_offset = 8 + nudge_pts
                        ax.scatter([f_peak_vis], [model_pk], color=color_e, s=50, zorder=10,
                                   edgecolors="white", linewidths=1.2)
                        if sz is not None and np.isfinite(sz):
                            label = f"$a_0$={model_pk:.3g}\n$f_0$={f0_fit:.2f}Hz $\\zeta$={zf:.3f}\u00b1{sz:.3f}"
                        else:
                            label = f"$a_0$={model_pk:.3g}\n$f_0$={f0_fit:.2f}Hz $\\zeta$={zf:.3f}"
                        ax.annotate(
                            label,
                            xy=(f_peak_vis, model_pk), xytext=(10, y_offset),
                            textcoords="offset points",
                            fontsize=9, fontweight="bold", color=color_e,
                            zorder=11 + (n_items - 1 - i),
                            arrowprops=dict(arrowstyle="-", color=color_e,
                                            lw=0.8, alpha=0.6),
                            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                      alpha=0.92, edgecolor=color_e, linewidth=0.8),
                        )
            self._draw_static_markers(ax, markers)
            self._draw_horizontal_reference_lines(ax, reference_lines)
            if nyquist_lines:
                xl, xr = ax.get_xlim()
                single = len(nyquist_lines) == 1
                for nyq, color in nyquist_lines.items():
                    if not (xl <= nyq <= xr):
                        continue
                    line_color = "#5E5E5E" if single else color
                    ax.axvline(
                        nyq, color=line_color, linestyle=(0, (5, 3)),
                        linewidth=1.1, alpha=0.7, zorder=2,
                    )
                    label = "f_nyq"
                    ax.text(
                        nyq, 1.01, label,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=line_color,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor=line_color, linewidth=0.8, alpha=0.9),
                        zorder=12,
                    )
            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)
        self._log_lorentz_fit_summary()
    def _draw_lorentz_compact_box(self, ax, window, items, psd_curves):
        """Render a single info box listing per-run Lorentz fits for one window.

        items: list of (model_pk, f_peak_vis, color, zeta, f0, sigma_zeta, group_name)
        Used when len(items) >= LORENTZ_COMPACT_THRESHOLD to avoid callout clutter.
        """
        from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
        f_lo, f_hi = window
        fontsize = 9
        header = TextArea(
            f"Lorentz fit [{f_lo:.1f}\u2013{f_hi:.1f} Hz]",
            textprops=dict(color="#1A1A1A", fontsize=fontsize,
                           fontweight="bold", family="Montserrat"),
        )
        children = [header]
        for _mp, _fpv, color, zf, f0, sz, gname in items:
            label = gname.upper() if gname else ""
            if sz is not None and np.isfinite(sz):
                txt = f"  {label}: $f_0$={f0:.2f}Hz $\\zeta$={zf:.3f}\u00b1{sz:.3f}"
            else:
                txt = f"  {label}: $f_0$={f0:.2f}Hz $\\zeta$={zf:.3f}"
            children.append(TextArea(
                txt,
                textprops=dict(color=color, fontsize=fontsize,
                               fontweight="bold", family="Montserrat"),
            ))
        vpacker = VPacker(children=children, pad=2, sep=2)
        if psd_curves:
            xs_data = np.concatenate([np.asarray(f) for _, f, _ in psd_curves])
            ys_data = np.concatenate([np.asarray(p) for _, _, p in psd_curves])
            try:
                trans = ax.transAxes.inverted().transform(
                    ax.transData.transform(np.column_stack([xs_data, ys_data]))
                )
                xs_ax = trans[:, 0]
                ys_ax = trans[:, 1]
                in_view = (xs_ax >= 0) & (xs_ax <= 1) & (ys_ax >= 0) & (ys_ax <= 1)
                xs_ax = xs_ax[in_view]
                ys_ax = ys_ax[in_view]
            except Exception:
                xs_ax = ys_ax = np.array([])
        else:
            xs_ax = ys_ax = np.array([])
        legend = ax.get_legend()
        legend_corner = None
        if legend is not None:
            try:
                bbox = legend.get_window_extent()
                inv = ax.transAxes.inverted()
                (lx0, ly0) = inv.transform((bbox.x0, bbox.y0))
                (lx1, ly1) = inv.transform((bbox.x1, bbox.y1))
                cx_ax = 0.5 * (lx0 + lx1)
                cy_ax = 0.5 * (ly0 + ly1)
                legend_corner = (
                    "left" if cx_ax < 0.5 else "right",
                    "bottom" if cy_ax < 0.5 else "top",
                )
            except Exception:
                pass
        w_frac, h_frac = 0.32, 0.38
        def _density(corner):
            ha, va = corner
            x_min = 0.0 if ha == "left" else (1.0 - w_frac)
            x_max = x_min + w_frac
            y_max = 1.0 if va == "top" else h_frac
            y_min = y_max - h_frac
            if xs_ax.size == 0:
                return 0
            return int(((xs_ax >= x_min) & (xs_ax <= x_max)
                        & (ys_ax >= y_min) & (ys_ax <= y_max)).sum())
        corners = [c for c in self._INFO_CORNER_XY if c != legend_corner
                   and c[0] in ("left", "right") and c[1] in ("top", "bottom")]
        corners.sort(key=_density)
        halign, valign = corners[0]
        x_anchor, y_anchor = self._INFO_CORNER_XY[(halign, valign)]
        ab = AnnotationBbox(
            vpacker,
            xy=(x_anchor, y_anchor),
            xycoords="axes fraction",
            box_alignment=(0.0 if halign == "left" else 1.0,
                           1.0 if valign == "top" else 0.0),
            bboxprops=dict(boxstyle="round,pad=0", facecolor="white",
                           alpha=0.92, edgecolor="#3C3C3C", linewidth=1.4),
            frameon=True, pad=0.3,
        )
        ab.set_zorder(11)
        ax.add_artist(ab)
    def _record_lorentz_fit(self, plot, group, channel, f_lo, f_hi,
                            f0_fit, zeta, sigma_zeta, r_squared, saturated,
                            failed=False):
        records = getattr(self, "_lorentz_fit_records", None)
        if records is None:
            records = []
            self._lorentz_fit_records = records
        records.append({
            "plot": plot, "group": group, "channel": channel,
            "f_lo": f_lo, "f_hi": f_hi,
            "f0_fit": f0_fit, "zeta": zeta,
            "sigma_zeta": sigma_zeta, "r_squared": r_squared,
            "saturated": saturated, "failed": failed,
        })
    def _log_lorentz_fit_summary(self):
        records = getattr(self, "_lorentz_fit_records", None)
        if not records:
            return
        header = ("%-32s %-10s %-14s %15s %7s %7s %7s %5s %s"
                  % ("Plot", "Run", "Channel",
                     "window (Hz)", "f0_fit", "zeta", "sig_z", "R^2", "Notes"))
        log.info("Lorentz fit summary (%d entries):", len(records))
        log.info(header)
        log.info("-" * len(header))
        for r in records:
            win_str = f"[{r['f_lo']:5.2f},{r['f_hi']:5.2f}]"
            if r["failed"]:
                log.info("%-32s %-10s %-14s %15s %7s %7s %7s %5s %s",
                         r["plot"][:32], r["group"][:10], r["channel"][:14],
                         win_str, "-", "-", "-", "-", "failed")
                continue
            sig = r["sigma_zeta"]
            sig_str = (f"{sig:7.4f}" if (sig is not None and np.isfinite(sig))
                       else "   nan ")
            r2 = r["r_squared"]
            r2_str = f"{r2:5.2f}" if (r2 is not None and np.isfinite(r2)) else "  nan"
            notes = "saturated" if r["saturated"] else ""
            log.info("%-32s %-10s %-14s %15s %7.3f %7.4f %s %s %s",
                     r["plot"][:32], r["group"][:10], r["channel"][:14],
                     win_str, r["f0_fit"], r["zeta"], sig_str, r2_str, notes)
    def generate_histogram_plots(self):
        self._ensure_preprocessed()
        plots = self._get_plot_group(3)
        if not plots:
            return
        plot_iter = plots if self.verbose else _tqdm(plots, desc="Histogram", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name   = plot_def.name
            channel     = plot_def.channel
            axis_limits = plot_def.axis_limits
            log_scale   = plot_def.log_scale
            markers     = plot_def.markers
            gate_spec   = plot_def.gate
            reference_lines = plot_def.reference_lines
            if self.verbose:
                log.debug("Creating histogram plot: %s (%s)", plot_name, channel)
            filename = self._sanitize_plot_filename("histogram", plot_name)
            figsize = self._resolve_plot_figsize(filename, self.histogram_FIGSIZE)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(
                datafunctions.add_units_to_label(channel, self.units_map),
                fontsize=13, fontweight="bold",
            )
            ax.set_ylabel("Time (s)", fontsize=13, fontweight="bold")
            all_values = []
            for run in self.runs:
                run_name = run["name"].lower()
                df = self.run_data.get(run_name)
                if df is None or channel not in df.columns:
                    continue
                if gate_spec is not None:
                    df = datafunctions.apply_gate_to_dataframe(df, gate_spec)
                    if df is None or df.empty:
                        continue
                vals = df[channel].dropna()
                if not vals.empty:
                    all_values.append(vals.values)
            if not all_values:
                log.warning(
                    "Histogram '%s': no valid data for '%s'. Plot not saved.",
                    plot_name, channel,
                )
                plt.close(fig)
                continue
            combined = np.concatenate(all_values)
            bins = datafunctions.compute_nice_histogram_bins(combined, num_bins=30)
            has_x_limits, has_y_limits = self._apply_2d_axis_limits(
                ax, axis_limits, log_scale_y=log_scale, y_floor=1e-6,
            )
            if has_x_limits and axis_limits[0][0] is not None and axis_limits[0][1] is not None:
                bins = datafunctions.compute_equal_width_bins_in_limits(
                    axis_limits[0][0], axis_limits[0][1], bins,
                )
            hist_data, hist_weights, hist_colors, hist_labels = [], [], [], []
            for run in self.runs:
                run_name = run["name"].lower()
                df = self.run_data.get(run_name)
                if df is None or channel not in df.columns:
                    continue
                if gate_spec is not None:
                    df = datafunctions.apply_gate_to_dataframe(df, gate_spec)
                    if df is None or df.empty:
                        continue
                data = df[channel].dropna()
                if data.empty:
                    continue
                dt = 1.0 / self._run_fs(run_name)
                hist_data.append(data.to_numpy())
                hist_weights.append(np.full(len(data), dt))
                hist_colors.append(run["color"])
                hist_labels.append(run["name"].upper())
            if hist_data:
                ax.hist(
                    hist_data, bins=bins, weights=hist_weights,
                    alpha=0.72, color=hist_colors, label=hist_labels,
                    edgecolor="white", linewidth=0.8,
                    log=log_scale, stacked=False, histtype="bar", rwidth=0.9,
                )
            if len(bins) > 1:
                max_major_ticks = 8
                major_step = max(1, int(np.ceil((len(bins) - 1) / (max_major_ticks - 1))))
                major_ticks = bins[::major_step]
                if not np.isclose(major_ticks[-1], bins[-1]):
                    major_ticks = np.append(major_ticks, bins[-1])
                ax.set_xticks(major_ticks)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f"{x:.4g}")
                )
                if len(bins) <= 31:
                    ax.set_xticks(bins, minor=True)
                    ax.grid(True, which="minor", axis="x", **self.GRID_STYLE["minor"])
                ax.grid(True, which="major", axis="x", **self.GRID_STYLE["major"])
            self._add_axis_edge_padding(
                ax,
                x_pad_ratio=(0 if has_x_limits else 0.02),
                y_pad_ratio=(0 if has_y_limits else 0.03),
            )
            self._apply_grid(ax, which="major", axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            yl, yr = ax.get_ylim()
            if yl <= 0 <= yr:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8)
            self._add_standard_legend(ax, loc="best")
            self._draw_horizontal_reference_lines(ax, reference_lines)
            if gate_spec is not None:
                gate_text = datafunctions.format_gate_text(gate_spec)
                if gate_text:
                    legend_obj = ax.get_legend()
                    self._display_gate_info(ax, gate_text, legend=legend_obj)
            self._draw_static_markers(ax, markers)
            plt.tight_layout(pad=0.25)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi, pad_inches=0.05, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)

class HeatmapMixin:

    def generate_heatmap_plots(self):
        self._ensure_preprocessed()
        plots = self._get_plot_group(6)
        if not plots:
            return
        plot_iter = plots if self.verbose else _tqdm(plots, desc="Heatmap", unit="plot", leave=True)
        for plot_def in plot_iter:
            plot_name   = plot_def.name
            x_channel   = plot_def.x_channel
            y_channel   = plot_def.y_channel
            z_channel   = plot_def.z_channel
            agg         = plot_def.aggregation
            bins        = plot_def.bins
            axis_limits = plot_def.axis_limits
            cmap        = plot_def.cmap
            z_limits    = plot_def.z_limits
            gate_spec   = plot_def.gate
            markers     = plot_def.markers
            min_count   = plot_def.min_count
            if self.verbose:
                log.debug("Creating heatmap plot: %s", plot_name)
            run_frames = []
            for run in self.runs:
                rn = run["name"].lower()
                df = self._get_filtered_run_dataframe(rn, gate_spec)
                if df is None or x_channel not in df.columns or y_channel not in df.columns:
                    continue
                if z_channel is not None and z_channel not in df.columns:
                    continue
                run_frames.append((run, df))
            if not run_frames:
                log.warning("Heatmap '%s': no usable runs. Skipping.", plot_name)
                continue
            ncols = len(run_frames)
            base_w = self.histogram_FIGSIZE[0]
            fig, axes = plt.subplots(
                1, ncols,
                figsize=(base_w * 0.9 * ncols, self.histogram_FIGSIZE[1]),
                squeeze=False,
            )
            axes = axes[0]
            xs_all = np.concatenate([
                pd.to_numeric(df[x_channel], errors="coerce").dropna().to_numpy()
                for _, df in run_frames
            ])
            ys_all = np.concatenate([
                pd.to_numeric(df[y_channel], errors="coerce").dropna().to_numpy()
                for _, df in run_frames
            ])
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
                                "median": np.median, "std": np.std,
                                "max": np.max, "min": np.min,
                            }[agg]
                            for (i_, j_), values in buckets.items():
                                if len(values) >= min_count:
                                    grid[i_, j_] = reducer(values)
                    grid = np.where(counts >= min_count, grid, np.nan)
                all_grids.append((run, grid, counts))
            zs_combined = np.concatenate([g.ravel() for _, g, _ in all_grids])
            zs_combined = zs_combined[np.isfinite(zs_combined)]
            if z_limits and (z_limits[0] is not None or z_limits[1] is not None):
                z_min = z_limits[0] if z_limits[0] is not None else float(np.nanmin(zs_combined)) if len(zs_combined) else 0.0
                z_max = z_limits[1] if z_limits[1] is not None else float(np.nanmax(zs_combined)) if len(zs_combined) else 1.0
            elif len(zs_combined):
                z_min, z_max = float(np.nanmin(zs_combined)), float(np.nanmax(zs_combined))
            else:
                z_min, z_max = 0.0, 1.0
            for ax, (run, grid, counts) in zip(axes, all_grids):
                im = ax.imshow(
                    grid.T, origin="lower", aspect="auto",
                    extent=(x_lo, x_hi, y_lo, y_hi),
                    cmap=cmap, vmin=z_min, vmax=z_max,
                    interpolation="nearest",
                )
                ax.set_title(run["name"].upper(), fontsize=11, fontweight="bold", color=run["color"])
                ax.set_xlabel(datafunctions.add_units_to_label(x_channel, self.units_map),
                              fontweight="bold")
                ax.set_ylabel(datafunctions.add_units_to_label(y_channel, self.units_map),
                              fontweight="bold")
                self._apply_grid(ax, which="major")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                self._draw_static_markers(ax, markers, x_clip=False)
            cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
            if z_channel is None:
                cbar.set_label("Count", fontweight="bold")
            else:
                cbar.set_label(
                    f"{agg}({datafunctions.add_units_to_label(z_channel, self.units_map)})",
                    fontweight="bold",
                )
            filename = self._sanitize_plot_filename("heatmap", plot_name)
            fig.suptitle(plot_name, fontweight="bold", fontsize=13)
            fig.savefig(self.plots_dir / filename, dpi=self.output_dpi,
                        pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                log.debug("Saved: %s", filename)
