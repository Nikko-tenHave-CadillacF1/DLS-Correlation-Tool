"""Waveform plot generator mixin for DataPlotter."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import datafunctions

try:
    from tqdm import tqdm as _tqdm_raw
    def _tqdm(it, **kw): return _tqdm_raw(it, file=__import__('sys').stderr, dynamic_ncols=True, **kw)
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable


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
            print(
                f"[WARNING][DataPlotter] Waveform row '{row_name}': dual-channel row expects axis limits as "
                f"((y1_min,y1_max),(y2_min,y2_max)). Applying provided limits to primary channel only."
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
                print(f"[WARNING][DataPlotter] Waveform row {missing_name} missing from all runs. Skipping row.")
                continue

            if p_count == 0 and secondary is not None and s_count > 0:
                print(
                    f"[WARNING][DataPlotter] Waveform row primary channel '{primary}' missing in all runs; "
                    f"using '{secondary}' as single-channel row."
                )
                primary, secondary = secondary, None
                p_count = s_count
                s_count = 0

            if p_count < len(self.runs):
                print(
                    f"[WARNING][DataPlotter] Waveform channel '{primary}' present in "
                    f"{p_count}/{len(self.runs)} runs. Plotting available runs only."
                )

            if secondary is not None:
                if s_count == 0:
                    print(
                        f"[WARNING][DataPlotter] Waveform secondary channel '{secondary}' "
                        "missing from all runs; rendering row as single-channel."
                    )
                    secondary = None
                elif s_count < len(self.runs):
                    print(
                        f"[WARNING][DataPlotter] Waveform secondary channel '{secondary}' present in "
                        f"{s_count}/{len(self.runs)} runs. Plotting available runs only."
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
            # Unpack with defaults for optional trailing items
            _d = list(plot_def) + [None] * (9 - len(plot_def))
            plot_name, channels, axis_limits, ref_lines = _d[:4]
            subplot_heights = _d[4]
            x_limits        = _d[5]
            x_channel       = _d[6] if isinstance(_d[6], str) and _d[6].strip() else "sLap"
            highlight_zones = _d[7]
            normalise       = bool(_d[8]) if _d[8] else False

            if self.verbose:
                print(f"Creating waveform plot: {plot_name}")

            prepared_rows, avail_heights = self._prepare_waveform_channels(
                channels, axis_limits, ref_lines, subplot_heights
            )

            if not prepared_rows:
                print(f"[WARNING][DataPlotter] No valid channels for '{plot_name}' — skipping.")
                continue

            filename = self._sanitize_plot_filename("waveform", plot_name)
            min_height = 1.6 * sum(avail_heights)
            figsize = self._resolve_plot_figsize(filename, self.waveform_figsize, min_height=min_height)

            fig, axes = plt.subplots(
                len(prepared_rows),
                1,
                figsize=figsize,
                sharex=True,
                squeeze=False,
                gridspec_kw={"height_ratios": avail_heights},
            )
            axes = axes.flatten()
            plotted_runs = set()

            # Resolve the x-axis channel, falling back gracefully if unavailable
            loaded_run_names = [r["name"].lower() for r in self.runs if r["name"].lower() in self.run_data]
            x_channel_available = x_channel and all(
                x_channel in self.run_data[rn].columns for rn in loaded_run_names
            )
            if not x_channel_available:
                fallback = "sLap" if x_channel != "sLap" else None
                if fallback and all(fallback in self.run_data[rn].columns for rn in loaded_run_names):
                    print(
                        f"[WARNING][DataPlotter] Waveform '{plot_name}': x_channel '{x_channel}' "
                        f"not available in all runs. Falling back to 'sLap'."
                    )
                    x_channel = "sLap"
                    x_channel_available = True
                else:
                    print(
                        f"[WARNING][DataPlotter] Waveform '{plot_name}': x_channel '{x_channel}' "
                        f"not available. Using row index."
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
                ax = axes[idx]
                ch_primary = row["primary"]
                ch_secondary = row["secondary"]
                ax_right = (ax.twinx() if ch_secondary is not None else None) if not normalise else None

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
                ax.grid(True, axis="y", alpha=0.28, linewidth=0.45)

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
                                alpha=0.25, color=shade_color, zorder=0, linewidth=0,
                            )

                if idx < len(prepared_rows) - 1:
                    ax.tick_params(labelbottom=False)

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
                    ax.grid(True, which="major", axis="x", alpha=0.45, linewidth=0.5)
                    ax.grid(True, which="minor", axis="x", alpha=0.225, linewidth=0.3)

            # Legend above subplots
            run_handles = [
                Line2D([0], [0], color=run["color"], linewidth=2.0)
                for run in self.runs if run["name"].lower() in plotted_runs
            ]
            run_labels = [
                run["name"].upper()
                for run in self.runs if run["name"].lower() in plotted_runs
            ]
            self._add_waveform_figure_legend(fig, run_handles, run_labels)

            plt.tight_layout(pad=0.3, h_pad=0.0, rect=(0, 0, 1, 0.95))
            fig.savefig(self.plots_dir / filename, dpi=300, pad_inches=0.15, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                print(f"  Saved: {filename}")
