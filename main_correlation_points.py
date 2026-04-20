"""Generate a simple, user-facing summary of main correlation differences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

import datafunctions


# ================================================================
# DATA CONTAINERS (DATACLASSES)
# ================================================================
# Structured containers for waveform and scatter metric rows.

@dataclass
class WaveformMetric:
    """Container for one waveform comparison metric row."""

    plot_name: str
    channel: str
    baseline_run: str
    compare_run: str
    mean_abs_diff: Optional[float]
    p95_abs_diff: Optional[float]
    corr: Optional[float]
    confidence_note: str
    notes: str


@dataclass
class ScatterMetric:
    """Container for one scatter fit comparison metric row."""

    plot_name: str
    segment: str
    x_var: str
    y_var: str
    baseline_run: str
    compare_run: str
    slope_baseline: Optional[float]
    slope_compare: Optional[float]
    slope_pct_delta: Optional[float]
    intercept_baseline: Optional[float]
    intercept_compare: Optional[float]
    intercept_delta: Optional[float]
    sample_count_baseline: int
    sample_count_compare: int
    confidence_note: str
    notes: str


# ================================================================
# UTILITY FUNCTIONS
# ================================================================
# Helper functions for correlations, formatting, and data extraction.

def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Return correlation when valid, otherwise None."""
    if a.size < 2 or b.size < 2:
        return None
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return None
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else None


def _extract_waveform_channels(channels_spec) -> List[str]:
    """Flatten waveform row specs into plain channel names."""
    channels = []
    for row in channels_spec:
        if isinstance(row, str):
            channels.append(row)
        elif isinstance(row, (list, tuple)):
            for item in row:
                if isinstance(item, str):
                    channels.append(item)
    return channels


def _format_bound(v) -> str:
    """Format a segment bound for report text."""
    if v is None:
        return "auto"
    try:
        return f"{float(v):g}"
    except Exception:
        return str(v)


def _segment_label(fit_def, x_var: str, y_var: str) -> str:
    """Return a readable segment label for one fit definition."""
    if not isinstance(fit_def, (list, tuple)) or len(fit_def) != 3:
        return "Segment"

    axis, vmin, vmax = fit_def
    axis_name = x_var if str(axis).lower() == "x" else y_var if str(axis).lower() == "y" else str(axis)
    return f"{axis_name}: {_format_bound(vmin)} to {_format_bound(vmax)}"


# ================================================================
# WAVEFORM METRICS COMPUTATION
# ================================================================
# Functions for computing and comparing waveform channel statistics.

def _align_channel_series(
    baseline_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    channel: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align one channel across runs using sLap interpolation when available."""
    if channel not in baseline_df.columns or channel not in compare_df.columns:
        return np.array([]), np.array([])

    if "sLap" in baseline_df.columns and "sLap" in compare_df.columns:
        b = pd.DataFrame(
            {
                "sLap": pd.to_numeric(baseline_df["sLap"], errors="coerce"),
                "v": pd.to_numeric(baseline_df[channel], errors="coerce"),
            }
        ).dropna()
        c = pd.DataFrame(
            {
                "sLap": pd.to_numeric(compare_df["sLap"], errors="coerce"),
                "v": pd.to_numeric(compare_df[channel], errors="coerce"),
            }
        ).dropna()

        if b.empty or c.empty:
            return np.array([]), np.array([])

        b = b[b["sLap"] >= 0].groupby("sLap", as_index=False)["v"].mean().sort_values("sLap")
        c = c[c["sLap"] >= 0].groupby("sLap", as_index=False)["v"].mean().sort_values("sLap")
        if len(b) < 5 or len(c) < 5:
            return np.array([]), np.array([])

        lo = max(float(b["sLap"].min()), float(c["sLap"].min()))
        hi = min(float(b["sLap"].max()), float(c["sLap"].max()))
        if hi <= lo:
            return np.array([]), np.array([])

        mask = (b["sLap"] >= lo) & (b["sLap"] <= hi)
        b_s = b.loc[mask, "sLap"].to_numpy(dtype=float)
        b_v = b.loc[mask, "v"].to_numpy(dtype=float)
        if b_s.size < 5:
            return np.array([]), np.array([])

        c_v_interp = np.interp(
            b_s,
            c["sLap"].to_numpy(dtype=float),
            c["v"].to_numpy(dtype=float),
        )
        valid = np.isfinite(b_v) & np.isfinite(c_v_interp)
        return b_v[valid], c_v_interp[valid]

    b = pd.to_numeric(baseline_df[channel], errors="coerce")
    c = pd.to_numeric(compare_df[channel], errors="coerce")
    n = min(len(b), len(c))
    if n < 5:
        return np.array([]), np.array([])
    b = b.iloc[:n].to_numpy(dtype=float)
    c = c.iloc[:n].to_numpy(dtype=float)
    valid = np.isfinite(b) & np.isfinite(c)
    return b[valid], c[valid]


def _compute_waveform_metrics(
    runs,
    run_data: Dict[str, pd.DataFrame],
    waveform_defs,
    corr_check_threshold: float,
    low_sample_threshold: int,
) -> Tuple[List[WaveformMetric], List[str]]:
    """Compute waveform comparison metrics for all configured channels."""
    items: List[WaveformMetric] = []
    coverage_notes: List[str] = []
    if not runs:
        return items, coverage_notes

    baseline = runs[0]["name"].lower()
    if baseline not in run_data:
        coverage_notes.append(f"Baseline run '{baseline.upper()}' not loaded.")
        return items, coverage_notes

    base_df = run_data[baseline]
    for plot_def in waveform_defs or []:
        if len(plot_def) < 2:
            continue
        plot_name = str(plot_def[0])
        channels = _extract_waveform_channels(plot_def[1])
        for channel in channels:
            for run in runs[1:]:
                rn = run["name"].lower()
                if rn not in run_data:
                    coverage_notes.append(f"{plot_name}: run '{rn.upper()}' not loaded.")
                    continue

                cmp_df = run_data[rn]
                b_vals, c_vals = _align_channel_series(base_df, cmp_df, channel)
                if b_vals.size == 0:
                    coverage_notes.append(f"{plot_name}: '{channel}' unavailable/alignment failed for {rn.upper()}.")
                    continue

                diff = np.abs(b_vals - c_vals)
                mad = float(np.mean(diff))
                p95 = float(np.percentile(diff, 95))
                corr = _safe_corr(b_vals, c_vals)

                confidence = "OK"
                notes = ""
                if b_vals.size < low_sample_threshold:
                    confidence = "LOW SAMPLES"
                if corr is not None and corr < corr_check_threshold:
                    notes = "CHECK: low correlation"

                items.append(
                    WaveformMetric(
                        plot_name=plot_name,
                        channel=channel,
                        baseline_run=baseline.upper(),
                        compare_run=rn.upper(),
                        mean_abs_diff=mad,
                        p95_abs_diff=p95,
                        corr=corr,
                        confidence_note=confidence,
                        notes=notes,
                    )
                )
    return items, coverage_notes


# ================================================================
# SCATTER METRICS COMPUTATION
# ================================================================
# Functions for computing and comparing scatter fit slope/intercepts.

def _prepare_scatter_xy(df: pd.DataFrame, x_var: str, y_var: str):
    """Prepare aligned scatter x/y series from one run dataframe."""
    if x_var not in df.columns or y_var not in df.columns:
        return None, None
    x = pd.to_numeric(df[x_var], errors="coerce").dropna()
    y = pd.to_numeric(df[y_var], errors="coerce").reindex(x.index).dropna()
    x = x.reindex(y.index)
    return x, y


def _compute_segment_fits_for_run(
    df: pd.DataFrame,
    run_name: str,
    plot_name: str,
    x_var: str,
    y_var: str,
    best_fit,
    gate_spec,
):
    """Compute fit values per segment label for one run."""
    out = {}
    x, y = _prepare_scatter_xy(df, x_var, y_var)
    if x is None or y is None or len(x) < 2:
        return out

    x, y = datafunctions.apply_scatter_gate(
        df,
        x,
        y,
        gate_spec,
        plot_name=plot_name,
        run_name=run_name,
    )
    if len(x) < 2:
        return out

    if best_fit in (None, 0):
        return out

    if best_fit in (1, 2):
        try:
            slope, interc, _, _, _ = linregress(x.to_numpy(dtype=float), y.to_numpy(dtype=float))
        except ValueError:
            return out
        out["Overall"] = {
            "slope": float(slope),
            "intercept": float(interc),
            "n": int(len(x)),
        }
        return out

    if not (isinstance(best_fit, (list, tuple)) and best_fit and isinstance(best_fit[0], (list, tuple))):
        return out

    fit_condition_data = datafunctions.build_fit_condition_data(
        df,
        x.index,
        best_fit,
        plot_name=plot_name,
        run_name=run_name,
    )
    x_vals = x.to_numpy(dtype=float)
    y_vals = y.to_numpy(dtype=float)

    for fit_def in best_fit:
        seg_label = _segment_label(fit_def, x_var, y_var)
        info = datafunctions.build_multi_fit_mask(
            fit_def,
            x_vals,
            y_vals,
            fit_condition_data=fit_condition_data,
            x_var=x_var,
            y_var=y_var,
        )
        mask = info.get("mask")
        if mask is None or int(np.sum(mask)) < 2:
            out[seg_label] = {"slope": None, "intercept": None, "n": int(np.sum(mask)) if mask is not None else 0}
            continue

        xb = x_vals[mask]
        yb = y_vals[mask]
        try:
            slope, interc, _, _, _ = linregress(xb, yb)
            out[seg_label] = {
                "slope": float(slope),
                "intercept": float(interc),
                "n": int(mask.sum()),
            }
        except ValueError:
            out[seg_label] = {"slope": None, "intercept": None, "n": int(mask.sum())}
    return out


def _compute_scatter_metrics(
    runs,
    run_data: Dict[str, pd.DataFrame],
    scatter_defs,
    low_sample_threshold: int,
    slope_delta_check_pct: float,
) -> Tuple[List[ScatterMetric], List[str]]:
    """Compute scatter fit comparison metrics vs baseline."""
    items: List[ScatterMetric] = []
    coverage_notes: List[str] = []
    if not runs:
        return items, coverage_notes

    baseline = runs[0]["name"].lower()
    if baseline not in run_data:
        coverage_notes.append(f"Baseline run '{baseline.upper()}' not loaded.")
        return items, coverage_notes

    for plot_def in scatter_defs or []:
        gate_spec = None
        
        if len(plot_def) == 4:
            plot_name, (x_var, y_var), _, best_fit = plot_def
        elif len(plot_def) == 5:
            plot_name, (x_var, y_var), _, best_fit, item5 = plot_def
            # 5th item can be gate_spec or show_equations (boolean)
            if isinstance(item5, bool):
                # show_equations parameter, no gate spec
                gate_spec = None
            else:
                # Assume it's a gate spec
                gate_spec = item5
        elif len(plot_def) == 6:
            plot_name, (x_var, y_var), _, best_fit, item5, item6 = plot_def
            # For 6 items, check which items are booleans
            if isinstance(item5, bool) and isinstance(item6, bool):
                # Format: [name, (x,y), limits, best_fit, show_equations, show_error]
                gate_spec = None
            elif isinstance(item6, bool):
                # Format: [name, (x,y), limits, best_fit, gate_spec, show_equations]
                gate_spec = item5
            elif isinstance(item5, bool):
                # Format: [name, (x,y), limits, best_fit, show_equations, gate_spec]
                gate_spec = item6
            else:
                # Both are not booleans - try to use item5 as gate_spec
                gate_spec = item5
        elif len(plot_def) >= 7:
            plot_name, (x_var, y_var), _, best_fit, item5, item6, item7 = plot_def[:7]
            # Format: [name, (x,y), limits, best_fit, gate_spec, show_equations, show_error]
            # Skip booleans at the end (show_equations, show_error)
            # item5 should be gate_spec
            if isinstance(item5, bool):
                gate_spec = None
            else:
                gate_spec = item5
        else:
            continue

        if best_fit is None:
            best_fit = 0
        if best_fit == 0:
            continue

        base_fit = _compute_segment_fits_for_run(
            run_data[baseline],
            baseline,
            plot_name,
            x_var,
            y_var,
            best_fit,
            gate_spec,
        )
        if not base_fit:
            coverage_notes.append(f"{plot_name}: no baseline fit segments available.")
            continue

        for run in runs[1:]:
            rn = run["name"].lower()
            if rn not in run_data:
                coverage_notes.append(f"{plot_name}: run '{rn.upper()}' not loaded.")
                continue

            cmp_fit = _compute_segment_fits_for_run(
                run_data[rn],
                rn,
                plot_name,
                x_var,
                y_var,
                best_fit,
                gate_spec,
            )

            all_segments = sorted(set(base_fit.keys()) | set(cmp_fit.keys()))
            for seg in all_segments:
                b = base_fit.get(seg, {"slope": None, "intercept": None, "n": 0})
                c = cmp_fit.get(seg, {"slope": None, "intercept": None, "n": 0})
                slope_pct_delta = None
                if b["slope"] not in (None, 0) and c["slope"] is not None:
                    slope_pct_delta = ((c["slope"] - b["slope"]) / b["slope"]) * 100.0
                intercept_delta = None
                if b["intercept"] is not None and c["intercept"] is not None:
                    intercept_delta = c["intercept"] - b["intercept"]

                confidence = "OK"
                notes = ""
                if b["n"] < low_sample_threshold or c["n"] < low_sample_threshold:
                    confidence = "LOW SAMPLES"
                if slope_pct_delta is not None and abs(slope_pct_delta) > slope_delta_check_pct:
                    notes = "CHECK: slope delta above threshold"

                items.append(
                    ScatterMetric(
                        plot_name=str(plot_name),
                        segment=str(seg),
                        x_var=str(x_var),
                        y_var=str(y_var),
                        baseline_run=baseline.upper(),
                        compare_run=rn.upper(),
                        slope_baseline=b["slope"],
                        slope_compare=c["slope"],
                        slope_pct_delta=slope_pct_delta,
                        intercept_baseline=b["intercept"],
                        intercept_compare=c["intercept"],
                        intercept_delta=intercept_delta,
                        sample_count_baseline=int(b["n"]),
                        sample_count_compare=int(c["n"]),
                        confidence_note=confidence,
                        notes=notes,
                    )
                )

    return items, coverage_notes


def _fmt(v, fmt: str = ".3f") -> str:
    """Format floats safely."""
    if v is None:
        return "n/a"
    try:
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return "n/a"
        return format(float(v), fmt)
    except Exception:
        return str(v)


# ================================================================
# REPORT GENERATION & FORMATTING
# ================================================================
# Functions for building suggested snippets and writing reports.

def _build_suggested_snippets(w_items: List[WaveformMetric], s_items: List[ScatterMetric]) -> List[str]:
    """Build plain-language snippets users can paste into reports."""
    snippets: List[str] = []

    wave_checks = [w for w in w_items if w.notes or w.confidence_note != "OK"]
    for w in wave_checks[:5]:
        snippets.append(
            f"- {w.plot_name} / {w.channel}: {w.compare_run} vs {w.baseline_run} "
            f"(corr={_fmt(w.corr)}, p95 diff={_fmt(w.p95_abs_diff)})."
        )

    scatter_checks = [s for s in s_items if s.notes or s.confidence_note != "OK"]
    for s in scatter_checks[:5]:
        snippets.append(
            f"- {s.plot_name} [{s.segment}]: {s.compare_run} slope delta vs {s.baseline_run} = "
            f"{_fmt(s.slope_pct_delta, '.1f')}% (n={s.sample_count_compare})."
        )

    if not snippets:
        snippets.append("- No major flagged checks; review highlighted plots for engineering context.")
    return snippets


def write_main_correlation_points_report(
    output_dir: Path,
    runs,
    waveform_items: List[WaveformMetric],
    scatter_items: List[ScatterMetric],
    coverage_notes: List[str],
    *,
    include_csv: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """Write the text report (and optional CSV) to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "main_correlation_points.txt"
    csv_path: Optional[Path] = output_dir / "main_correlation_points.csv" if include_csv else None

    baseline = runs[0]["name"].upper() if runs else "N/A"
    compares = ", ".join([r["name"].upper() for r in runs[1:]]) if len(runs) > 1 else "N/A"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_wave_checks = sum(1 for w in waveform_items if w.notes or w.confidence_note != "OK")
    summary_scatter_checks = sum(1 for s in scatter_items if s.notes or s.confidence_note != "OK")
    summary_low_samples = sum(1 for w in waveform_items if w.confidence_note == "LOW SAMPLES") + sum(
        1 for s in scatter_items if s.confidence_note == "LOW SAMPLES"
    )

    lines: List[str] = []
    lines.append("Main Correlation Points")
    lines.append("=" * 72)
    lines.append(f"Generated: {now}")
    lines.append(f"Baseline run: {baseline}")
    lines.append(f"Comparison runs: {compares}")
    lines.append("")
    lines.append("Executive Summary")
    lines.append("-" * len("Executive Summary"))
    lines.append(f"- Waveform comparisons with CHECK/LOW SAMPLES: {summary_wave_checks}")
    lines.append(f"- Scatter fit comparisons with CHECK/LOW SAMPLES: {summary_scatter_checks}")
    lines.append(f"- Total LOW SAMPLES entries: {summary_low_samples}")
    lines.append("")
    lines.append("Waveform Differences")
    lines.append("-" * len("Waveform Differences"))
    if not waveform_items:
        lines.append("None")
    else:
        for w in waveform_items:
            lines.append(
                f"- {w.plot_name} | {w.channel} | {w.compare_run} vs {w.baseline_run} | "
                f"mean_abs={_fmt(w.mean_abs_diff)} | p95_abs={_fmt(w.p95_abs_diff)} | "
                f"corr={_fmt(w.corr)} | {w.confidence_note}{' | ' + w.notes if w.notes else ''}"
            )

    lines.append("")
    lines.append("Scatter Trendline Differences")
    lines.append("-" * len("Scatter Trendline Differences"))
    if not scatter_items:
        lines.append("None")
    else:
        for s in scatter_items:
            lines.append(
                f"- {s.plot_name} | {s.segment} | {s.compare_run} vs {s.baseline_run} | "
                f"slope_b={_fmt(s.slope_baseline)} slope_c={_fmt(s.slope_compare)} "
                f"delta={_fmt(s.slope_pct_delta, '.1f')}% | "
                f"int_b={_fmt(s.intercept_baseline)} int_c={_fmt(s.intercept_compare)} "
                f"delta={_fmt(s.intercept_delta)} | "
                f"n_b={s.sample_count_baseline} n_c={s.sample_count_compare} | "
                f"{s.confidence_note}{' | ' + s.notes if s.notes else ''}"
            )

    lines.append("")
    lines.append("Data Coverage Notes")
    lines.append("-" * len("Data Coverage Notes"))
    if not coverage_notes:
        lines.append("None")
    else:
        lines.extend([f"- {v}" for v in coverage_notes])

    lines.append("")
    lines.append("Suggested Write-Up Snippets")
    lines.append("-" * len("Suggested Write-Up Snippets"))
    lines.extend(_build_suggested_snippets(waveform_items, scatter_items))

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if include_csv and csv_path is not None:
        csv_rows = []
        for w in waveform_items:
            csv_rows.append(
                {
                    "section": "waveform",
                    "plot_name": w.plot_name,
                    "channel_or_xy": w.channel,
                    "segment": "",
                    "baseline_run": w.baseline_run,
                    "compare_run": w.compare_run,
                    "mean_abs_diff": w.mean_abs_diff,
                    "p95_abs_diff": w.p95_abs_diff,
                    "corr": w.corr,
                    "slope_baseline": None,
                    "slope_compare": None,
                    "slope_pct_delta": None,
                    "intercept_baseline": None,
                    "intercept_compare": None,
                    "intercept_delta": None,
                    "sample_count_baseline": None,
                    "sample_count_compare": None,
                    "confidence_note": w.confidence_note,
                    "notes": w.notes,
                }
            )
        for s in scatter_items:
            csv_rows.append(
                {
                    "section": "scatter",
                    "plot_name": s.plot_name,
                    "channel_or_xy": f"{s.x_var} vs {s.y_var}",
                    "segment": s.segment,
                    "baseline_run": s.baseline_run,
                    "compare_run": s.compare_run,
                    "mean_abs_diff": None,
                    "p95_abs_diff": None,
                    "corr": None,
                    "slope_baseline": s.slope_baseline,
                    "slope_compare": s.slope_compare,
                    "slope_pct_delta": s.slope_pct_delta,
                    "intercept_baseline": s.intercept_baseline,
                    "intercept_compare": s.intercept_compare,
                    "intercept_delta": s.intercept_delta,
                    "sample_count_baseline": s.sample_count_baseline,
                    "sample_count_compare": s.sample_count_compare,
                    "confidence_note": s.confidence_note,
                    "notes": s.notes,
                }
            )
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8")

    return txt_path, csv_path


def generate_main_correlation_points_report(
    *,
    runs,
    run_data,
    plot_definitions,
    output_dir: Path,
    include_csv: bool = False,
    low_sample_threshold: int = 200,
    corr_check_threshold: float = 0.90,
    slope_delta_check_pct: float = 10.0,
) -> Tuple[Path, Optional[Path]]:
    """Build and write the main-correlation summary report."""
    waveform_defs = plot_definitions[0] if plot_definitions and len(plot_definitions) > 0 else []
    scatter_defs = plot_definitions[1] if plot_definitions and len(plot_definitions) > 1 else []

    waveform_items, w_notes = _compute_waveform_metrics(
        runs=runs,
        run_data=run_data,
        waveform_defs=waveform_defs,
        corr_check_threshold=corr_check_threshold,
        low_sample_threshold=low_sample_threshold,
    )
    scatter_items, s_notes = _compute_scatter_metrics(
        runs=runs,
        run_data=run_data,
        scatter_defs=scatter_defs,
        low_sample_threshold=low_sample_threshold,
        slope_delta_check_pct=slope_delta_check_pct,
    )

    return write_main_correlation_points_report(
        output_dir=Path(output_dir),
        runs=runs,
        waveform_items=waveform_items,
        scatter_items=scatter_items,
        coverage_notes=w_notes + s_notes,
        include_csv=include_csv,
    )
