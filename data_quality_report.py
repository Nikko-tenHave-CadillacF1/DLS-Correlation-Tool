"""Preflight data-quality checks and report writing utilities."""

from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# CHANNEL REFERENCE COLLECTION
# ================================================================

def collect_referenced_channels(plot_definitions):
    """Collect channels referenced by configured plot definitions."""
    referenced = set()

    def _extract(item):
        """Recursively collect channel names from strings/tuples/lists."""
        if isinstance(item, str):
            referenced.add(item)
            return
        if isinstance(item, (list, tuple)):
            for value in item:
                _extract(value)

    for plot_group in plot_definitions or []:
        for plot_def in plot_group or []:
            if len(plot_def) < 2:
                continue
            _extract(plot_def[1])
    return sorted(referenced)


# ================================================================
# SLAP ALIGNMENT ESTIMATION
# ================================================================
# Helper functions for estimating linear sLap mappings between runs
# using vCar similarity as a correlation proxy.

def _prepare_slap_vcar_series(df):
    """Return cleaned (sLap, vCar) arrays for alignment diagnostics."""
    if "sLap" not in df.columns or "vCar" not in df.columns:
        return None, None

    s = pd.to_numeric(df["sLap"], errors="coerce")
    v = pd.to_numeric(df["vCar"], errors="coerce")
    tmp = pd.DataFrame({"s": s, "v": v}).dropna()
    if tmp.empty:
        return None, None

    tmp = tmp[tmp["s"] >= 0].sort_values("s")
    if tmp.empty:
        return None, None

    tmp = tmp.groupby("s", as_index=False)["v"].mean()
    if len(tmp) < 50:
        return None, None

    return tmp["s"].to_numpy(dtype=float), tmp["v"].to_numpy(dtype=float)


def _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset):
    """Score one linear mapping transformed_s = oth_s*scale + offset."""
    transformed = oth_s * scale + offset
    lo = max(ref_s.min(), transformed.min())
    hi = min(ref_s.max(), transformed.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    grid = np.arange(np.ceil(lo), np.floor(hi) + 1.0, 5.0)
    if grid.size < 100:
        return None

    ref_interp = np.interp(grid, ref_s, ref_v)
    oth_interp = np.interp(grid, transformed, oth_v)

    if np.std(ref_interp) < 1e-9 or np.std(oth_interp) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ref_interp, oth_interp)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    mae = float(np.mean(np.abs(ref_interp - oth_interp)))
    return corr, mae, grid.size


def estimate_slap_alignment(runs, run_data):
    """
    Estimate linear sLap alignment between baseline run and other runs.
    Uses transformed_sLap = sLap * scale + offset and vCar similarity scoring.
    """
    lines = []
    if not runs:
        return lines

    base_name = runs[0]["name"].lower()
    if base_name not in run_data:
        return [f"{base_name.upper()}: baseline dataframe not loaded"]

    ref_s, ref_v = _prepare_slap_vcar_series(run_data[base_name])
    if ref_s is None:
        return [f"{base_name.upper()}: missing usable sLap/vCar for baseline"]

    ref_range = float(ref_s.max() - ref_s.min())
    if ref_range <= 0:
        return [f"{base_name.upper()}: invalid sLap range for baseline"]

    for run in runs[1:]:
        rn = run["name"].lower()
        if rn not in run_data:
            lines.append(f"{rn.upper()}: dataframe not loaded")
            continue

        oth_s, oth_v = _prepare_slap_vcar_series(run_data[rn])
        if oth_s is None:
            lines.append(f"{rn.upper()}: missing usable sLap/vCar")
            continue

        oth_range = float(oth_s.max() - oth_s.min())
        if oth_range <= 0:
            lines.append(f"{rn.upper()}: invalid sLap range")
            continue

        scale_guess = ref_range / oth_range
        offset_guess = float(ref_s.min() - oth_s.min() * scale_guess)
        best = None

        for scale in np.linspace(scale_guess - 0.01, scale_guess + 0.01, 25):
            for offset in np.linspace(offset_guess - 40, offset_guess + 40, 41):
                score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset)
                if score is None:
                    continue
                corr, mae, n = score
                key = (corr, -mae, n)
                if best is None or key > best["key"]:
                    best = {
                        "scale": float(scale),
                        "offset": float(offset),
                        "corr": corr,
                        "mae": mae,
                        "n": int(n),
                        "key": key,
                    }

        if best is not None:
            s0 = best["scale"]
            o0 = best["offset"]
            for scale in np.linspace(s0 - 0.002, s0 + 0.002, 21):
                for offset in np.linspace(o0 - 8, o0 + 8, 33):
                    score = _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset)
                    if score is None:
                        continue
                    corr, mae, n = score
                    key = (corr, -mae, n)
                    if key > best["key"]:
                        best = {
                            "scale": float(scale),
                            "offset": float(offset),
                            "corr": corr,
                            "mae": mae,
                            "n": int(n),
                            "key": key,
                        }

        if best is None:
            lines.append(f"{rn.upper()}: could not estimate sLap mapping")
            continue

        drift_end = (best["scale"] - 1.0) * ref_range
        lines.append(
            (
                f"{rn.upper()} vs {base_name.upper()}: "
                f"scale={best['scale']:.6f}, offset={best['offset']:+.2f} m, "
                f"end_drift_est={drift_end:+.2f} m, "
                f"vCar_corr={best['corr']:.4f}, vCar_mae={best['mae']:.2f} kph, "
                f"samples={best['n']}"
            )
        )

    return lines


# ================================================================
# DATA QUALITY ASSESSMENT
# ================================================================
# Main functions for building comprehensive quality check sections.

def build_quality_sections(runs, run_data, plot_definitions):
    """Build preflight data-quality sections from loaded run data."""
    referenced = collect_referenced_channels(plot_definitions)
    missing_by_run = []
    high_nan = []
    flatlined = []
    slap_resets = []

    for run in runs:
        run_name = run["name"].lower()
        if run_name not in run_data:
            missing_by_run.append(f"{run_name.upper()}: dataframe not loaded")
            continue

        df = run_data[run_name]
        missing = [ch for ch in referenced if ch not in df.columns]
        if missing:
            missing_by_run.append(f"{run_name.upper()}: {', '.join(missing[:20])}")

        for ch in referenced:
            if ch not in df.columns:
                continue

            series = pd.to_numeric(df[ch], errors="coerce")
            nan_ratio = float(series.isna().mean()) if len(series) else 0.0
            if nan_ratio > 0.20:
                high_nan.append(f"{run_name.upper()} {ch}: {nan_ratio:.1%} NaN")

            valid = series.dropna()
            if len(valid) > 20 and float(valid.std()) < 1e-9:
                flatlined.append(f"{run_name.upper()} {ch}")

        if "sLap" in df.columns:
            sl = pd.to_numeric(df["sLap"], errors="coerce")
            resets = int((sl.diff() < 0).sum())
            if resets > 0:
                slap_resets.append(f"{run_name.upper()}: {resets}")

    return [
        ("Missing Referenced Channels", missing_by_run),
        ("High NaN Ratios (>20%)", high_nan),
        ("Flatlined Channels", flatlined),
        ("sLap Resets", slap_resets),
        ("sLap Alignment Estimate (vCar-based)", estimate_slap_alignment(runs, run_data)),
    ]


def write_data_quality_report(plots_dir, sections):
    """Write preflight data-quality findings to the plots directory."""
    report_path = Path(plots_dir) / "data_quality_report.txt"
    lines = ["Data Quality Report", "=" * 72]
    for title, values in sections:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        if not values:
            lines.append("None")
            continue
        lines.extend(values)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
