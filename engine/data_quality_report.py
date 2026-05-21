"""Structured Markdown data-quality report generation.

Replaces the previous flat-text ``data_quality_report.txt`` writer with a
report that's easy to skim in any Markdown viewer or in plain-text.

Sections produced:
  1. Summary table (per-run: rows, channels, missing, NaN%, flatlines, sLap resets)
  2. Sample rate detection
  3. Missing referenced channels
  4. High NaN ratios (>20%)
  5. Flatlined channels
  6. sLap resets
  7. sLap alignment estimate (vCar-based)
  8. Outlier rejection (populated when robust scatter fits were used)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# Header used by main.run_quality_checks to wrap the final summary.
REPORT_FILENAME = "data_quality_report.md"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a simple GitHub-flavoured Markdown table."""
    if not rows:
        return "_None_\n"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def build_quality_sections(
    runs: List[Dict[str, Any]],
    run_data: Dict[str, pd.DataFrame],
    plot_definitions: Iterable[Iterable[Any]],
    run_sample_rates: Optional[Dict[str, Tuple[float, str]]] = None,
    outlier_log: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[str, List[Any]]]:
    """Build all data-quality sections as (title, rows) pairs.

    `rows` is a list of either:
      * plain strings (rendered as bullet list), or
      * dicts with a ``"_table"`` key (rendered as a Markdown table).
    """
    from dataplotter import collect_referenced_channels, estimate_slap_alignment

    referenced = collect_referenced_channels(plot_definitions)

    summary_rows: List[List[str]] = []
    missing_rows: List[List[str]] = []
    high_nan_rows: List[List[str]] = []
    flatlined_rows: List[List[str]] = []
    slap_reset_rows: List[List[str]] = []

    for run in runs:
        run_name = run["name"].lower()
        if run_name not in run_data:
            summary_rows.append([run["name"].upper(), "—", "—", "—", "—", "—", "_not loaded_"])
            continue

        df = run_data[run_name]
        n_rows = len(df)
        n_cols = len(df.columns)

        missing = [ch for ch in referenced if ch not in df.columns]
        if missing:
            missing_rows.append([run["name"].upper(), ", ".join(missing[:20]) + (" …" if len(missing) > 20 else "")])

        nan_count = 0
        flat_count = 0
        for ch in referenced:
            if ch not in df.columns:
                continue
            series = pd.to_numeric(df[ch], errors="coerce")
            nan_ratio = float(series.isna().mean()) if len(series) else 0.0
            if nan_ratio > 0.20:
                nan_count += 1
                high_nan_rows.append([run["name"].upper(), ch, f"{nan_ratio:.1%}"])

            valid = series.dropna()
            if len(valid) > 20 and float(valid.std()) < 1e-9:
                flat_count += 1
                flatlined_rows.append([run["name"].upper(), ch])

        resets = 0
        if "sLap" in df.columns:
            sl = pd.to_numeric(df["sLap"], errors="coerce")
            resets = int((sl.diff() < 0).sum())
            if resets > 0:
                slap_reset_rows.append([run["name"].upper(), str(resets)])

        summary_rows.append([
            run["name"].upper(),
            f"{n_rows:,}",
            str(n_cols),
            str(len(missing)),
            str(nan_count),
            str(flat_count),
            str(resets),
        ])

    # Sample-rate rows
    sr_rows: List[List[str]] = []
    if run_sample_rates:
        for run_name, (rate, source) in run_sample_rates.items():
            sr_rows.append([run_name.upper(), f"{rate:.1f} Hz", source])

    # sLap alignment from the existing helper — keep as bullet list.
    align_lines = estimate_slap_alignment(runs, run_data)

    # Outlier rejection from robust scatter fits.
    outlier_rows: List[List[str]] = []
    if outlier_log:
        for entry in outlier_log:
            outlier_rows.append([
                entry.get("plot", ""),
                entry.get("run", ""),
                f"{entry.get('n_outliers', 0)} / {entry.get('n_total', 0)}",
                f"{entry.get('pseudo_r2', 0.0):.3f}",
            ])

    sections: List[Tuple[str, List[Any]]] = [
        ("Summary", [{
            "_table": True,
            "headers": ["Run", "Rows", "Cols", "Missing", "High-NaN", "Flatlined", "sLap Resets"],
            "rows": summary_rows,
        }]),
        ("Sample Rate", [{
            "_table": True,
            "headers": ["Run", "Detected Rate", "Source"],
            "rows": sr_rows,
        }] if sr_rows else []),
        ("Missing Referenced Channels", [{
            "_table": True,
            "headers": ["Run", "Channels"],
            "rows": missing_rows,
        }] if missing_rows else []),
        ("High NaN Ratios (>20%)", [{
            "_table": True,
            "headers": ["Run", "Channel", "NaN Ratio"],
            "rows": high_nan_rows,
        }] if high_nan_rows else []),
        ("Flatlined Channels", [{
            "_table": True,
            "headers": ["Run", "Channel"],
            "rows": flatlined_rows,
        }] if flatlined_rows else []),
        ("sLap Resets", [{
            "_table": True,
            "headers": ["Run", "Reset Count"],
            "rows": slap_reset_rows,
        }] if slap_reset_rows else []),
        ("sLap Alignment Estimate (vCar-based)", list(align_lines)),
        ("Outlier Rejection (Robust Scatter Fits)", [{
            "_table": True,
            "headers": ["Plot", "Run", "Outliers / Total", "pseudo-R²"],
            "rows": outlier_rows,
        }] if outlier_rows else []),
    ]
    return sections


def write_data_quality_report(plots_dir, sections) -> Path:
    """Write the structured Markdown data-quality report."""
    plots_dir = Path(plots_dir)
    report_path = plots_dir / REPORT_FILENAME

    out = ["# Data Quality Report", ""]

    # Table of contents.
    out.append("## Contents")
    for title, _ in sections:
        anchor = title.lower().replace(" ", "-").replace(",", "").replace("(", "").replace(")", "")
        anchor = anchor.replace(">", "").replace("/", "").replace("%", "").replace("²", "2")
        anchor = anchor.replace("--", "-")
        out.append(f"- [{title}](#{anchor})")
    out.append("")

    for title, values in sections:
        out.append(f"## {title}")
        out.append("")
        if not values:
            out.append("_None_")
            out.append("")
            continue
        for item in values:
            if isinstance(item, dict) and item.get("_table"):
                out.append(_md_table(item["headers"], item["rows"]))
            else:
                out.append(f"- {item}")
        out.append("")

    report_path.write_text("\n".join(out), encoding="utf-8")
    return report_path


def print_quality_summary(sections) -> None:
    """Print a compact summary of the data-quality report to stdout."""
    print("\nData Quality Summary")
    print("=" * 72)
    for title, values in sections:
        # Count records: dict tables -> length of rows; plain list -> len.
        count = 0
        for item in values:
            if isinstance(item, dict) and item.get("_table"):
                count += len(item.get("rows", []))
            else:
                count += 1
        marker = "  " if count == 0 else "* "
        print(f"{marker}{title}: {count} record{'s' if count != 1 else ''}")
