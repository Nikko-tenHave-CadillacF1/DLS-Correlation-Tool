# Correlation Reporting Tool

Lightweight workflow for comparing multiple telemetry runs, generating engineering plots, and optionally exporting a PowerPoint report.

## Entry Point

Use `Run_Correlation.py` as the main configuration and execution script.

## Project Files

- **Run_Correlation.py**: run definitions, channel config, plot definitions, and execution.
- **dataplotter.py**: orchestration for loading, preprocessing, and generating all plots.
- **datafunctions.py**: reusable numeric/plot helpers (mapping, filtering, PSD, scatter rendering/fits, histogram binning).
- **data_quality_report.py**: preflight data-quality checks and report writing.
- **powerpointexporter.py**: PowerPoint template parsing and image insertion.
- **main_correlation_points.py**: optional summary report of main correlation differences between runs.

## Install

```powershell
pip install -r requirements.txt
```

Parquet inputs require a parquet backend (`pyarrow` or `fastparquet`) in your active Python environment.

## Run

```powershell
python Run_Correlation.py
```

Generated plots are saved to `Data/plots`.

## Quick Start

1. Open `Run_Correlation.py` and update `RUNS` with your input files.
2. (Optional) Disable PowerPoint export for faster plot-only runs: set `EXPORT_TO_POWERPOINT = False`.
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you use `.parquet` run files, make sure at least one parquet backend is installed (`pyarrow` or `fastparquet`).

4. Run:

```powershell
python Run_Correlation.py
```

5. Review outputs in `Data/plots` (including `data_quality_report.txt`).

## Supported Plot Types

- Waveform (multi-panel traces, with optional dual-channel overlay per row)
- Scatter (plain, single-fit, or segmented-fit linear trend overlays, with optional gating)
- PSD (Welch method; default `nperseg=512` unless overridden per plot)
- Histogram (shared run overlays; equal-width bins when `xmin/xmax` limits are set)

## Core Configuration (Run_Correlation.py)

- `RUNS`: input datasets and display colors.
- `CHANNEL_MAPPINGS`: source-to-common channel naming.
- `CHANNEL_TRANSFORMS`: per-source value corrections (sign, scale, offsets).
- `CALCULATED_CHANNELS`: derived channels built after mapping and cleaning.
- `LOW_PASS_FILTERS`: per-channel and global low-pass settings.
- `WAVEFORM_PLOT_DEFINITIONS`, `SCATTER_PLOT_DEFINITIONS`, `PSD_PLOT_DEFINITIONS`, `HISTOGRAM_PLOT_DEFINITIONS`, `BAR_PLOT_DEFINITIONS`: plot requests.
- `POWERPOINT_EXPORT_MAP`: slide/image mapping for report output.
- `SCATTER_RENDER_MODE`, `SCATTER_DENSITY_THRESHOLD`, `SCATTER_MAX_POINTS`, `SCATTER_HEXBIN_GRIDSIZE`: readability controls for dense scatter clouds.
- `ENABLE_MAIN_CORRELATION_SUMMARY` and related `MAIN_CORRELATION_*` settings: optional post-run summary report for main correlation differences.

## PowerPoint Export

If `EXPORT_TO_POWERPOINT = True`, the script fills `POWERPOINT_TEMPLATE` and writes `POWERPOINT_OUTPUT`.

## Plot Behavior Notes

- Waveform legends are placed at figure level (above subplots) to avoid covering trace data.
- Waveform rows support either:
  - single channel: `"vCar"`
  - dual overlay: `("vCar", "nEngine")` (left/right y-axes, max 2 channels per row)
- Scatter gradient error callouts are reported relative to the first run in `RUNS` (baseline), and baseline rows are omitted from the per-segment listing.
- PSD plots now render when at least one run has valid channel data; runs missing data are skipped with warnings.
- Scatter plots support automatic density rendering for large point clouds (`SCATTER_RENDER_MODE = "auto"`).
- Scatter `best_fit=None` is treated as `0` (plain scatter) with a warning, to prevent empty plots from misconfiguration.
- Scatter gate info boxes are auto-positioned to avoid overlap with legend/trendline callouts.

## Waveform Definition Format

Each waveform definition is:

```python
["Name", (row_specs...), (axis_limits...), (reference_lines...), (subplot_heights...)]
```

Row spec options:

- Single row: `"channel"`
- Dual row: `("left_channel", "right_channel")`

Axis limits per row:

- Single row: `(ymin, ymax)`
- Dual row: `((left_ymin, left_ymax), (right_ymin, right_ymax))`

Reference lines per row:

- Single row: `None`, scalar, or tuple/list
- Dual row: `(left_refs, right_refs)`

Example:

```python
[
    "Waveform Example",
    ("SM", ("vCar", "nEngine"), ("pBrakeF", "rThrottle")),
    ((-0.2, 1.2), ((60, 360), (7000, 13000)), ((-10, 80), (-5, 105))),
    (None, (None, (10000,)), (None, None)),
    (0.2, 0.8, 0.5),
]
```

## Scatter Definition Format

Supported formats:

```python
["Name", (x, y), [(xmin, xmax), (ymin, ymax)], best_fit]
["Name", (x, y), [(xmin, xmax), (ymin, ymax)], best_fit, gate_spec]
["Name", (x, y), [(xmin, xmax), (ymin, ymax)], best_fit, legacy_item, gate_spec]
```

`best_fit` values:

- `0`: no fit
- `1`: single linear fit
- `2`: legacy-compatible single-fit fallback
- `[("x" or "y" or "<channel>", min, max), ...]`: segmented fits

Multi-fit condition notes:

- `("x", min, max)` or `("y", min, max)` segments by scatter axes.
- `("<channel>", min, max)` segments by another channel (for example `("SM", 0, 0.5)`).
- Scatter points are still plotted in full; only trendline fitting uses the condition subset.
- Channel-conditioned segments use data aligned to plotted samples (after any scatter gate).

Gate spec:

- Single condition: `("channel", ">", value)`
- Range: `("channel", "between", (low, high))`
- Multi-condition AND: `[("ch_a", ">", v1), ("ch_b", "<", v2)]`

Example:

```python
["Ride Height vs Speed (SM condition)", ("vCar", "hRideF"), None, [("SM", 0, 0.5), ("SM", 0.5, 1.0)]]
```

## PSD Plot Definition Format

Power Spectral Density plots use Welch's method for frequency domain analysis.

```python
["Name", "channel"]
["Name", "channel", nperseg]
["Name", "channel", nperseg, (freq_min, freq_max), (power_min, power_max)]
```

Parameters:

- `"channel"`: channel to analyze
- `nperseg`: number of samples per segment (default: 512)
- `freq_min, freq_max`: frequency axis limits (optional)
- `power_min, power_max`: power axis limits (optional)

Example:

```python
["Engine Speed PSD", "nEngine", 512, (0, 2000), (0, 100)]
```

## Histogram Plot Definition Format

Histograms overlay multiple runs with shared or equal-width bins.

```python
["Name", "channel"]
["Name", "channel", num_bins]
["Name", "channel", num_bins, (xmin, xmax)]
```

Parameters:

- `"channel"`: channel to histogram
- `num_bins`: target number of bins (default: 30)
- `xmin, xmax`: axis limits; enables equal-width bins in this range

Bin behavior:

- Without limits: auto-computed "nice" round-number bins
- With limits: equal-width bins spanning `[xmin, xmax]`

Example:

```python
["Throttle Distribution", "rThrottle", 40, (0, 100)]
```

## Bar Plot Definition Format

Bar plots aggregate channel metrics (e.g., mean, sum, integral) across runs.

```python
["Name", metric_specs]
["Name", metric_specs, group_label]
```

Metric spec formats:

- Single channel: `"channel"` (uses default aggregation: "last")
- Channel with aggregation: `["channel", "aggregation"]`
- Multiple metrics: `["channel1", ["channel2", "sum"], ["channel3", "integral"]]`

Supported aggregations:

- `"last"`: last value (default)
- `"first"`: first value
- `"mean"`: average
- `"min"`, `"max"`: minimum/maximum
- `"median"`: median value
- `"sum"`: total sum
- `"abs_sum"`: sum of absolute values
- `"integral"`: trapezoidal integration over time
- `"abs_integral"`: integration of absolute values

Example:

```python
["Engine Performance Metrics", [
    "nEngine",
    ["pBoost", "mean"],
    ["vCar", "max"],
    ["rThrottle", "integral"]
]]
```

## Data Quality Report

Each run writes a preflight report to `Data/plots/data_quality_report.txt` with:

- Missing referenced channels by run
- High NaN ratios (>20%)
- Flatlined channels
- `sLap` reset counts
- `sLap` alignment estimate (vCar-based scale/offset/end-drift) relative to the baseline run

## Optional Main Correlation Summary

When enabled in `Run_Correlation.py`:

- `ENABLE_MAIN_CORRELATION_SUMMARY = True`

the tool writes:

- `Data/plots/main_correlation_points.txt`
- `Data/plots/main_correlation_points.csv` (optional via `MAIN_CORRELATION_INCLUDE_CSV`)

This summary is designed to help report writing with:

- waveform difference metrics (mean abs, p95 abs, correlation)
- scatter trendline deltas vs baseline (slope/intercept/sample counts)
- simple coverage/confidence notes and copy-ready snippet suggestions

## Typical Workflow

1. Add input files under `Data` and update `RUNS`.
2. Align channels via `CHANNEL_MAPPINGS` and `CHANNEL_TRANSFORMS`.
3. Define derived channels and filtering settings.
4. Configure required plots.
5. Run the script and review plots in `Data/plots`.
6. If enabled, review the generated PowerPoint report.

## Code Quality & Maintainability

Recent improvements have enhanced code quality without changing functionality:

- **Helper Functions**: Reduced duplication through shared utilities (`_safe_get_config`, `_to_numeric_safe`, etc.)
- **Filter Consolidation**: Unified parquet filtering logic for consistent behavior across nRun/nLap selection
- **Filter Extraction**: Single source of truth for Butterworth filter application
- **Optimization**: Set-based column lookups for improved performance
- **Code Reduction**: ~5.3% reduction in total lines while maintaining 100% backward compatibility

These changes improve maintainability, reduce bugs, and make future enhancements easier to implement.
