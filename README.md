# DLS Correlation Tool

Lightweight workflow for comparing multiple telemetry runs, generating engineering plots, and optionally exporting a PowerPoint report.

## Entry Point

Use [Run_Correlation.py](/c:/GitHub_Local/DLS-Correlation-Tool/Run_Correlation.py) as the main configuration and execution script.

## Project Files

- [Run_Correlation.py](/c:/GitHub_Local/DLS-Correlation-Tool/Run_Correlation.py): run definitions, channel config, plot definitions, and execution.
- [dataplotter.py](/c:/GitHub_Local/DLS-Correlation-Tool/dataplotter.py): orchestration for loading, preprocessing, and generating all plots.
- [datafunctions.py](/c:/GitHub_Local/DLS-Correlation-Tool/datafunctions.py): reusable numeric/plot helpers (mapping, filtering, PSD, scatter rendering/fits, histogram binning).
- [data_quality_report.py](/c:/GitHub_Local/DLS-Correlation-Tool/data_quality_report.py): preflight data-quality checks and report writing.
- [powerpointexporter.py](/c:/GitHub_Local/DLS-Correlation-Tool/powerpointexporter.py): PowerPoint template parsing and image insertion.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python Run_Correlation.py
```

Generated plots are saved to `Data/plots`.

## Quick Start

1. Open [Run_Correlation.py](/c:/GitHub_Local/DLS-Correlation-Tool/Run_Correlation.py) and update `RUNS` with your input files.
2. (Optional) Disable PowerPoint export for faster plot-only runs:
   set `EXPORT_TO_POWERPOINT = False`.
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run:

```powershell
python Run_Correlation.py
```

5. Review outputs in `Data/plots` (including `data_quality_report.txt`).

## Supported Plot Types

- Waveform (multi-panel time/distance traces)
- Scatter (with optional single, double, or segmented linear fits)
- PSD (Welch method; default `nperseg=512` unless overridden per plot)
- Histogram (shared run overlays; equal-width bins when `xmin/xmax` limits are set)

## Core Configuration (Run_Correlation.py)

- `RUNS`: input datasets and display colors.
- `CHANNEL_MAPPINGS`: source-to-common channel naming.
- `CHANNEL_TRANSFORMS`: per-source value corrections (sign, scale, offsets).
- `CALCULATED_CHANNELS`: derived channels built after mapping and cleaning.
- `LOW_PASS_FILTERS`: per-channel and global low-pass settings.
- `WAVEFORM_PLOT_DEFINITIONS`, `SCATTER_PLOT_DEFINITIONS`, `PSD_PLOT_DEFINITIONS`, `HISTOGRAM_PLOT_DEFINITIONS`: plot requests.
- `POWERPOINT_EXPORT_MAP`: slide/image mapping for report output.
- `SCATTER_RENDER_MODE`, `SCATTER_DENSITY_THRESHOLD`, `SCATTER_MAX_POINTS`, `SCATTER_HEXBIN_GRIDSIZE`: readability controls for dense scatter clouds.

## PowerPoint Export

If `EXPORT_TO_POWERPOINT = True`, the script fills `POWERPOINT_TEMPLATE` and writes `POWERPOINT_OUTPUT`.

## Plot Behavior Notes

- Waveform legends are placed at figure level (above subplots) to avoid covering trace data.
- Scatter gradient error callouts are reported relative to the first run in `RUNS` (baseline), and baseline rows are omitted from the per-segment listing.
- PSD plots now render when at least one run has valid channel data; runs missing data are skipped with warnings.
- Scatter plots support automatic density rendering for large point clouds (`SCATTER_RENDER_MODE = "auto"`).

## Data Quality Report

Each run writes a preflight report to `Data/plots/data_quality_report.txt` with:

- Missing referenced channels by run
- High NaN ratios (>20%)
- Flatlined channels
- `sLap` reset counts
- `sLap` alignment estimate (vCar-based scale/offset/end-drift) relative to the baseline run

## Typical Workflow

1. Add input files under `Data` and update `RUNS`.
2. Align channels via `CHANNEL_MAPPINGS` and `CHANNEL_TRANSFORMS`.
3. Define derived channels and filtering settings.
4. Configure required plots.
5. Run the script and review plots in `Data/plots`.
6. If enabled, review the generated PowerPoint report.
