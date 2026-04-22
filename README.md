# Correlation Reporting Tool

Generate engineering plots from multiple telemetry runs and optionally export a PowerPoint report.

## Entry Points

- `Run_Correlation.py`: correlation plots and PowerPoint export
- `Run_BoxPlots.py`: box-plot analysis

## Setup

```powershell
pip install -r requirements.txt
```

If you read `.parquet` files, install at least one parquet backend:

- `pyarrow`
- `fastparquet`

## Run

```powershell
python Run_Correlation.py
python Run_BoxPlots.py
```

Generated output is written under `Data/outputs/`.

## Data Layout

The project now uses a fixed, user-friendly data structure:

```text
Data/
  inputs/
    correlation/
    boxplots/
      fuel_investigation/
  templates/
  outputs/
    correlation/
      plots/
    boxplots/
      plots/
  archive/
```

See [Data/README.md](./Data/README.md) for the purpose of each folder.

## What To Edit

- Put correlation input files in `Data/inputs/correlation/`
- Put box-plot input files in `Data/inputs/boxplots/fuel_investigation/`
- Put PowerPoint templates in `Data/templates/`
- Review plots and reports in `Data/outputs/correlation/plots/` and `Data/outputs/boxplots/plots/`

## Key Files

- `Run_Correlation.py`: plot definitions, run definitions, export settings
- `Run_BoxPlots.py`: box-plot definitions and settings
- `plot_runtime.py`: shared runner helpers
- `plot_shared_config.py`: shared channel mappings and units
- `dataplotter.py`: loading, preprocessing, and plot generation
- `datafunctions.py`: filtering, fitting, aggregation, and plotting helpers
- `data_layout.py`: canonical project paths
- `powerpointexporter.py`: PowerPoint export helpers

## Plot Types

- Waveform: time-series plots with optional dual-axis rows
- Scatter: correlation plots with optional gates and trend lines
- PSD: frequency-domain plots
- Histogram: distribution plots
- Bar: aggregate metrics per run
- Box: per-run or combined distributions

## Performance Notes

The plotting path now avoids a few unnecessary repeats:

- PowerPoint template aspect ratios are cached
- Calculated-channel dependency discovery is cached per function
- Box-plot gates are applied once per run and reused across channels
- Scatter plots automatically switch to density rendering for very large point clouds

## Configuration Notes

`Run_Correlation.py` is the main place to update:

- `RUNS`: input files and colors
- `CHANNEL_MAPPINGS`: source-to-standard channel names
- `CHANNEL_TRANSFORMS`: unit/sign corrections
- `CALCULATED_CHANNELS`: derived channels
- `LOW_PASS_FILTERS`: per-channel filtering
- `WAVEFORM_PLOT_DEFINITIONS`, `SCATTER_PLOT_DEFINITIONS`, `PSD_PLOT_DEFINITIONS`, `HISTOGRAM_PLOT_DEFINITIONS`, `BAR_PLOT_DEFINITIONS`, `BOX_PLOT_DEFINITIONS`: plot requests
- `POWERPOINT_EXPORT_MAP`: slide-to-image mapping

`Run_BoxPlots.py` contains the same kind of focused settings for the box-plot workflow.

## Notes

- PowerPoint export is optional.
- Correlation plots are saved under `Data/outputs/correlation/plots/`.
- Box plots are saved under `Data/outputs/boxplots/plots/`.
