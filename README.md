# Correlation Reporting Tool

Generate engineering plots from multiple telemetry runs and optionally export a PowerPoint report.

## Entry Points

- `Run_Correlation.py`: main correlation workflow
- `Run_BoxPlots.py`: box-plot workflow

Each entry point is intentionally small and mostly configuration-driven:

- `Run_Correlation.py` holds the correlation run list, plot definitions, and export settings
- `Run_BoxPlots.py` holds the box-plot run list and box-plot settings
- Shared mapping/unit definitions live in `plot_shared_config.py`
- Canonical project paths live in `data_layout.py`

For new users, a minimal `Run_Example.py` would usually look like this:

```python
"""Minimal plotting entry point."""

from data_layout import resolve_correlation_input_dir
from plot_runtime import build_plot_groups, build_plotter, run_plot_job
from plot_shared_config import CHANNEL_MAPPINGS, UNITS_MAP

ROOT_FOLDER = resolve_correlation_input_dir()

RUNS = [
    {
        "name": "Example Run",
        "file": "example_run.parquet",
        "color": "#FF5500",
        "type": "CAR",
    }
]

PLOT_DEFINITIONS = build_plot_groups(
    [],
    [],
    [],
    [],
    [],
    [],
)

def build_example_plotter():
    return build_plotter(
        root_folder=ROOT_FOLDER,
        output_dir=ROOT_FOLDER,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        units_map=UNITS_MAP,
    )

if __name__ == "__main__":
    run_plot_job(
        title="EXAMPLE PLOT RUN",
        plotter=build_example_plotter(),
        plot_method="plot_data",
        generate_message="Generating example plots...",
    )
```

The important idea is:

- choose your input folder
- define the runs you want to compare
- define the plot groups you want to generate
- hand everything to the shared runner

## Setup

```powershell
pip install -r requirements.txt
```

If you use `.parquet` input files, install at least one parquet backend:

- `pyarrow`
- `fastparquet`

## Run

```powershell
python Run_Correlation.py
python Run_BoxPlots.py
```

## Data Layout

The project uses a fixed layout under `Data/`:

```text
Data/
  inputs/
    correlation/
    boxplots/
  templates/
  outputs/
    correlation/
      plots/
    boxplots/
      plots/
  archive/
```

See [Data/README.md](./Data/README.md) for a short explanation of each folder.

## What To Edit

- Put correlation input files in `Data/inputs/correlation/`
- Put box-plot input files in `Data/inputs/boxplots/`
- Put PowerPoint templates in `Data/templates/`
- Review generated correlation outputs in `Data/outputs/correlation/plots/`
- Review generated box-plot outputs in `Data/outputs/boxplots/plots/`

## Core Modules

- `dataplotter.py`: loads data, preprocesses it, and generates plots
- `datafunctions.py`: filtering, fitting, aggregation, and plotting helpers
- `plot_runtime.py`: shared runner helpers for the entry points
- `powerpointexporter.py`: PowerPoint export helpers
- `data_quality_report.py`: preflight validation and reporting

## Supported Plot Types

- Waveform: time-series traces with optional dual-axis rows
- Scatter: correlation plots with optional gates and trend lines
- PSD: frequency-domain plots
- Histogram: distribution plots
- Bar: aggregate metrics per run
- Box: per-run or combined distributions

## Performance Notes

The plotting path now avoids a few unnecessary repeats:

- PowerPoint template aspect ratios are cached
- Calculated-channel dependency discovery is cached
- Scatter styling adapts slightly for dense point clouds
- Scatter and box-plot gates are reused where possible
- Box-plot overlays reuse prefiltered data instead of recomputing it

## Configuration Guide

`Run_Correlation.py` is the main place to edit when working on the correlation workflow:

- `RUNS`: input files and colors
- `CHANNEL_TRANSFORMS`: source-specific sign/unit adjustments
- `CALCULATED_CHANNELS`: derived channels
- `LOW_PASS_FILTERS`: per-channel filtering
- `WAVEFORM_PLOT_DEFINITIONS`
- `SCATTER_PLOT_DEFINITIONS`
- `PSD_PLOT_DEFINITIONS`
- `HISTOGRAM_PLOT_DEFINITIONS`
- `BAR_PLOT_DEFINITIONS`
- `BOX_PLOT_DEFINITIONS`
- `POWERPOINT_EXPORT_MAP`

`Run_BoxPlots.py` contains the equivalent configuration for the box-plot workflow.

## Notes

- PowerPoint export is optional.
- Correlation plots are saved under `Data/outputs/correlation/plots/`.
- Box plots are saved under `Data/outputs/boxplots/plots/`.

## Future Improvements

If we revisit the tool later, the most useful upgrades would likely be:

1. Add a cached run-artifact layer so repeated runs can skip unchanged preprocessing.
2. Move the large plot-definition blocks into dedicated config files for easier editing.
3. Add a simple CLI wrapper for switching between common run presets without editing Python files.
