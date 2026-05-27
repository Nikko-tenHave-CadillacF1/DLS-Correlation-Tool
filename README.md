# DLS Correlation Tool

Generate engineering plots from multiple telemetry runs and optionally export a PowerPoint report.

## Setup

The tool is **plug-and-play**: every `Run_*.py` script auto-bootstraps a local
`.venv` and installs dependencies the first time you run it. There is no need
to open a terminal or call `pip` manually.

```powershell
python Run_Correlation.py        # creates .venv, installs deps, then runs
```

If you prefer to set things up explicitly:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Parquet input files require at least one of: `pyarrow`, `fastparquet` (both included in `requirements.txt`).

Set `DLS_SKIP_BOOTSTRAP=1` in your environment to disable the auto-installer
(useful in CI or shared/managed Python environments).

See also [tools/README.md](tools/README.md) for helper utilities (data
organisation, config validation, etc.).

---

## Quickstart

1. Drop input files into `Data/inputs/<workflow>/<event>/` (folders are auto-created)
2. Edit `WORKFLOW_NAME` and `EVENT` in the relevant `Run_*.py` file
3. Edit `RUNS` to point at your files
4. Run the script

```powershell
python Run_Correlation.py
python Run_BoxPlots.py
python Run_Dampers.py
```

Plots are saved to `Data/outputs/<workflow>/<event>/plots/`. The output folder opens automatically on completion.

To start a new workflow from scratch, copy `Run_Template.py` and change `WORKFLOW_NAME` and `EVENT` — all directories are created automatically.

### CLI Options

All three entry points support command-line arguments:

```powershell
python Run_Correlation.py --only "Front Heave" "Engine Efficiency"
python Run_Correlation.py --types scatter bar
python Run_Correlation.py --no-open
```

| Flag | Effect |
|------|--------|
| `--only NAME [NAME ...]` | Generate only plots whose name matches (case-insensitive) |
| `--types TYPE [TYPE ...]` | Generate only these plot types (`waveform`, `scatter`, `psd`, `histogram`, `bar`, `box`, `heatmap`) |
| `--no-open` | Do not auto-open the output folder after completion |
| `--dry-run` | Validate config and show what would be generated without creating plots |
| `--list-plots` | Print all configured plot names, grouped by type |
| `--check-only` | Run data-quality checks and produce the report without plotting |
| `--runs NAME [...]` | Restrict to a subset of configured runs by name |

Flags can be combined: `--types scatter --only "Front Heave" --no-open`.

---

## File Structure

### Files you edit

| File | Purpose |
|---|---|
| `Run_Template.py` | Reference template — copy this to start a new workflow |
| `Run_Correlation.py` | Runs, plot definitions, and PowerPoint export for correlation |
| `Run_BoxPlots.py` | Runs and box plot definitions |
| `Run_Dampers.py` | Runs, waveform, and scatter definitions for damper analysis |
| `channel_config.py` | Project-wide settings: folder paths, channel mappings, unit labels, transforms, calculated channels, filters, and render settings |

### Engine files (do not edit)

| File | Purpose |
|---|---|
| `dataplotter.py` | Data loading, preprocessing, data-quality checks, and plot dispatch |
| `plot_generators_waveform.py` | Waveform plot generation |
| `plot_generators_scatter.py` | Scatter plot generation |
| `plot_generators_misc.py` | PSD, histogram, and heatmap plot generation |
| `plot_generators_bar_box.py` | Bar and box plot generation |
| `plot_runtime.py` | `PlotJobConfig`, CLI parser, plot constructors, job runner, and PowerPoint export |
| `datafunctions.py` | Filtering, fitting, aggregation, and gating helpers |

---

## Data Layout

```
Data/
  inputs/
    correlation/          ← correlation input files, organised by event
      26R04MIA/
      26R03SUZ/
      misc/               ← files without a clear event
    boxplots/
      26T01BCN/
      26R01MEL/
    dampers/
      26R04MIA/
  templates/              ← PowerPoint template files (.pptx)
  outputs/
    correlation/
      26R04MIA/plots/     ← generated plots and reports
    boxplots/
      26T01BCN/plots/
    dampers/
      26R04MIA/plots/
```

All directories are created automatically on first run via `get_workflow_dirs(workflow, event)`.

### Cross-event comparisons

To compare runs from different events, set `EVENT = None` so the root folder
becomes `Data/inputs/<workflow>/`, then prefix filenames with the event subfolder:

```python
WORKFLOW_NAME = "correlation"
EVENT = None
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

RUNS = [
    {"name": "MIA LTS", "file": "26R04MIA/my_mia_file.parquet", "color": "#0083BF", "type": "DLS", "nlap": 1},
    {"name": "SUZ LTS", "file": "26R03SUZ/my_suz_file.parquet", "color": "#D70000", "type": "DLS", "nlap": 1},
]
```

Outputs go to `Data/outputs/correlation/plots/`. For a named output folder, override the
output directory:

```python
_, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, "MIA_vs_SUZ")
```

---

## Defining Runs

In any `Run_*.py` file, `RUNS` is a list of dicts:

```python
RUNS = [
    {
        "name":  "Baseline",              # display label used in all plots
        "file":  "my_run.parquet",        # path relative to Data/inputs/<workflow>/<event>/
        "color": "#D70000",               # hex colour for this run's traces
        "type":  "OC",                    # OC | CAR | DLS | DIL
        "nrun":  1,                       # (parquet only) rank-based run selection
        "nlap":  1,                       # exact lap filter; ignored when nrun is set
    },
]
```

`nrun=1` selects the lap with the lowest `nRun` value in the file; `nrun=2` selects the next lowest, etc.

---

## Job Configuration

Each `Run_*.py` file uses a `PlotJobConfig` dataclass to bundle all parameters:

```python
from plot_runtime import PlotJobConfig, run_from_config, parse_plot_cli, build_plot_groups

config = PlotJobConfig(
    title="CORRELATION PLOT GENERATION",
    root_folder=ROOT_FOLDER,
    output_dir=CORRELATION_OUTPUT_DIR,
    runs=RUNS,
    plot_definitions=build_plot_groups(
        waveforms=WAVEFORMS,
        scatters=SCATTERS,
        psds=PSDS,
        histograms=HISTOGRAMS,
        bars=BARS,
        boxes=BOXES,
        heatmaps=HEATMAPS,
    ),
    channel_mappings=CHANNEL_MAPPINGS,
    channel_transforms=CHANNEL_TRANSFORMS,
    calculated_channels=CORRELATION_CALCULATED,
    filters=CORRELATION_FILTERS,
    units_map=UNITS_MAP,
    powerpoint_template=POWERPOINT_TEMPLATE,
    powerpoint_output=POWERPOINT_OUTPUT,
    export_map=POWERPOINT_EXPORT_MAP,
)

if __name__ == "__main__":
    run_from_config(config, parse_plot_cli("Correlation plots"))
```

`build_plot_groups()` accepts keyword-only arguments: `waveforms`, `scatters`, `psds`, `histograms`, `bars`, `boxes`, `heatmaps`. Omitted categories default to empty.

---

## Defining Plots

All plot definitions use named-argument constructors imported from `plot_runtime`. These provide IDE autocomplete and validate parameters at import time.

```python
from plot_runtime import (
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot,
    BarPlot, BoxPlot, HeatmapPlot, Marker,
)
```

### Waveform

```python
WaveformPlot(
    name="Driver Input",
    channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel'),
    # One entry per row. Use ('left_ch', 'right_ch') for dual-axis rows.
    axis_limits=(None, ((60, 400), (-1, 9)), (-160, 160)),
    # Per-row y-limits. Dual-axis: ((y1_min, y1_max), (y2_min, y2_max)).
    reference_lines=((-350, 0, 350), None, (0,)),
    # Per-row horizontal reference lines.
    subplot_heights=(0.4, 0.8, 0.4),
    # Relative row heights; 0.8 is twice as tall as 0.4.
    x_channel="sLap",
    # X-axis channel. Default "sLap" (distance). Use "tLap" for elapsed time.
    x_limits=(0, 4000),
    # Optional (x_min, x_max) to zoom to a section.
    highlight_zones=('SM', '<', 0.3),
    # Shade x-regions where the gate condition is true.
    # Each run is evaluated against its own data and shaded in a highly
    # transparent version of that run's own colour.
    # Single condition: ('channel', 'operator', value)
    # Multiple (all must match): [('ch1', '>', v1), ('ch2', '<', v2)]
    # Operators: '>' '<' '>=' '<=' '==' 'between'
    normalise=False,
    # If True, all channels on each subplot are normalised to [0, 1] using
    # the global min/max across all runs. Dual-axis becomes single-axis.
    legend_position="top",
    # "top" (default): run legend above the subplots.
    # "right": vertical legend to the right of the plot area.
    show_delta=False,
    # If True and exactly 2 runs are loaded, a thin difference row
    # (run_B − run_A) is appended below each primary row.
    markers=[
        # Vertical reference lines. Two flavours:
        #   • Static  — fixed x position, drawn once on every run.
        Marker(x=1500, label="T1 entry", color="#FF6600", linestyle="--"),
        Marker(x=2800, label="T-final", row=0),  # row=0 → only on the top subplot
        #   • Condition — per-run; one marker is emitted at the x_channel
        #     value of each rising / falling / either transition of a gate
        #     condition. The line is drawn in the run's colour so DRY and WET
        #     are visually distinguishable without a run-name in the label.
        Marker(condition=('SM', '>', 0.5), edge="rising", label="SM>0.5"),
        Marker(
            condition=[('pBrakeF', '>', 50), ('vCar', '>', 100)],
            edge="rising",          # 'rising' | 'falling' | 'both'
            label="hard brake",
            max_count=3,             # cap markers per run (first N kept)
            linestyle="-.",
        ),
    ],
)
```

Marker rules:
- Each `Marker` requires **exactly one** of `x=...` (static) or `condition=...` (per-run).
- `edge` only applies to condition markers: `"rising"` (default), `"falling"`, or `"both"`.
- A condition that is already true at the first sample is **not** counted as a rising edge (true transition required).
- `row=N` restricts the marker to the N-th subplot row; default draws on every row.
- `color=None` (default): static markers fall back to grey, condition lines use the run colour.
- Condition markers are silently skipped on non-waveform plot types (scatter/PSD/etc. accept only static `x=...` markers).

### Scatter

```python
ScatterPlot(
    name="Front Ride vCar",
    x_channel="vCar",
    y_channel="hRideF",
    axis_limits=[(0, 350), (None, None)],   # [(x_min, x_max), (y_min, y_max)]
    best_fit=[('SM', 0, 0.5)],
    # Trend line options:
    #   None / 0              → no fit
    #   1                     → single fit across all data
    #   [('ch', low, high)]   → segmented fits by channel value range
    #   [('x'/'y', low, high)]→ segmented fits by axis value range
    gate=('SM', '<', 1),
    # Data filter applied before plotting.
    # Single: ('channel', 'operator', value)
    # Multi (all must match): [('ch1', '>', v1), ('ch2', '<', v2)]
    # Operators: '>' '<' '>=' '<=' '==' 'between'
    show_equations=True,
    show_error=True,
    color_gate=('SM', '<', 0.5, '#00AAFF'),
    # Highlight a subset of points with a second color.
    # Format: ('channel', 'operator', value, '#hexcolor')
    # Only points satisfying the condition are drawn in the gate color.
    # Fit line still uses all (gated) data.
    annotate_fit_at=250.0,
    # Draw a vertical dashed line at this x-value and annotate each run's
    # fit-line y-value at that x. Only works with single fits (best_fit=1).
    robust=False,
    # If True with best_fit=1, uses Theil-Sen + MAD outlier rejection.
    # Outliers are drawn as faint grey 'x' markers and logged in the
    # data-quality report. ``robust_threshold`` (default 3.0) sets the
    # MAD multiplier for outlier rejection.
    markers=[Marker(x=250, label="250 km/h")],
    # Static vertical reference markers (condition= markers are ignored
    # on scatter plots — they only make sense on time/distance axes).
)
```

### PSD

```python
PsdPlot(
    name="Front Ride PSD",
    channel="hRideF (raw)",
    # Also accepts a list to overlay multiple channels on the same plot:
    # channel=["hRideF (raw)", "hRideR (raw)"]
    # Legend entries become "RUNNAME — channel".
    axis_limits=[(0, 50), (1e-4, None)],    # [(f_min, f_max), (power_min, power_max)]
    log_scale=True,
)
```

### Histogram

```python
HistogramPlot(
    name="Plank Power Distribution",
    channel="PPlank_F",
    axis_limits=[(1, 51), (None, None)],
)
```

### Bar

```python
BarPlot(
    name="Cumulative Fuel",
    metrics=(("dmInjector (kg/s)", "integral"),),
    # Each entry: "channel" (uses default_aggregation) or ("channel", "aggregation")
    # Aggregations: "integral" "sum" "last" "mean" "max" "min"
    target_line=12.5,
    # Draw a dashed horizontal reference line at this value (in the same units
    # as the bar metric). Annotated with its value.
)
```

### Box

```python
BoxPlot(
    name="Low Speed Corner Distribution",
    channels=["xDamperFL", "xDamperFR"],
    aggregation_mode="per_run",   # "per_run" | "aggregated" | "per_run_aggregated"
    gate=('vCar', '<', 120),
)
```

| Mode | Behaviour |
|------|-----------|
| `per_run` | One box per run, coloured by run colour |
| `aggregated` | All runs merged into a single box per channel |
| `per_run_aggregated` | Per-run boxes followed by a combined "ALL" box (separated by a dashed line) |

### Heatmap

Two-dimensional density or aggregation grids. One panel per run, with a shared
colour scale so panels are directly comparable.

```python
HeatmapPlot(
    name="gLat vs gLong density",
    x_channel="gLat",
    y_channel="gLong",
    z_channel=None,             # None → 2D-histogram (counts per bin)
    aggregation="mean",         # used only when z_channel is set:
                                # "mean" | "median" | "std" | "sum" | "max" | "min"
    bins=100,                   # int, or (nx, ny) for non-square grids
    axis_limits=[(None, None), (None, None)],
    gate=('SM', '<', 1),        # optional pre-filter
)
```

---

## channel_config.py

This file contains all project-wide settings. Edit it to:

- **`CHANNEL_MAPPINGS`** — map raw source column names to the canonical names used in plot definitions. One entry per source type (`OC`, `CAR`, `DLS`, `DIL`).
- **`UNITS_MAP`** — channel name → unit label shown on axes (case-insensitive).
- **`CHANNEL_TRANSFORMS`** — sign corrections and unit conversions applied per source type (e.g. W → kW).
- **`CORRELATION_CALCULATED` / `BOXPLOT_CALCULATED` / `DAMPER_CALCULATED`** — derived channels computed from existing ones.
- **`CORRELATION_FILTERS` / `BOXPLOT_FILTERS` / `DAMPER_FILTERS`** — per-channel low-pass filter settings. Use `cutoff=0` to disable. The `"all"` key sets a fallback for any unlisted channel.
- **`SCATTER_MAX_POINTS`** — maximum points drawn per run; data is randomly downsampled above this.
- **`BAR_SECONDARY_AXIS_RATIO`** — y-axis scale factor for the right axis in dual-axis bar charts.

---

## Filtering Plots at Runtime

The CLI flags (`--only`, `--types`) are the primary interface. For programmatic use, `plot_data` accepts two optional filters:

```python
plotter.plot_data(
    plot_types=["scatter", "psd"],   # restrict to these plot categories
    plot_names=["Front Ride vCar"],  # restrict to plots with these names (case-insensitive)
)
```

---

## Output

- All output filenames are normalised to **lowercase** (e.g. `scatter_front_heave.png`).
- A data-quality report is written to `data_quality_report.txt` in the plots folder.
- A summary line is printed on completion: `Generated 47 plot(s) in 16.9s → <path>`.
- The output folder opens automatically unless `--no-open` is passed.
- Warnings (missing channels, empty gate results, etc.) always print regardless of verbose mode.

---

## PowerPoint Export

Set `EXPORT_TO_POWERPOINT = True` in `Run_Correlation.py` and place a `.pptx` template in `Data/templates/`.

Use the `Slide()` helper to build the export map declaratively:

```python
from plot_runtime import Slide

POWERPOINT_EXPORT_MAP = [
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("double_plot", "scatter/Gear Ratios", "scatter/Engine Power"),
]
```

Set `POWERPOINT_START_SLIDE` to the 1-based slide number where the first entry should be placed (e.g. `4` to leave cover/intro slides untouched):

```python
POWERPOINT_START_SLIDE = 4
```

Pass it through in the `run_workflow()` call:

```python
run_workflow(
    ...,
    powerpoint_start_slide=POWERPOINT_START_SLIDE,
)
```

Layouts:
- `"main_plot"` — single image, full-width on the slide
- `"double_plot"` — two images side-by-side

Plot references use `"type/Plot Name"` notation and are automatically converted to filenames (`type_plot_name.png`, all lowercase). The plot figure size is automatically matched to the template slide aspect ratio.
