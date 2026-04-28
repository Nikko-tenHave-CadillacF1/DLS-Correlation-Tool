# DLS Correlation Tool

Generate engineering plots from multiple telemetry runs and optionally export a PowerPoint report.

## Setup

```powershell
pip install -r requirements.txt
```

Parquet input files require at least one of: `pyarrow`, `fastparquet` (both included in `requirements.txt`).

---

## Quickstart

1. Drop input files into the appropriate folder under `Data/inputs/`
2. Edit `RUNS` in the relevant `Run_*.py` file to point at your files
3. Run the script

```powershell
python Run_Correlation.py
python Run_BoxPlots.py
python Run_Dampers.py
```

Plots are saved to `Data/outputs/<workflow>/plots/`.

---

## File Structure

### Files you edit

| File | Purpose |
|---|---|
| `Run_Correlation.py` | Runs, plot definitions, and PowerPoint export settings for the correlation workflow |
| `Run_BoxPlots.py` | Runs and box plot definitions for the box-plot workflow |
| `Run_Dampers.py` | Runs, waveform, and scatter definitions for the damper workflow |
| `channel_config.py` | Project-wide settings: folder paths, channel mappings, unit labels, transforms, calculated channels, filters, and render settings |

### Engine files (do not edit)

| File | Purpose |
|---|---|
| `dataplotter.py` | Data loading, preprocessing, data-quality checks, and plot dispatch |
| `plot_generators_waveform.py` | Waveform plot generation |
| `plot_generators_scatter.py` | Scatter plot generation |
| `plot_generators_misc.py` | PSD and histogram plot generation |
| `plot_generators_bar_box.py` | Bar and box plot generation |
| `plot_runtime.py` | Plot constructors, runner helpers, and PowerPoint export |
| `datafunctions.py` | Filtering, fitting, aggregation, and gating helpers |

---

## Data Layout

```
Data/
  inputs/
    correlation/    ← correlation input files
    boxplots/       ← box-plot input files
    dampers/        ← damper input files
  templates/        ← PowerPoint template files (.pptx)
  outputs/
    correlation/plots/
    boxplots/plots/
    dampers/plots/
```

All directories are created automatically on first run.

---

## Defining Runs

In any `Run_*.py` file, `RUNS` is a list of dicts:

```python
RUNS = [
    {
        "name":  "Baseline",              # display label used in all plots
        "file":  "my_run.parquet",        # path relative to the workflow's input folder
        "color": "#D70000",               # hex colour for this run's traces
        "type":  "OC",                    # OC | CAR | DLS | DIL
        "nrun":  1,                       # (parquet only) rank-based run selection
        "nlap":  1,                       # exact lap filter; ignored when nrun is set
    },
]
```

`nrun=1` selects the lap with the lowest `nRun` value in the file; `nrun=2` selects the next lowest, etc.

---

## Defining Plots

All plot definitions use named-argument constructors imported from `plot_runtime`. These provide IDE autocomplete and validate parameters at import time.

```python
from plot_runtime import WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot
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
)
```

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
    aggregation_mode="per_run",   # "per_run" | "aggregated"
    gate=('vCar', '<', 120),
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

`plot_data` accepts two optional filters:

```python
plotter.plot_data(
    plot_types=["scatter", "psd"],   # restrict to these plot categories
    plot_names=["Front Ride vCar"],  # restrict to plots with these names (case-insensitive)
)
```

Both can be combined. `run_plot_job` exposes the same options via `plot_types` and `plot_names`.

---

## PowerPoint Export

Set `EXPORT_TO_POWERPOINT = True` in `Run_Correlation.py` and place a `.pptx` template in `Data/templates/`.

`POWERPOINT_EXPORT_MAP` maps slide numbers to plot image files:

```python
POWERPOINT_EXPORT_MAP = {
    4:  {"layout": "main_plot",   "images": ["waveform_Driver_Input.png"]},
    6:  {"layout": "double_plot", "images": ["scatter_Gear_Ratios.png", "scatter_Engine_Power.png"]},
}
```

Layouts:
- `"main_plot"` — single image, full-width on the slide
- `"double_plot"` — two images side-by-side

Image filenames follow the pattern `{plot_type}_{Plot_Name_with_spaces_replaced_by_underscores}.png`.
