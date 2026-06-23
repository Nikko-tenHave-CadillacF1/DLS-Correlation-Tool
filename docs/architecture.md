# Architecture

How the engine is wired together, where to change project-wide behaviour,
and what every advanced option does.

---

## Package layout

| Module | Purpose |
|---|---|
| `engine/plot_runtime.py` | `run_workflow()`, `PlotJobConfig`, CLI parser, PowerPoint export |
| `engine/plot_definitions.py` | All plot dataclasses (`WaveformPlot`, `ScatterPlot`, …) + `Marker` |
| `engine/dataplotter.py` | Data loading, preprocessing, dispatch to generators |
| `engine/plot_generators_waveform.py` | Waveform renderer |
| `engine/plot_generators_scatter.py` | Scatter renderer (incl. robust fits) |
| `engine/plot_generators_misc.py` | PSD / Histogram / Heatmap renderers |
| `engine/plot_generators_bar_box.py` | Bar / Box / BoxPlotGrid renderers |
| `engine/datafunctions.py` | Filtering, fitting, aggregation, gating helpers |
| `engine/data_quality_report.py` | Per-run summary + missing-channel report |
| `engine/logger.py` | Single logger configured via `configure(verbose=…)` |
| `channel_config.py` | Project-wide settings (mappings, units, calc channels, filters) |
| `bootstrap.py` | Auto-creates `.venv` and installs dependencies on first run |

---

## Job lifecycle

A call to `run_workflow(...)` performs the following steps:

1. **Build plot groups** — `build_plot_groups()` packs each plot-type list
   into a fixed-order 7-tuple in `PLOT_TYPE_ORDER`.
2. **Resolve workflow defaults** — `workflow_config()` looks up the input /
   output directories and calculated-channel / filter configs from
   `channel_config.py` based on the workflow name.
3. **Parse CLI** — `parse_plot_cli()` returns a `Namespace` of filter and
   diagnostic flags.
4. **Pre-flight validation** — `validate_config()` checks every run file
   exists and every run type is one of `OC | CAR | DLS | DIL`.
5. **Load data** — `DataPlotter` reads each run file, applies channel
   renames, transforms, calculated channels, resampling, and filters.
6. **Fail-fast typo check** — if any channel referenced by a plot
   definition is absent from **every** loaded run, exit with suggestions
   drawn from `difflib.get_close_matches`. Run with `--list-channels` to
   inspect what's available.
7. **Render plots** — each generator iterates its plot list and writes
   one PNG per plot to `<output_dir>/`.
8. **PowerPoint export** (optional) — `python-pptx` swaps placeholder
   images into a template by matching slot index.

---

## PlotJobConfig

`run_workflow()` is shorthand for building a `PlotJobConfig` and passing it
to `run_from_config()`. Use the dataclass directly when you need to share a
config between multiple call sites:

```python
from engine import PlotJobConfig, run_from_config, parse_plot_cli, build_plot_groups

config = PlotJobConfig(
    title="CORRELATION PLOT GENERATION",
    root_folder=ROOT_FOLDER,
    output_dir=CORRELATION_OUTPUT_DIR,
    runs=RUNS,
    plot_definitions=build_plot_groups(
        waveforms=WAVEFORMS, scatters=SCATTERS, psds=PSDS,
        histograms=HISTOGRAMS, bars=BARS, boxes=BOXES, heatmaps=HEATMAPS,
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

`build_plot_groups()` takes keyword-only arguments — any omitted category
defaults to `[]`.

---

## channel_config.py

Project-wide settings live here. Edit it to change defaults across every
workflow:

- **`CHANNEL_MAPPINGS`** — raw-source-column → canonical-name maps,
  one per source type (`OC`, `CAR`, `DLS`, `DIL`).
- **`UNITS_MAP`** — channel name → unit label shown on axes (case-insensitive).
- **`CHANNEL_TRANSFORMS`** — sign corrections and unit conversions
  applied per source type (e.g. W → kW).
- **`CALCULATED_CHANNELS`** — derived channels computed after loading.
  Per-workflow copies (`CORRELATION_CALCULATED`, `BOXPLOT_CALCULATED`,
  `DAMPER_CALCULATED`, `RIDE_DIL_CALCULATED`) default to this dict;
  override individually by modifying the copy in `channel_config.py` or by
  passing `calculated_channels=` to `run_workflow()`.
- **`CORRELATION_FILTERS` / `BOXPLOT_FILTERS` / `DAMPER_FILTERS` /
  `RIDE_DIL_FILTERS`** — per-channel Butterworth filter settings.
  Use `cutoff=0` to disable. The `"all"` key sets a fallback.
- **`RESAMPLE_RATE`** — uniform resample rate applied to every run (Hz).
- **`SCATTER_MAX_POINTS`** — max points drawn per scatter; data is randomly
  decimated above this.
- **`BAR_SECONDARY_AXIS_RATIO`** — auto-trigger dual y-axis on bar charts
  when value ratios exceed this.
- **`BOX_PLOT_SETTINGS`** — global box-plot styling overrides.

### Calculated channels

```python
CORRELATION_CALCULATED = {
    "gLat_Abs":    lambda df: df["gLat"].abs(),
    "FPRodDeltaF": lambda df: df["FPRodFL"] - df["FPRodFR"],
}
```

Use `calc_channel()` to declare dependencies explicitly when the lambda
body is too dynamic for regex-based auto-detection:

```python
from engine import calc_channel

CALCULATED = {
    "EngineEff": calc_channel("nEngine", "tThrottle")(
        lambda df: df["nEngine"] * df["tThrottle"] / 1000.0
    ),
}
```

### Filters

Per-channel Butterworth filter settings, applied after resampling:

```python
CORRELATION_FILTERS = {
    "hRideF":   {"cutoff": 5, "order": 2},                     # low-pass (default)
    "gVertF":   {"cutoff": 0.5, "order": 4, "type": "high"},   # high-pass
    "nYaw":     {"cutoff": [1, 10], "order": 2, "type": "bandpass"},
    "all":      {"cutoff": 5, "order": 2},                     # fallback
}
```

Set `cutoff=0` to disable filtering for a channel.

---

## Data layout

```
Data/
  inputs/
    correlation/26R04MIA/      ← workflow / event subfolders
    boxplots/26T01BCN/
    dampers/26R04MIA/
    ride_dil/26R05MTL/
  templates/                   ← PowerPoint template files (.pptx)
  outputs/
    correlation/26R04MIA/plots/
    boxplots/26T01BCN/plots/
    ...
```

All directories are created automatically by `get_workflow_dirs(workflow, event)`.

### Cross-event comparisons

Set `EVENT = None` to flatten the root to `Data/inputs/<workflow>/`, then
prefix `file` entries with the event subfolder name:

```python
WORKFLOW_NAME = "correlation"
EVENT = None
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

RUNS = [
    {"name": "MIA LTS", "file": "26R04MIA/my_mia_file.parquet", "color": "#0083BF", "type": "DLS", "nlap": 1},
    {"name": "SUZ LTS", "file": "26R03SUZ/my_suz_file.parquet", "color": "#D70000", "type": "DLS", "nlap": 1},
]
```

Outputs land in `Data/outputs/correlation/plots/`. Use a named event override
to redirect:

```python
_, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, "MIA_vs_SUZ")
```

---

## Defining runs

```python
RUNS = [
    {
        "name":  "Baseline",        # display label used in all plots
        "file":  "my_run.parquet",  # path relative to Data/inputs/<workflow>/<event>/
        "color": "#D70000",         # hex colour (optional — auto-assigned if omitted)
        "type":  "OC",              # OC | CAR | DLS | DIL
        "nrun":  1,                 # (parquet) rank-based run selection
        "nlap":  1,                 # exact lap filter; ignored when nrun is set
        "reference": True,          # workflow-wide baseline (see below)
    },
]
```

`nrun=1` selects the lap with the lowest `nRun` value in the file; `nrun=2`
selects the next lowest, etc.

Set `"reference": True` on exactly one run to mark it as the workflow-wide
baseline. That run is the reference for waveform delta subplots
(`show_delta=True`) and the `tDiff` channel (lap-time difference vs reference
at each `sLap`). If no run is flagged, the first loaded run is used.

### Folder-based runs

To load every matching file in a directory as its own run, replace `file`
with `folder` + `filetype`. The entry is expanded into one run per file at
load time, with names derived from the filename stems and colours auto-shaded
from the `type` colormap.

```python
RUNS = [
    {"folder": "2xStopChoc", "filetype": ".parquet", "type": "DLS", "nlap": 1},
    {"folder": "2xStopChoc", "filetype": ".txt",     "type": "CAR"},
]
```

Optional keys for folder runs:

| Key | Description |
|-----|-------------|
| `contains` | Case-insensitive substring filter. Only files whose names contain this string are loaded. Lets a single folder feed multiple per-condition configurations (e.g. `"FP1"`, `"Q"`). |
| `name_prefix` | String prepended to each auto-generated run name. |
| `colors` | Explicit list of hex colours, cycled per file (overrides auto-shading). |
| `color` | Single hex colour applied to every expanded run. |

Any other keys (`nlap`, `nrun`, `type`, etc.) are forwarded to each expanded
run. An empty folder, an unknown `filetype`, or a `contains` filter with no
matches raises a clear error before plotting starts.

---

## Filtering plots programmatically

The CLI flag `--only NAME [NAME ...]` is the primary interface. For embedded
use:

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
- Warnings (missing channels in some runs, empty gate results, etc.) always print regardless of verbose mode.

---

## PowerPoint export

Set `EXPORT_TO_POWERPOINT = True` in the runner and place a `.pptx` template
in `Data/templates/`. Build the export map declaratively with `Slide()`:

```python
from engine import Slide

POWERPOINT_EXPORT_MAP = [
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("double_plot", "scatter/Gear Ratios", "scatter/Engine Power"),
]
```

`POWERPOINT_START_SLIDE` sets the 1-based slide number where the first entry
is placed (e.g. `4` to leave cover/intro slides untouched):

```python
POWERPOINT_START_SLIDE = 4

run_workflow(
    ...,
    powerpoint_start_slide=POWERPOINT_START_SLIDE,
)
```

Layouts:
- `"main_plot"` — single image, full-width on the slide
- `"double_plot"` — two images side-by-side

Plot references use `"type/Plot Name"` notation and are automatically converted
to filenames (`type_plot_name.png`, all lowercase). The plot figure size is
automatically matched to the template slide aspect ratio.
