# DLS Correlation Tool

Generate engineering plots from multiple telemetry runs and optionally export
a PowerPoint report.

## Setup

Recommended for engineers — set up the environment once, then run any
`Run_*.py`. The runners no longer auto-install on first run; they check that
dependencies are importable and print an install hint if not.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # full install (recommended)
python Run_Correlation.py
```

`requirements.txt` is the recommended install path — it pulls in the full
set (core + parquet + PowerPoint + 3-D HTML export + tqdm). Alternatively:

```powershell
pip install -e ".[parquet,pptx,plotly]"   # editable install with extras
pip install -e .                           # minimal (core only)
```

Optional extras and what you lose without them:

| Extra | Package(s) | Needed for |
|---|---|---|
| `parquet` | `pyarrow`, `fastparquet` | reading `.parquet` inputs (either engine is enough) |
| `pptx` | `python-pptx` | PowerPoint export |
| `plotly` | `plotly` | the interactive `.html` beside each 3-D scatter PNG |

`tqdm` (progress bars) is a core dependency but the tool runs without it.

### Alternative: console-script entry points

`pip install -e .` also installs a set of console scripts that call the same
`Run_*.py` `main()` functions. Useful for CI, packaged installs, or when
`.venv\Scripts\` is on PATH:

```powershell
dls-correlation          # equivalent to: python Run_Correlation.py
dls-boxplots             # equivalent to: python Run_BoxPlots.py
dls-ridedil              # equivalent to: python Run_RideDIL.py
dls-ridereport           # equivalent to: python Run_RideReport.py
dls-vibrations           # equivalent to: python Run_Vibrations.py
dls-oc-checks            # equivalent to: python Run_OC_Checks.py
dls-bumpstop             # equivalent to: python Run_Bumpstop.py
dls-template             # equivalent to: python Run_Template.py
```

The `Run_*.py` files remain the primary user-editable configuration; the
console scripts are just shortcuts.

### Environment overrides

- `DLS_SKIP_BOOTSTRAP=1` — skip the built-in dependency check at
  `import engine` time (used by CI where the environment is already
  provisioned).

---

## Quickstart

1. Drop input files into `Data/inputs/<workflow>/<event>/` (folders are auto-created).
2. Edit `WORKFLOW_NAME`, `EVENT`, and `RUNS` in the relevant `Run_*.py`.
3. Run it.

```powershell
python Run_Correlation.py
python Run_BoxPlots.py
python Run_RideDIL.py
```

Plots are saved to `Data/outputs/<workflow>/<event>/plots/`. The folder opens
automatically on completion.

To start a new workflow from scratch, copy [Run_Template.py](Run_Template.py)
and change `WORKFLOW_NAME` and `EVENT` — all directories are created
automatically. `Run_Template.py` is a tutorial: it demonstrates every plot
type with annotated examples.

### Discover and validate before plotting

```powershell
python Run_Correlation.py --list-channels   # channels in each loaded run
python Run_Correlation.py --list-plots      # configured plot names + channels
python Run_Correlation.py --check-only      # data-quality report only
python Run_Correlation.py --dry-run         # preview the plan, no data load
```

Two different checks run automatically on every job:

- **Config validation is fatal.** Missing/duplicate run names, a missing input
  file, an unknown `type`, or a missing PowerPoint template abort the run with
  exit code 1 before any data is read.
- **Unknown channel names are a warning.** If a plot references a channel that
  exists in no loaded run, the runner prints the name with close-match
  suggestions and carries on; affected plots are skipped. Use
  `--list-channels` to see what your data actually contains.

A successful run always exits 0, so the runners are safe to chain in scripts.

### CLI

The CLI is intentionally minimal — the runners are designed to work without
any flags. The full set:

| Flag | Effect |
|------|--------|
| `--only NAME [NAME ...]` | Generate only plots whose name matches (case-insensitive) |
| `--runs NAME [NAME ...]` | Restrict to a subset of configured runs by name |
| `--no-open` | Don't auto-open the output folder after completion |
| `--list-plots` | Print all configured plot names and exit |
| `--list-channels` | Load each run and print available channel names, then exit |
| `--check-only` | Run data-quality checks and exit without plotting |
| `--dry-run` | Preview what would be generated (no data load, no plots) |

---

## Plot types

Each type is a dataclass imported from `engine` and passed to `run_workflow()`
via the matching keyword. Full field documentation is in
[docs/plot-reference.md](docs/plot-reference.md); working examples of all of
them are in [Run_Template.py](Run_Template.py).

| Dataclass | `run_workflow` keyword | Produces |
|---|---|---|
| `WaveformPlot` | `waveforms=` | Stacked time/distance traces, optional delta-vs-baseline rows |
| `ScatterPlot` | `scatters=` | X-Y scatter with linear / polynomial / piecewise / robust fits |
| `Scatter3DPlot` | `scatter3d=` | 3-channel scatter (PNG, plus interactive HTML with `plotly`) |
| `PsdPlot` | `psds=` | Welch PSD, optional Lorentzian peak fits (f₀, ζ) |
| `HistogramPlot` | `histograms=` | Per-run channel distributions |
| `BarPlot` | `bars=` | Aggregated metrics per run, optional errorbars and deltas |
| `BoxPlot` / `BoxPlotGrid` | `boxes=` | Box-and-whisker, per-run / pooled / gridded by condition |
| `HeatmapPlot` | `heatmaps=` | 2-D density or per-bin aggregation |

PNGs are written to `Data/outputs/<workflow>/<event>/plots/<type>/`.

---

## File structure

### Files you edit

| File | Purpose |
|---|---|
| [Run_Template.py](Run_Template.py) | Reference / tutorial — every plot type with annotated examples |
| [Run_Correlation.py](Run_Correlation.py) | Correlation plots + PowerPoint export |
| [Run_DLSCorrelation.py](Run_DLSCorrelation.py) | DLS-focused correlation preset |
| [Run_DILCorrelation.py](Run_DILCorrelation.py) | DIL simulator correlation preset |
| [Run_BoxPlots.py](Run_BoxPlots.py) | Box plots and `BoxPlotGrid` examples |
| [Run_RideDIL.py](Run_RideDIL.py) | Ride / DIL simulator comparison (PSD) |
| [Run_RideReport.py](Run_RideReport.py) | Ride report + modal-evolution deck |
| [Run_Vibrations.py](Run_Vibrations.py) | 4-DOF body modal analysis (Heave, Pitch, Roll, Warp) |
| [Run_OC_Checks.py](Run_OC_Checks.py) | OC data-quality checks |
| [Run_Bumpstop.py](Run_Bumpstop.py) | Bumpstop characterisation |
| [channel_config.py](channel_config.py) | Project-wide settings: paths, channel mappings, units, transforms, calc channels, filters |

### Engine (do not edit)

Single package under [engine/](engine/). See
[docs/architecture.md](docs/architecture.md) for the module map.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `Configuration validation failed` | Input file not found, duplicate run name, or unknown `type` | Check `EVENT` and the `file` paths in `RUNS`; `type` must be `CAR`, `OC`, `DLS`, `DIL`, or `FMIOpt` |
| `references channels that exist in no loaded run` | Channel typo, or the source genuinely lacks that channel | `--list-channels`, then fix the name or add a calculated channel in [channel_config.py](channel_config.py) |
| A plot is empty or has far fewer points than expected | A `gate` filtered out everything | Relax or remove the `gate`, re-run with `--only "<Plot Name>"` |
| Traces from different loggers don't line up | `sLap` offset/scale between sources | Alignment is automatic; check the `sLap Alignment Estimate` section of `data_quality_report.md` |
| `'<name>.pptx' is open in another application` | The deck is locked by PowerPoint | Close it and re-run; the plots still generated and the rest of the job completed |
| Parquet run fails to load | No parquet engine installed | `pip install pyarrow` (or `fastparquet`) |

Every run also writes `plots/data_quality_report.md` with sample rates,
missing channels, NaN ratios, flatlined channels, and `sLap` diagnostics.

---

## Documentation

- **[docs/plot-reference.md](docs/plot-reference.md)** — every plot type, every field, with examples.
- **[docs/architecture.md](docs/architecture.md)** — job lifecycle, `PlotJobConfig`, `channel_config.py`, data layout, calc channels, filters, cross-event comparisons, PowerPoint export.
- **[tools/README.md](tools/README.md)** — helper scripts (data organisation, config validation).
