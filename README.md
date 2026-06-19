# DLS Correlation Tool

Generate engineering plots from multiple telemetry runs and optionally export
a PowerPoint report. Each runner script is **plug-and-play**: it auto-creates
its own virtualenv, installs dependencies, and runs.

## Setup

```powershell
python Run_Correlation.py        # first run: creates .venv, installs deps, then runs
```

To set things up explicitly instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Parquet input files require at least one of: `pyarrow`, `fastparquet` (both
included in `requirements.txt`).

Set `DLS_SKIP_BOOTSTRAP=1` in your environment to disable the auto-installer
(useful in CI or shared/managed Python environments).

---

## Quickstart

1. Drop input files into `Data/inputs/<workflow>/<event>/` (folders are auto-created).
2. Edit `WORKFLOW_NAME`, `EVENT`, and `RUNS` in the relevant `Run_*.py`.
3. Run it.

```powershell
python Run_Correlation.py
python Run_BoxPlots.py
python Run_Dampers.py
python Run_RideDIL.py
```

Plots are saved to `Data/outputs/<workflow>/<event>/plots/`. The folder opens
automatically on completion.

To start a new workflow from scratch, copy [Run_Template.py](Run_Template.py)
and change `WORKFLOW_NAME` and `EVENT` — all directories are created
automatically. `Run_Template.py` is a tutorial: it demonstrates every plot
type with annotated examples.

### Discover and validate before plotting

If a plot definition references a channel name that exists in **no** loaded
run, the runner exits with a clear error and suggests close matches. To list
what's available in your data:

```powershell
python Run_Correlation.py --list-channels   # channels in each loaded run
python Run_Correlation.py --list-plots      # configured plot names
python Run_Correlation.py --check-only      # data-quality report only
```

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

## File structure

### Files you edit

| File | Purpose |
|---|---|
| [Run_Template.py](Run_Template.py) | Reference / tutorial — every plot type with annotated examples |
| [Run_Correlation.py](Run_Correlation.py) | Correlation plots + PowerPoint export |
| [Run_BoxPlots.py](Run_BoxPlots.py) | Box plots and `BoxPlotGrid` examples |
| [Run_Dampers.py](Run_Dampers.py) | Damper analysis (waveform + scatter) |
| [Run_RideDIL.py](Run_RideDIL.py) | Ride / DIL simulator comparison (PSD) |
| [Run_Vibrations.py](Run_Vibrations.py) | 4-DOF body modal analysis (Heave, Pitch, Roll, Warp) |
| [channel_config.py](channel_config.py) | Project-wide settings: paths, channel mappings, units, transforms, calc channels, filters |

### Engine (do not edit)

Single package under [engine/](engine/). See
[docs/architecture.md](docs/architecture.md) for the module map.

---

## Documentation

- **[docs/plot-reference.md](docs/plot-reference.md)** — every plot type, every field, with examples.
- **[docs/architecture.md](docs/architecture.md)** — job lifecycle, `PlotJobConfig`, `channel_config.py`, data layout, calc channels, filters, cross-event comparisons, PowerPoint export.
- **[tools/README.md](tools/README.md)** — helper scripts (data organisation, config validation).
