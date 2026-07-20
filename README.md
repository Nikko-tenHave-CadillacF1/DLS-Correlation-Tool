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
pip install -e .            # or: pip install -r requirements.txt
python Run_Correlation.py
```

Parquet input files require at least one of: `pyarrow`, `fastparquet` (both
included in `requirements.txt`).

### Alternative: console-script entry points

`pip install -e .` also installs a set of console scripts that call the same
`Run_*.py` `main()` functions. Useful for CI, packaged installs, or when
`.venv\Scripts\` is on PATH:

```powershell
dls-correlation          # equivalent to: python Run_Correlation.py
dls-boxplots             # equivalent to: python Run_BoxPlots.py
dls-dampers              # equivalent to: python Run_Dampers.py
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

- `DLS_SKIP_BOOTSTRAP=1` — skip the dependency check entirely (used by CI
  where the environment is already provisioned).
- `DLS_ENABLE_AUTO_VENV=1` — opt IN to the legacy auto-venv escape hatch:
  the first run auto-creates `.venv`, `pip install`s `requirements.txt`, and
  re-execs into the venv Python. Useful for double-click / hand-off scenarios
  where the user does not have a preconfigured environment.

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
