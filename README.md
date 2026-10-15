# DLS Correlation Tool

Plots telemetry from several runs (CAR, DLS, DIL, OC, FMIOpt) on shared axes
and can export the plots to a PowerPoint report.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` installs everything. For an editable install, choose the
extras you need:

```powershell
pip install -e ".[parquet,pptx,plotly]"   # full
pip install -e .                           # core only
```

| Extra | Package(s) | Without it |
|---|---|---|
| `parquet` | `pyarrow`, `fastparquet` | `.parquet` inputs can't be read (one engine is enough) |
| `pptx` | `python-pptx` | No PowerPoint export |
| `plotly` | `plotly` | 3-D scatters are saved as PNG only, without the interactive `.html` |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `pre-commit` | No tests or linting |

`import engine` checks that pandas, numpy, matplotlib and scipy are installed
and prints an install hint if they're missing. Set `DLS_SKIP_BOOTSTRAP=1` to
skip this check.

`pip install -e .` also adds console scripts that call each runner's `main()`:
`dls-correlation`, `dls-boxplots`, `dls-ridedil`, `dls-ridereport`,
`dls-vibrations`, `dls-oc-checks`, `dls-bumpstop` and `dls-template`.
`Run_DILCorrelation.py` and `Run_DLSCorrelation.py` don't have one.

## Quickstart

1. Put the input files in `Data/inputs/<workflow>/<event>/`.
2. In the runner you want, set `EVENT` and `RUNS`, and edit the plot
   definitions if needed.
3. Run it, e.g. `python Run_Correlation.py`.

Plots are saved to `Data/outputs/<workflow>/<event>/plots/<type>/`, and the
folder opens when the job finishes. Folders are created automatically.

`Data/` is ignored by git. You need to copy in the input data and the
PowerPoint templates (`Data/templates/template.pptx`, `oc_template.pptx`) on
each machine.

To start a new workflow, copy [Run_Template.py](Run_Template.py). It shows
every plot type with annotated examples.

## Runners

| File | Workflow | Purpose |
|---|---|---|
| [Run_Template.py](Run_Template.py) | `template` | Tutorial covering every plot type |
| [Run_Correlation.py](Run_Correlation.py) | `correlation` | CAR vs DLS correlation report (`template.pptx`) |
| [Run_DLSCorrelation.py](Run_DLSCorrelation.py) | `correlation` | Alternative CAR vs DLS correlation preset |
| [Run_DILCorrelation.py](Run_DILCorrelation.py) | `correlation` | CAR vs DIL simulator checks (blank deck) |
| [Run_OC_Checks.py](Run_OC_Checks.py) | `correlation` | CAR vs OC lap-sim report (`oc_template.pptx`) |
| [Run_BoxPlots.py](Run_BoxPlots.py) | `boxplots` | Box plots across sessions and drivers |
| [Run_RideDIL.py](Run_RideDIL.py) | `ride_dil` | CAR vs DIL ride comparison (waveform, scatter, PSD) |
| [Run_RideReport.py](Run_RideReport.py) | `ride_report` | Multi-session ride report with modal fits, plus a modal-evolution deck |
| [Run_Bumpstop.py](Run_Bumpstop.py) | `bumpstops` | Comparison of DLS bumpstop configurations |
| [Run_Vibrations.py](Run_Vibrations.py) | `ride_dil` | Standalone body modal fit (see [Modal analysis](#modal-analysis)) |

Project-wide settings are in [channel_config.py](channel_config.py): paths,
channel mappings, units, transforms, calculated channels, filters and the
resample rate. You shouldn't need to edit [engine/](engine/) for normal use.

To turn off PowerPoint export, set `POWERPOINT_OUTPUT = None`. To get a blank
16:9 deck instead of a template, set `POWERPOINT_TEMPLATE = None`.

## Command line

Every runner except `Run_Vibrations.py` accepts these flags. All of them are
optional.

| Flag | Effect |
|---|---|
| `--only NAME ...` | Generate only the plots with these names (case-insensitive) |
| `--runs NAME ...` | Use only these runs |
| `--no-open` | Don't open the output folder or deck when the job finishes |
| `--list-plots` | Print the configured plot names and exit |
| `--list-channels` | Load the runs, print their channels and exit |
| `--check-only` | Load the data, write the quality report and exit |
| `--dry-run` | Validate the config and show the plan without loading data |

- **Config errors stop the run.** A missing input file or PowerPoint
  template, a duplicate run name or an unknown `type` exits with code 1
  before any data is read.
- **Unknown channels only produce a warning.** The runner prints close matches
  and skips the affected plots.
- A successful run always exits with code 0.

Every run writes `plots/data_quality_report.md`, which covers sample rates,
missing channels, NaN ratios, flatlined channels and `sLap` alignment.

---

## Plot types

Each plot type is a dataclass imported from `engine` and passed to
`run_workflow()` with the matching keyword. Every field is documented in
[docs/plot-reference.md](docs/plot-reference.md), and
[Run_Template.py](Run_Template.py) has a working example of each.

| Dataclass | Keyword | Produces |
|---|---|---|
| `WaveformPlot` | `waveforms=` | Stacked traces against time or distance, with optional delta-to-baseline rows |
| `ScatterPlot` | `scatters=` | X-Y scatter with linear, polynomial, piecewise or robust fits |
| `Scatter3DPlot` | `scatter3d=` | 3-channel scatter as PNG, plus interactive HTML if `plotly` is installed |
| `PsdPlot` | `psds=` | Welch PSD with optional Lorentzian peak fits (f₀, ζ) |
| `HistogramPlot` | `histograms=` | Channel distribution per run |
| `BarPlot` | `bars=` | Aggregated metrics per run, with optional error bars and deltas |
| `BoxPlot` / `BoxPlotGrid` | `boxes=` | Box plots per run, pooled, or in a grid by condition |
| `HeatmapPlot` | `heatmaps=` | 2-D density or aggregation per bin |

---

## Modal analysis

[Run_Vibrations.py](Run_Vibrations.py) fits Heave, Pitch, Roll and Warp body
modes to the four FPushrod corner forces. Set `DISPLACEMENT_MODE = True` to
fit damper displacements instead. Edit `RUNS` and the settings block, then run
the file. It doesn't take command-line flags.

| Setting | Default | Meaning |
|---|---|---|
| `F_MIN` / `F_MAX` | 2 / 13 Hz | Fit window; keep `F_MAX` below wheel hop (~15 Hz) |
| `EXPECTED_FREQS` | Band per mode | Soft hints; the fit can leave a band if the data supports it. Set to `None` to disable |
| `NPERSEG` | 512 | Welch segment length, as an int or `"auto"`. Aim for at least 30 averages (≈ `2·N/NPERSEG − 1`) |
| `METHOD` | `lorentzian_combined` | Fast Lorentzian fit with f₀ and ζ shared between front and rear. `body4dof` adds a two-axle model (needs damper pots): front/rear axle rate k, damping c and masses, with each mode labelled by whether front and rear move in or out of phase |

Outputs are written to `Data/outputs/ride_dil/<event>/plots/vibrations/`:
`vibrations_diag_<run>.png`, `vibrations_comparison_fit.png` and, for
`body4dof`, `vibrations_axle_<run>.png` / `vibrations_comparison_axle.png`
(measured force/travel response against the fitted k and c per axle). Every
figure has one row per body DOF; comparisons overlay all runs in their colours.

You can also run the same fit inside a plotting workflow by passing
`vibrations_fit=` (see [Run_RideReport.py](Run_RideReport.py)). Each result
becomes a set of constant channels (`modal_<mode>_f0`, `modal_<mode>_zeta`
and their uncertainty channels) that `BarPlot` can use. `body4dof` also adds
`modal_<k|c|m>_<heave|roll>_<F|R|FR>` and matching `_sigma` channels.
`plot_modal_evolution()` then plots how the modes change across sessions.

To fit a single file: `python -m engine.vibrations_io <file> --fmin 2 --fmax 13`.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `Configuration validation failed` | Input file or template not found, duplicate run name, or unknown `type` | Check `EVENT` and the `file` paths in `RUNS`. `type` must be `CAR`, `OC`, `DLS`, `DIL` or `FMIOpt` |
| `...reference channels that exist in no loaded run` | A channel name is misspelled, or the source doesn't have that channel | Run with `--list-channels`, then fix the name or add a calculated channel in [channel_config.py](channel_config.py) |
| A plot is empty or has very few points | A `gate` removed most of the data | Relax or remove the `gate`, then re-run with `--only "<Plot Name>"` |
| Traces from different loggers don't line up | Different `sLap` offset or scale between sources | Alignment is automatic; check the `sLap Alignment Estimate` section of `data_quality_report.md` |
| `PowerPoint export skipped: ... is open in another application` | The deck is open in PowerPoint | Close the deck and re-run. The plots were still generated |
| A `.parquet` run fails to load | No parquet engine is installed | `pip install pyarrow` (or `fastparquet`) |
| `Run_Correlation.py` / `Run_DLSCorrelation.py` doesn't finish | After the run, the 3-D scatters reopen in an interactive window | Close the window |

---

## Development

```powershell
pip install -e ".[dev]"
pytest          # tests + coverage (writes coverage.xml)
ruff check .
```

CI runs `pytest` on Windows and Ubuntu (Python 3.10 and 3.12), `ruff check`,
and a per-file coverage guard (see [tests/golden/README.md](tests/golden/README.md)).

## Documentation

- [docs/plot-reference.md](docs/plot-reference.md): every plot type and field, with examples.
- [docs/architecture.md](docs/architecture.md): job lifecycle, `PlotJobConfig`, `channel_config.py`, calculated channels, filters, PowerPoint export.
- [tools/README.md](tools/README.md): helper scripts (data sorting, config check, FARB deadband tables, golden-output regression).
