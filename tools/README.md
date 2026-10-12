# Tools

Helper scripts. Run them from the repository root.

| Script | Purpose |
|---|---|
| [organise_data.py](organise_data.py) | Sorts loose input files into event folders |
| [validate_config.py](validate_config.py) | Checks that a `Run_*.py` file imports cleanly |
| [farb_deadband.py](farb_deadband.py) | Builds front-ARB deadband lookup tables from CAR data |
| [golden_regression.py](golden_regression.py) | Compares a runner's output hashes before and after a change |

## organise_data.py

Moves files from `Data/inputs/<workflow>/` into `<event>/` subfolders, using
the event code in the filename (`26R03SUZ`, `26T01BCN`, ...). It leaves a file
where it is if the filename has no code or the target file already exists.

```powershell
python tools/organise_data.py --workflow correlation --dry-run
python tools/organise_data.py --workflow correlation
```

## validate_config.py

```powershell
python tools/validate_config.py Run_Correlation.py
```

Imports the runner and reports any error raised at import time, such as a
syntax error or a bad plot definition. It doesn't call `main()`, so it doesn't
check input files, folder expansion or the export map. For a full check, run
`python Run_Correlation.py --dry-run`.

## farb_deadband.py

Builds front-ARB lookup tables in the same format as `FARB_8000.json`:
`M [Nm] = f(aARBFL, aARBFR)`, with breakpoints in rad. Each table has a low
gradient inside ±deadband and a high gradient outside it.

Run `python tools/farb_deadband.py`. It takes no arguments; change settings in
the `SETTINGS` block at the top of the file.

- **Inputs** (in `tools/`): `FARB_8000.json` is the template table.
  `ARBS with no TB.csv` lists the FARB used in each session. The tool
  processes every FARB in the CSV using the CAR runs that match it.
  `FARB_VALUES` limits which FARBs are processed. `CAR_FILES` overrides the run
  selection and only works with a single FARB.
- **Fit**: the low gradient, high gradient and deadband half-width are fitted
  jointly to all selected runs on `xDamperDeltaF` vs `FPRodDeltaF`. Each run
  keeps its own sensor-zero offset and heave-crosstalk coefficient. To fix a
  value instead of fitting it, set `K_LOW_N_PER_MM`, `K_HIGH_N_PER_MM` or
  `DEADBAND_HALF_WIDTH_MM`.
- **Filters** (all in `SETTINGS`): `vCar ≥ 100 kph`,
  `|vDamperDeltaF| ≤ 10 mm/s`, duplicate files removed, heave-crosstalk
  compensation and a 3σ outlier trim.
- **Units**: `DLS_FILE`, a DLS run on the template table, converts plot units
  (N/mm, mm) to table units (Nm/rad, rad). If that file is missing, built-in
  fallback values are used.
- **Outputs**: `Data/outputs/farb_deadband/FARB_<k>_deadband.json` and
  `plots/FARB_<k>_deadband.png`, overwritten on each run, plus a summary in the
  console. FARBs with no usable CAR data are skipped. The bilinear table can't
  reproduce the deadband kink exactly; the tool reports the maximum
  interpolation error, and `GRID_POINTS` trades table size against it.

## golden_regression.py

Runs a runner headlessly and hashes every output PNG and PPTX, so you can
confirm a refactor hasn't changed any plot. See
[tests/golden/README.md](../tests/golden/README.md).

```powershell
python tools/golden_regression.py snapshot Run_Correlation
python tools/golden_regression.py verify   Run_Correlation
```

## Vibrations

The modal-fit code is now in `engine/` (`vibrations_io.py`,
`vibrations_lorentz.py`, `vibrations_body4dof.py`). See
[Modal analysis](../README.md#modal-analysis) in the main README.
