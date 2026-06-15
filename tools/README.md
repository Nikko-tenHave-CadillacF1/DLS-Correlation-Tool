# Tools

Utility scripts for the DLS Correlation Tool.

## `organise_data.py`

Moves raw data files into per-event subfolders based on the event code in
filenames (e.g. `26R03SUZ`, `26T01BCN`).

```bash
python tools/organise_data.py --workflow correlation --dry-run
python tools/organise_data.py --workflow correlation
```

## `validate_config.py`

Validates a `Run_*.py` configuration file without generating plots.

```bash
python tools/validate_config.py Run_Correlation.py
```

## `vibrations.py`

Fits a 4-DOF body dynamics transfer function (Heave, Pitch, Roll, Warp)
to measured FPushrod Power Spectral Densities.

**Preferred usage** — edit settings in `Run_Vibrations.py` and run directly:

```bash
python Run_Vibrations.py
```

Standalone CLI is also available:

```bash
python tools/vibrations.py Data/inputs/ride_dil/26R07BCN/26R07BCN_260612_MAC26-03_BOT_P2_R01.txt
python tools/vibrations.py path/to/file.txt --fmin 1.5 --fmax 15 --no-plots
```
