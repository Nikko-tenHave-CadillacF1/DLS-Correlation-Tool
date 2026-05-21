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
