# DLS-Correlation-Tool

This tool compares one DLS run and one track run, standardizes channel names, applies preprocessing, and exports waveform and scatter plots to `Data/plots`.

The main file you edit is [DLS_Correlation.py](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/DLS_Correlation.py).

**Files**
- [DLS_Correlation.py](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/DLS_Correlation.py): run selection and user configuration
- [dataplotter.py](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/dataplotter.py): loading, preprocessing, and plotting workflow
- [datafunctions.py](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/datafunctions.py): helper functions for mappings, filtering, fits, and utilities
- [requirements.txt](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/requirements.txt): Python dependencies

**Install**
```powershell
pip install -r requirements.txt
```

**Run**
```powershell
python DLS_Correlation.py
```

Generated plots are saved in `Data/plots`.

If `EXPORT_TO_POWERPOINT = True`, the script generates a filled report as `Data/DLS_Correlation_Report.pptx` and automatically opens it when complete.

**Typical Workflow**
1. Put the DLS and track export files in `Data`.
2. Set `DLS_RUN` and `TRACK_RUN`.
3. Update `CHANNEL_MAPPINGS` so equivalent channels share one name.
4. Update `CHANNEL_TRANSFORMS` if signs, units, or offsets differ.
5. Add any shared derived channels in `CALCULATED_CHANNELS`.
6. Set `LOW_PASS_FILTERS`.
7. Define the waveform and scatter plots you want.
8. Define any PSD plots you want in `PSD_PLOT_DEFINITIONS`.
9. Run the script and review the images in `Data/plots`.
10. If PowerPoint export is enabled, review the generated `.pptx` report in `Data`.

**Key Configuration Blocks**

**`DLS_RUN` / `TRACK_RUN`**
Choose the two files to compare and the plot color for each dataset.

The `file` can be either a legacy text export such as `.txt` or a `.parquet` file.

**`POWERPOINT_TEMPLATE` / `POWERPOINT_OUTPUT` / `EXPORT_TO_POWERPOINT`**
Configure PowerPoint report generation:
- `POWERPOINT_TEMPLATE`: path to the blank template `.pptx`
- `POWERPOINT_OUTPUT`: path where the filled report will be saved
- `EXPORT_TO_POWERPOINT`: enable/disable PowerPoint export (default: `True`)

When enabled, the script automatically opens the generated report on completion.

**`CHANNEL_MAPPINGS`**
Use this to rename raw channels into a shared internal naming convention.

Example:
```python
'dls': {'FPushrodFL': 'FProdFL'},
'track': {'FPRodFL': 'FProdFL'}
```

**`CHANNEL_TRANSFORMS`**
Use this for per-source value changes after mapping, such as:
- sign inversion
- unit conversion
- distance offset correction

Example:
```python
'dls': {'FProdFL': lambda x: -x},
'track': {'PMGUK': lambda x: x / 1000}
```

**`CALCULATED_CHANNELS`**
Use this for derived signals that should be created for both datasets after mapping and cleaning.

Example:
```python
'FProdDeltaF': lambda df: df['FProdFL'] - df['FProdFR']
```

**`UNITS_MAP`**
Maps channel name patterns (case-insensitive, partial match) to their display units on plots.

Example:
```python
UNITS_MAP = {
    'glat': 'g',
    'fprod': 'N',
    'vcar': 'kph',
    'aroll': '°'
}
```

Channels containing these patterns will automatically have their units displayed in axis labels.

**`LOW_PASS_FILTERS`**
Controls low-pass filtering. `cutoff: 0` means no filtering. `'all'` applies to every remaining numeric channel.

Example:
```python
LOW_PASS_FILTERS = {
    'SM': {'cutoff': 0, 'order': 2},
    'all': {'cutoff': 5, 'order': 2},
}
```

**`WAVEFORM_PLOT_DEFINITIONS`**
Defines multi-panel line plots.

Format:
```python
[
    "Plot Name",
    ('channel1', 'channel2'),
    ((ymin1, ymax1), (ymin2, ymax2)),
    (reference_lines1, reference_lines2),
    (height_ratio1, height_ratio2)
]
```

**`SCATTER_PLOT_DEFINITIONS`**
Defines scatter plots and optional fit lines.

Format:
```python
["Plot Name", ('x channel', 'y channel'), [(xmin, xmax), (ymin, ymax)], best_fit_mode, split]
```

`best_fit_mode`:
- `0`: scatter only
- `1`: single linear fit
- `2`: split fit

For split fits, use:
- `('x', value)`
- `('y', value)`

**`PSD_PLOT_DEFINITIONS`**
Defines PSD plots comparing the same channel between DLS and track using Welch's method.

Format:
```python
["Plot Name", 'channel', [(xmin, xmax), (ymin, ymax)], nperseg(optional), log_scale(optional)]
```

- `nperseg`: Window length for Welch's method (default: 256)
- `log_scale`: Whether to use logarithmic y-axis (default: `True`). Set to `False` for linear/absolute scale.

**`POWERPOINT_EXPORT_MAP`**
Maps generated plot images into the PowerPoint template by slide number and layout.

Supported layouts:
- `main_plot`
- `double_plot`

**How The Script Works**
The plotting flow in [dataplotter.py](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/dataplotter.py) is:

1. Load both files and read headers/units.
2. Apply `CHANNEL_MAPPINGS`.
3. Apply `CHANNEL_TRANSFORMS`.
4. Clean data by removing string columns, coercing numeric values, replacing sentinel values, and interpolating gaps.
5. Create `CALCULATED_CHANNELS`.
6. Apply `LOW_PASS_FILTERS`.
7. Generate waveform plots.
8. Generate scatter plots.
9. Generate PSD plots.
10. If enabled, open the PowerPoint template and place the latest plot images into the configured slides.

A few useful details:
- waveform plots use `sLap` as the x-axis when available, otherwise sample index
- negative `sLap` values are masked from waveform plots
- waveform plots are saved without a plot title
- scatter plots can use no fit, one fit, or split fits
- PSD plots use Welch's method on the processed channels
- zero reference lines on scatter plots are only drawn when zero is inside the plotted axis range
- the default PowerPoint export places the `gVertF` and `gVertR` PSD plots on the Ride/PSD slide

**Output**
Plots are saved automatically into:

```text
Data/plots/
```

Examples:
- `waveform_Driver_Input.png`
- `scatter_Rear_Heave.png`

**Troubleshooting**

**Channel not found**
- check the raw column name in the file header
- confirm the mapping exists in the correct source block
- confirm the plot definition uses the mapped name exactly

**Calculated channel missing**
- one of the source channels used by the lambda is probably missing after mapping
- check the runtime warnings

**Unexpected axis scaling**
- check explicit axis limits in `SCATTER_PLOT_DEFINITIONS`
- check whether zero is relevant to that plot

**Corrupted values**
- check the raw export for sentinel or invalid trailing values
- the script already sanitizes common integer sentinel values before filtering

Most changes you will make for new comparisons should stay inside [DLS_Correlation.py](/abs/path/c:/GitHub_Local/DLS-Correlation-Tool/DLS_Correlation.py).
