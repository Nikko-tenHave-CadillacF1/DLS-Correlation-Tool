# Box Plot Implementation Summary

## Overview
Box plot functionality has been successfully implemented for the DLS Correlation Tool. The implementation supports:
- **Per-run mode**: Compare distributions across runs with one box per run per channel
- **Aggregated mode**: Combine data from all runs into single/multiple boxes for overall distribution analysis
- **Multi-channel support**: Display multiple channels as subplots in a single figure
- **Gate specifications**: Conditional filtering of data (e.g., "only when vCar > 100")
- **Optional point overlay**: Show individual data points with configurable transparency and jitter

## Files Modified/Created

### 1. **datafunctions.py** (MODIFIED)
Added three new box plot utility functions at the end of the file:

#### `apply_gate_to_dataframe(df, gate_spec)`
- Applies gate conditions to a dataframe, returning filtered copy
- Supports operators: `>`, `<`, `>=`, `<=`, `==`, `between`
- Handles single and multiple gate conditions

#### `aggregate_channel_for_boxplot(run_data_dict, channels, aggregation_mode='per_run', gate_spec=None)`
- Core aggregation function for box plots
- Two modes:
  - `'per_run'`: Returns `{run_name: {channel: array}}`
  - `'aggregated'`: Returns `{channel: combined_array}`
- Automatically applies gates before aggregation

#### `get_boxplot_data_for_matplotlib(agg_data, aggregation_mode, channels, run_names)`
- Converts aggregated data to matplotlib.boxplot() compatible format
- Returns `(data_list, labels_list)` tuple

#### `format_boxplot_colors(data_list, aggregation_mode, run_colors, channels, run_names, box_color)`
- Generates color list for boxes based on aggregation mode
- Per-run mode: colors by run
- Aggregated mode: uniform color

### 2. **dataplotter.py** (MODIFIED)
Updated DataPlotter class with box plot support:

#### Constructor Changes
- Updated default `fig_size` from 4 elements to 5 elements: `[(15.5, 6.4), (10, 8), (10, 8), (10, 8), (10, 6)]`
  - Index 0: Waveform figures
  - Index 1: Scatter plots
  - Index 2: PSD plots
  - Index 3: Histogram plots
  - Index 4: **Box plots** (NEW)
- Added `self.boxplot_FIGSIZE` instance variable

#### New Methods

##### `generate_box_plots(self)`
- Main entry point for box plot generation
- Retrieves plot definitions from `PLOT_DEFINITIONS[5]`
- Routes to appropriate generator based on aggregation mode
- Handles error cases gracefully

##### `_generate_boxplot_per_run(self, plot_name, channels, axis_limits, gate_spec, options, figsize, box_settings)`
- Generates per-run box plots
- Creates subplots for multiple channels
- Colors boxes by run color from RUNS configuration
- Displays run names on X-axis
- Optional point overlay with jitter

##### `_generate_boxplot_aggregated(self, plot_name, channels, axis_limits, gate_spec, options, figsize, box_settings)`
- Generates aggregated box plots
- Combines all runs into single box(es)
- Creates subplots for multiple channels
- Optional point overlay colored by run
- Legend displayed if points are shown

#### Updated Methods
- `plot_all()`: Added call to `self.generate_box_plots()` after bar plots

### 3. **Run_BoxPlots.py** (NEW FILE)
New configuration file in the same style as `Run_Correlation.py`:

#### Key Sections

**BOX_PLOT_DEFINITIONS** - Configuration for each box plot:
```python
["Plot Name", "channel" or ("ch1", "ch2", ...), 
 aggregation_mode, axis_limits, gate_spec, options]
```

**BOX_PLOT_SETTINGS** - Rendering control:
- `show_points`: Overlay individual points (default: True)
- `show_fliers`: Show statistical outliers (default: True)
- `jitter`: Point jitter magnitude 0-1 (default: 0.15)
- `point_alpha`: Point transparency (default: 0.25)
- `point_size`: Marker size (default: 20)
- `box_width`: Box width ratio (default: 0.6)
- `box_linewidth`: Border width (default: 1.5)
- `medianline_color`: Median line color (default: black)
- `medianline_width`: Median line width (default: 2.0)
- `aggregated_box_color`: Color for aggregated boxes (default: #3498DB)
- `aggregated_box_alpha`: Aggregated box transparency (default: 0.7)
- `per_run_box_alpha`: Per-run box transparency (default: 0.7)
- `figsize_single_channel`: Size for single channel (default: 10x6)
- `figsize_multi_channel`: Size for multiple channels (default: 14x10)

#### Example Plot Definitions Included
1. **Per-run vCar**: Compare velocity distributions across runs
2. **Aggregated vCar**: Overall velocity distribution with gating
3. **Multi-channel aggregated**: Lateral dynamics (gLat, aSteerWheel, nYaw)
4. **Per-run suspension**: Suspension loads by run

**PLOT_DEFINITIONS** - Tuple structure:
```python
([], [], [], [], [], BOX_PLOT_DEFINITIONS)
```
Index 5 contains box plot definitions; indices 0-4 are empty (waveform, scatter, PSD, histogram, bar)

## Data Flow

```
Run_BoxPlots.py (Configuration)
    ↓
DataPlotter.__init__()
    ├─→ Load all runs (self.run_data = {run_name: dataframe})
    ├─→ Apply channel mappings
    ├─→ Apply transformations
    └─→ Apply calculated channels & filters
    ↓
DataPlotter.plot_all()
    ├─→ generate_waveform_plots() [skipped - empty]
    ├─→ generate_scatter_plots() [skipped - empty]
    ├─→ generate_psd_plots() [skipped - empty]
    ├─→ generate_histogram_plots() [skipped - empty]
    ├─→ generate_bar_plots() [skipped - empty]
    └─→ generate_box_plots() ← NEW
        ├─→ For each plot definition:
        │   ├─→ Parse parameters (channels, mode, gate, options)
        │   ├─→ aggregate_channel_for_boxplot()
        │   │   ├─→ apply_gate_to_dataframe() [if gated]
        │   │   └─→ Extract and combine values
        │   ├─→ Route to generator:
        │   │   ├─→ _generate_boxplot_per_run()
        │   │   └─→ _generate_boxplot_aggregated()
        │   ├─→ Create figure with matplotlib
        │   ├─→ Render boxes
        │   ├─→ Optional point overlay
        │   └─→ Save PNG
        └─→ Print completion message
```

## Coloring Strategy

**Per-Run Mode:**
- Each box colored using the run's color from RUNS configuration
- Run name displayed on X-axis label
- Professional, color-coordinated appearance

**Aggregated Mode:**
- All boxes use uniform color (#3498DB - professional blue-grey)
- Optional point overlay can show individual runs with their colors
- Legend displayed when points are shown

## Usage Example

```python
# In Run_BoxPlots.py configuration:

BOX_PLOT_DEFINITIONS = [
    # Simple per-run comparison
    ["vCar by Run", "vCar", "per_run", (50, 350), None, {"show_points": False}],
    
    # Aggregated with gate
    ["High Speed vCar", "vCar", "aggregated", None, ("vCar", ">", 100), {"show_points": True}],
    
    # Multi-channel aggregated
    ["Dynamics", ("gLat", "nYaw", "aSteerWheel"), "aggregated", None, None, {"show_points": True}],
]

# Then run:
# python Run_BoxPlots.py
```

## Features

✅ **Matplotlib-based** (no seaborn dependency)
✅ **Multi-channel support** (creates subplots automatically)
✅ **Gate specifications** (conditional filtering)
✅ **Point overlay** (with configurable jitter & transparency)
✅ **Run coloring** (uses RUNS configuration colors)
✅ **Aggregation modes** (per_run and aggregated)
✅ **Error handling** (graceful degradation with warnings)
✅ **Consistent styling** (matches existing plots)
✅ **Professional appearance** (proper labels, legends, grid)

## Integration with Existing Code

- ✅ Uses existing `RUNS`, `CHANNEL_MAPPINGS`, `CHANNEL_TRANSFORMS`, `CALCULATED_CHANNELS`, `LOW_PASS_FILTERS`, `UNITS_MAP`
- ✅ Reuses gate specification format from scatter plots
- ✅ Compatible with existing data preprocessing pipeline
- ✅ Uses `_sanitize_plot_filename()` for consistent naming
- ✅ Reuses `add_units_to_label()` for proper axis labels
- ✅ Follows DataPlotter class patterns and conventions

## Testing

All Python files have been syntax-checked successfully:
- ✅ datafunctions.py - No errors
- ✅ dataplotter.py - No errors
- ✅ Run_BoxPlots.py - No errors
