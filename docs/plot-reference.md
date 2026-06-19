# Plot Reference

Every plot type is a dataclass with named-argument constructors and
`__post_init__` validation. Imports:

```python
from engine import (
    WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot,
    BarPlot, BoxPlot, BoxPlotGrid, HeatmapPlot, Marker, calc_channel,
)
```

The fields shown below are the full set; most are optional with sensible
defaults. See [Run_Template.py](../Run_Template.py) for working examples of
every plot type in one place.

---

## Waveform

```python
WaveformPlot(
    name="Driver Input",
    channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel'),
    # One entry per row. Use ('left_ch', 'right_ch') for dual-axis rows.
    axis_limits=(None, ((60, 400), (-1, 9)), (-160, 160)),
    # Per-row y-limits. Dual-axis: ((y1_min, y1_max), (y2_min, y2_max)).
    reference_lines=((-350, 0, 350), None, (0,)),
    # Per-row horizontal reference lines.
    subplot_heights=(0.4, 0.8, 0.4),
    # Relative row heights; 0.8 is twice as tall as 0.4.
    x_channel="sLap",
    # X-axis channel. Default "sLap" (distance). Use "tLap" for elapsed time.
    x_limits=(0, 4000),
    # Optional (x_min, x_max) to zoom to a section.
    highlight_zones=('SM', '<', 0.3),
    # Shade x-regions where the gate condition is true.
    # Each run is evaluated against its own data and shaded in a highly
    # transparent version of that run's own colour.
    # Single condition: ('channel', 'operator', value)
    # Multiple (all must match): [('ch1', '>', v1), ('ch2', '<', v2)]
    # Operators: '>' '<' '>=' '<=' '==' 'between'
    normalise=False,
    # If True, all channels on each subplot are normalised to [0, 1] using
    # the global min/max across all runs. Dual-axis becomes single-axis.
    legend_position="top",
    # "top" (default): run legend above the subplots.
    # "right": vertical legend to the right of the plot area.
    show_delta=False,
    # Controls per-row delta subplots (requires 2+ loaded runs):
    #   False             → no delta rows
    #   True              → append delta row below every primary row
    #   (True, False, ..) → per-row control; tuple must match len(channels)
    # Each active delta row shows (run_i − reference) for every non-reference
    # run, in that run's colour. The reference is selected workflow-wide via
    # ``"reference": True`` on a RUN entry (not per-plot).
    markers=[
        # Vertical reference lines. Two flavours:
        #   • Static  — fixed x position, drawn once on every run.
        Marker(x=1500, label="T1 entry", color="#FF6600", linestyle="--"),
        Marker(x=2800, label="T-final", row=0),  # row=0 → only on the top subplot
        #   • Condition — per-run; one marker is emitted at the x_channel
        #     value of each rising / falling / either transition of a gate
        #     condition. The line is drawn in the run's colour so DRY and WET
        #     are visually distinguishable without a run-name in the label.
        Marker(condition=('SM', '>', 0.5), edge="rising", label="SM>0.5"),
        Marker(
            condition=[('pBrakeF', '>', 50), ('vCar', '>', 100)],
            edge="rising",          # 'rising' | 'falling' | 'both'
            label="hard brake",
            max_count=3,            # cap markers per run (first N kept)
            show_label=False,       # suppress label text; line is still drawn
            linestyle="-.",
        ),
    ],
)
```

### Marker rules

- Each `Marker` requires **exactly one** of `x=...` (static) or `condition=...` (per-run).
- `edge` only applies to condition markers: `"rising"` (default), `"falling"`, or `"both"`.
- A condition that is already true at the first sample is **not** counted as a rising edge (true transition required).
- `row=N` restricts the marker to the N-th subplot row; default draws on every row.
- `color=None` (default): static markers fall back to grey, condition lines use the run colour.
- Condition markers are silently skipped on non-waveform plot types (scatter / PSD / etc. accept only static `x=...` markers).

---

## Scatter

```python
ScatterPlot(
    name="Front Ride vCar",
    x_channel="vCar",
    y_channel="hRideF",
    axis_limits=[(0, 350), (None, None)],   # [(x_min, x_max), (y_min, y_max)]
    best_fit=[('SM', 0, 0.5)],
    # Trend line options:
    #   None / 0              → no fit
    #   1                     → single linear fit across all data
    #   2                     → single quadratic (2nd-order polynomial) fit
    #   [('ch', low, high)]   → segmented fits by channel value range
    #   [('x'/'y', low, high)]→ segmented fits by axis value range
    gate=('SM', '<', 1),
    # Data filter applied before plotting.
    # Single: ('channel', 'operator', value)
    # Multi (all must match): [('ch1', '>', v1), ('ch2', '<', v2)]
    # Operators: '>' '<' '>=' '<=' '==' '!=' 'between' 'outside'
    show_equations=True,
    show_error=True,                  # gradient delta between runs
    error_as_factor=False,            # True → "× 1.10" instead of "+10.0%"
    color_gate=('SM', '<', 0.5, '#00AAFF'),
    # Highlight a subset of points with a second color. Fit line still uses
    # all (gated) data.
    annotate_fit_at=250.0,
    # Draw a vertical dashed line at this x-value and annotate each run's
    # fit-line y-value at that x. Only works with single fits (best_fit=1).
    robust=False,
    # If True with best_fit=1, uses Theil-Sen + MAD outlier rejection.
    # Outliers are drawn as faint grey 'x' markers and logged in the
    # data-quality report. ``robust_threshold`` (default 3.0) sets the
    # MAD multiplier for outlier rejection.
    reference_lines=[0.0],            # horizontal y-value reference lines
    markers=[Marker(x=250, label="250 km/h")],
)
```

---

## PSD

```python
PsdPlot(
    name="Front Ride PSD",
    channel="hRideF (raw)",
    # Also accepts a list to overlay multiple channels on the same plot:
    # channel=["hRideF (raw)", "hRideR (raw)"]
    # Legend entries become "RUNNAME — channel".
    axis_limits=[(0, 50), (1e-4, None)],    # [(f_min, f_max), (power_min, power_max)]
    log_scale=True,         # semilogy axis (default True)
    nperseg=256,            # Welch window length override (≥8); None uses default 512
    gate=[('vCar', '>', 100)],
    # Segment-aware Welch — only segments satisfying the gate contribute to
    # the PSD. Segments shorter than nperseg are discarded.
    show_envelope=False,    # ±1σ shading when multiple runs are present
    annotate_at=(5, 15),    # annotate PSD values at these frequencies
    reference_lines=[1e-3], # horizontal y-value reference lines
    lorentz_fit=[5.0, 15.0],
    # Fit a single-DOF Lorentzian + baseline near each f₀ and overlay it as a
    # dotted line in the curve's colour, annotated with the estimated damping
    # ratio ζ. Default window is ±25% of f₀ (floored at 1 Hz); f₀ is bounded
    # to within ±5% of the user value so the fit cannot slide onto a
    # neighbouring peak. Pass (f0, half_width_hz) tuples to override the
    # window, e.g. lorentz_fit=[(5.0, 1.0), (15.0, 2.0)].
    markers=[Marker(x=10, label="10 Hz")],
)
```

---

## Histogram

```python
HistogramPlot(
    name="Plank Power Distribution",
    channel="PPlank_F",
    axis_limits=[(1, 51), (None, None)],    # [(bin_min, bin_max), (count_min, count_max)]
    log_scale=False,        # True for log-scale y-axis
    gate=('vCar', '>', 100),
    reference_lines=[100.0],
    markers=[Marker(x=25, label="Target")],
)
```

---

## Bar

```python
BarPlot(
    name="Cumulative Fuel",
    metrics=(("dmInjector (kg/s)", "integral"),),
    # Each entry: "channel" (uses default_aggregation) or ("channel", "aggregation")
    # Aggregations: "integral" "abs_integral" "sum" "abs_sum"
    #               "mean" "median" "max" "min" "first" "last"
    default_aggregation="last",
    axis_limits=(0, 15),
    reference_lines=[10.0, 12.5],
    # Draw one or more horizontal dashed reference lines at these y-values.
    gate=("vCar", "<", 100),
)
```

---

## Box

```python
BoxPlot(
    name="Low Speed Corner Distribution",
    channels=["xDamperFL", "xDamperFR"],
    aggregation_mode="per_run",   # "per_run" | "aggregated" | "per_run_aggregated"
    axis_limits=(0, 30),
    gate=('vCar', '<', 120),
    reference_lines=[0.0],
)
```

| Mode | Behaviour |
|------|-----------|
| `per_run` | One box per run, coloured by run colour |
| `aggregated` | All runs merged into a single box per channel |
| `per_run_aggregated` | Per-run boxes followed by a combined "ALL" box (separated by a dashed line) |

---

## BoxPlotGrid

A 2D matrix of box plots defined by row and column gate dimensions. Each cell
combines the row + column gate conditions (AND-ed together).

```python
BoxPlotGrid(
    name="Ride Height Grid",
    channels="hRideF",
    rows={
        "LS": [("vCar", "<", 120)],
        "MS": [("vCar", ">=", 120), ("vCar", "<", 200)],
        "HS": [("vCar", ">=", 200)],
    },
    cols={
        "Entry": [("gLong", "<", -0.5)],
        "Apex":  [("gLong", "between", (-0.5, 0.5))],
        "Exit":  [("gLong", ">", 0.5)],
    },
    aggregation_mode="aggregated",
    render_mode="grid",       # "grid" → subplot matrix | "expand" → one file per cell
    axis_limits=(20, 80),
)
```

| Render mode | Behaviour |
|-------------|-----------|
| `expand` | One individual BoxPlot figure per grid cell (default) |
| `grid` | A single figure with a rows×cols subplot matrix |

---

## Heatmap

Two-dimensional density or aggregation grids. One panel per run, with a shared
colour scale so panels are directly comparable.

```python
HeatmapPlot(
    name="gLat vs gLong density",
    x_channel="gLat",
    y_channel="gLong",
    z_channel=None,             # None → 2D-histogram (counts per bin)
    aggregation="mean",         # used only when z_channel is set:
                                # "mean" | "median" | "std" | "count" | "sum" | "max" | "min"
    bins=100,                   # int, or (nx, ny) for non-square grids
    axis_limits=[(None, None), (None, None)],
    cmap="viridis",
    z_limits=(0, 50),           # clamp colour bar range
    min_count=3,                # cells with fewer points are masked
    gate=('SM', '<', 1),
    markers=[Marker(x=0, label="Centre")],
)
```
