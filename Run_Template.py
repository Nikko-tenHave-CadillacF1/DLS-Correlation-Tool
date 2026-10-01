"""
TEMPLATE — reference configuration for the DLS Correlation Tool.

Copy this file as a starting point for a new workflow. Every section
demonstrates the current options for the corresponding plot type. Comment
out or delete anything you don't need.

CLI (every ``Run_*.py`` picks up the same flags automatically):

    python Run_Template.py                     # normal run
    python Run_Template.py --only "GG Plot"    # regenerate one plot
    python Run_Template.py --runs "Test Run 4" # only a subset of runs
    python Run_Template.py --list-plots        # print plot names + exit
    python Run_Template.py --list-channels     # print channels per run + exit
    python Run_Template.py --check-only        # load + quality-check, no plots
    python Run_Template.py --dry-run           # preview the plan, no data load
    python Run_Template.py --no-open           # don't pop the output folder open
"""

from channel_config import get_workflow_dirs
from engine import (
    BarPlot,
    BoxPlot,
    BoxPlotGrid,
    HeatmapPlot,
    HistogramPlot,
    Marker,
    PsdPlot,
    Scatter3DPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)

# ─── WORKFLOW NAME & EVENT ────────────────────────────────────────────────────
# Directories are auto-created relative to Data/:
#   Data/inputs/<WORKFLOW_NAME>/                — when EVENT is None
#   Data/inputs/<WORKFLOW_NAME>/<EVENT>/        — when EVENT is set (recommended)
#   Data/outputs/<WORKFLOW_NAME>/<EVENT>/       — plots are saved here
#
# Set EVENT = None to keep runs flat. For cross-event comparisons, set
# EVENT = None and prefix filenames with the event subfolder (see the
# "Cross-event comparison" example in RUNS below).
WORKFLOW_NAME = "template"
EVENT = "26R04MIA"

_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)


# ─── RUNS ─────────────────────────────────────────────────────────────────────
# Required keys per run:
#   name    — display label used in plots and legends
#   type    — one of "CAR" | "OC" | "DLS" | "DIL" | "FMIOpt" (selects channel
#             mappings + sign transforms from channel_config.py)
#   file    — path relative to the workflow input folder (single-file mode)
#     OR
#   folder  — subfolder of the workflow input folder (folder mode; every
#             matching file loads as its own run, auto-named + auto-coloured)
#
# Optional keys (any run mode):
#   color        — hex colour override for this run's traces
#   nrun         — (parquet) rank-based selection; nrun=1 → lowest nRun
#   nlap         — exact lap-number filter; ignored when nrun is also set
#   best_n       — pick the N fastest laps (uses nLap + tLap_Calc grouping)
#   reference    — mark exactly one run as the baseline used by show_delta
#                  waveforms, tDiff, and bar/scatter delta annotations.
#                  If no run is flagged, the first loaded run is used.
#   group        — free-form tag consumed by plot_modal_evolution(group_by=…)
#                  and any custom cross-run comparison (e.g. "RED" / "BLUE").
#
# Folder-mode extras (only valid alongside "folder"):
#   filetype     — required, e.g. ".parquet" or ".txt"
#   contains     — case-insensitive substring filter over filenames
#   name_prefix  — prepended to each auto-generated run name
#   colors       — list of hex colours cycled per file
#   color_range  — 2-tuple ("#start", "#end") for a gradient across files
#
# Consolidation (folder mode only) — merge sources into synthetic runs:
#   consolidate       — True (keep sources AND merged) or "only" (drop sources)
#   consolidate_by    — "session" preset, a regex with one capture group, or
#                       callable(path) -> str/None. Splits the folder into
#                       groups; one merged run is emitted per group.
#   consolidated_name — name template for the merged run(s). Use "{group}"
#                       to embed the group key (e.g. "26R04MIA_{group}").
#
# In-memory splitting (any mode) — partition ONE loaded DataFrame into N sub-runs:
#   split_by     — column name to group on, e.g. "nRun"; or
#                  {"column": "nRun", "values": [1, 3, 5]} to keep a subset;
#                  or callable(df) -> Series of group keys.
#   split_by is mutually exclusive with consolidate/consolidate_by.

RUNS = [
    # ── DLS single-file run (baseline for show_delta / tDiff) ────────────────
    {
        "name": "DLS Baseline",
        "type": "DLS",
        "color": "#0083BF",
        "file": r"26R04MIA  PER Q1R3_DLS.parquet",
        "nlap": 1,
        "reference": True,
    },

    # ── CAR .txt single-file run ─────────────────────────────────────────────
    {
        "name": "Test Run 4",
        "type": "CAR",
        "color": "#D70000",
        "file": "26R04MIA_260502_MAC26-01_PER_Q_R03_4.txt",
    },

    # ── OC parquet run (rank-based lap selection) ────────────────────────────
    # {"name": "OC Reference",    "type": "OC",     "color": "#51FF00", "nrun": 1, "file": r"my_oc_file.parquet"},

    # ── DIL parquet run ──────────────────────────────────────────────────────
    # {"name": "DIL Baseline",    "type": "DIL",    "color": "#FF8800", "nlap": 1, "file": r"my_dil_file.parquet"},

    # ── best_n: auto-select the N fastest laps in a file ─────────────────────
    # {"name": "PER FP2 - fastest 3", "type": "CAR", "color": "#00A149",
    #  "file": r"26R04MIA_260502_MAC26-01_PER_FP2.txt", "best_n": 3},

    # ── Folder mode: every matching file becomes its own run ─────────────────
    # {"folder": "2xStopChoc", "filetype": ".parquet", "contains": "FP1",
    #  "type": "DLS", "nlap": 1, "name_prefix": "2x - "},

    # ── Consolidation: merge every file in a folder into one synthetic run ───
    # {"folder": "26R07BCN/RED", "filetype": ".parquet", "type": "DLS",
    #  "consolidate": "only", "consolidated_name": "26R07BCN_RED",
    #  "group": "RED", "nlap": 1},

    # ── Per-session consolidation: one merged run per practice session ───────
    # {"folder": "26R07BCN/RED", "filetype": ".parquet", "type": "DLS",
    #  "consolidate": "only", "consolidate_by": "session",  # → RED_P1, RED_P2 …
    #  "consolidated_name": "26R07BCN_RED_{group}", "group": "RED"},

    # ── In-memory split: one parquet, N sub-runs by column value ─────────────
    # {"name": "Sweep",           "type": "OC",     "file": r"weight_sweep.parquet",
    #  "split_by": "nRun",        # ← every distinct nRun value becomes a sub-run
    #  "colors": ["#0083BF", "#FF8800", "#51FF00"]},

    # ── Cross-event comparison (set EVENT = None above first) ────────────────
    # {"name": "MIA LTS", "type": "DLS", "color": "#0083BF", "nlap": 1,
    #  "file": "26R04MIA/26R04MIA  PER Q1R3_DLS.parquet"},
    # {"name": "SUZ LTS", "type": "DLS", "color": "#D70000", "nlap": 1,
    #  "file": "26R03SUZ/26R03SUZ  77  Quali  Run 3 Q1R3  Stint 1_DLS.parquet"},
]


# ─── POWERPOINT EXPORT (optional) ─────────────────────────────────────────────
# Blank 16:9 deck by default. Set POWERPOINT_OUTPUT = None to disable.
# For a corporate template with cover slides, pass a template path to
# run_workflow() alongside a start_slide index; see the main() call.
POWERPOINT_OUTPUT = None  # e.g. _OUTPUT_DIR / "Report.pptx"


# ─── CALCULATED CHANNELS (optional override) ─────────────────────────────────
# Every workflow already inherits the shared CALCULATED_CHANNELS from
# channel_config.py. Only define overrides here for workflow-specific
# derived channels not in the shared config, e.g.:
#
#   from channel_config import CALCULATED_CHANNELS as _SHARED_CALC
#   from engine import calc_channel
#   CALCULATED_CHANNELS = {
#       **_SHARED_CALC,
#       # Simple one-liner (deps auto-detected from df["…"] literals):
#       "MyRatio": lambda df: df["A"] / df["B"],
#       # Multi-line lambda → declare deps explicitly with @calc_channel:
#       "EngineEff": calc_channel("nEngine", "tThrottle")(
#           lambda df: df["nEngine"] * df["tThrottle"] / 1000.0
#       ),
#   }
# Then pass ``calculated_channels=CALCULATED_CHANNELS`` in the run_workflow() call.


# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────
# One row per subplot; each row is a channel name or a ("left", "right") pair.
# ``axis_limits`` / ``reference_lines`` / ``subplot_heights`` must be the same
# length as ``channels`` (or None to skip that row).

WAVEFORM_PLOT_DEFINITIONS = [
    # ── Basic multi-channel waveform ─────────────────────────────────────────
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((60, 400), (-1, 9)), (-180, 180), None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.4, 0.8, 0.4, 0.4, 0.4),
    ),

    # ── Zoom to a section of the lap via x_limits ────────────────────────────
    WaveformPlot(
        name="Zoomed Section",
        channels=('vCar', 'pBrakeF', 'rThrottle'),
        axis_limits=(None, None, (0, 105)),
        subplot_heights=(0.6, 0.4, 0.4),
        x_limits=(1200, 1800),
    ),

    # ── Time-based x-axis + shaded highlight_zones ───────────────────────────
    WaveformPlot(
        name="Highlight Zones Demo",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, None, (20,)),
        subplot_heights=(0.6, 0.4, 0.4),
        x_channel="tLap",  # sLap (default) or tLap
        highlight_zones=('rThrottle', '<', 20, '#FF4444'),
    ),

    # ── Normalised overlay: every channel scaled to peak-abs 1 ───────────────
    WaveformPlot(
        name="Normalised Overlay",
        channels=('PMGUK', 'vCar', 'pBrakeF', 'rThrottle'),
        subplot_heights=(0.4, 0.4, 0.4, 0.4),
        normalise=True,
    ),

    # ── Markers: static (x=…) + condition-triggered (rising/falling/both) ────
    # ``row`` scopes a marker to one subplot index (0-based).
    # ``max_count`` caps how many condition markers fire per run.
    WaveformPlot(
        name="Waveform Markers Demo",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        subplot_heights=(0.6, 0.4, 0.4),
        markers=[
            Marker(x=1500, label="SM zone", color="#00B050", linestyle="--"),
            Marker(x=2200, label="apex", row=0),                     # only on row 0
            Marker(
                condition=[('pBrakeF', '>', 50), ('vCar', '>', 100)],
                edge="rising",       # "rising" | "falling" | "both"
                max_count=5,         # limit how many events are drawn per run
                label="hard brake",
                linestyle="-.",
                show_label=False,
            ),
        ],
    ),

    # ── show_delta: appends a slim delta-vs-reference row under each toggled
    # primary row. Requires 2+ runs; the reference is chosen workflow-wide via
    # ``"reference": True`` (else the first loaded run). ``show_delta`` accepts
    # True/False (all rows) or a per-row tuple.
    WaveformPlot(
        name="Delta Comparison",
        channels=('vCar', 'pBrakeF', 'rThrottle'),
        axis_limits=(None, None, (0, 105)),
        subplot_heights=(0.6, 0.4, 0.4),
        show_delta=(True, False, True),
    ),

    # ── tDiff — auto-computed lap-time delta channel (available with 2+ runs) ─
    WaveformPlot(
        name="Lap Time Delta",
        channels=('vCar', 'tDiff'),
        axis_limits=(None, (-5, 5)),
        reference_lines=(None, 0),
        subplot_heights=(0.6, 0.4),
    ),

    # ── legend_position: "top" (default) or "right" ──────────────────────────
    WaveformPlot(
        name="Right Legend Demo",
        channels=('vCar', 'pBrakeF'),
        subplot_heights=(0.6, 0.4),
        legend_position="right",
    ),

    # ── annotate_at: read off values at x-positions (dot + label per run) ────
    WaveformPlot(
        name="Annotate At Demo",
        channels=('vCar', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((0, 105), (0, 1.3))),
        subplot_heights=(0.6, 0.4, 0.4),
        annotate_at=(500, 1000, 2000),
    ),
]


# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────
# best_fit:
#   0 / None  — no fit
#   1         — single linear regression
#   2         — quadratic
#   list      — piecewise segments; each entry is (axis, lo, hi) where axis
#               is "x", "y", or any data column (e.g. "NGear", "SM").
# gate:  ("channel", "op", value) or a list AND-ed together.
# Valid ops: > < >= <= == != between outside robust
#   - "between" / "outside" take a (lo, hi) tuple as the value
#   - "robust" takes a k-sigma float (MAD-based outlier rejection)

SCATTER_PLOT_DEFINITIONS = [
    # ── Basic scatter (no fit) ───────────────────────────────────────────────
    ScatterPlot("GG Plot", "gLat", "gLong", best_fit=0),

    # ── Single linear fit + equation + %-error vs first run ──────────────────
    ScatterPlot("Engine Efficiency", "dmInjector", "PEngine",
                best_fit=1, show_equations=True, show_error=True),

    # ── Segmented fits by category channel (per-gear ratios) ─────────────────
    ScatterPlot("Gear Ratios", "nWheelAvg_R", "nEngine",
                best_fit=[('NGear', g - 0.5, g + 0.5) for g in range(2, 9)],
                show_equations=False),

    # ── Segmented fits by axis value ─────────────────────────────────────────
    ScatterPlot("Front Heave", "xDamperAvgF", "FPRodAvgF",
                best_fit=[('y', -6000, None), ('y', None, -6000)],
                error_as_factor=True),  # show "×1.05" instead of "+5.0%"

    # ── Gate: simple + multi-condition (AND) ─────────────────────────────────
    ScatterPlot("Braking Efficiency", "pBrakeF", "gLong",
                best_fit=[('y', None, -0.2)],
                gate=('gLong', '<', 0)),
    ScatterPlot("Front Pushrod vCar", "vCar", "FPRodAvgF",
                best_fit=[('gLat_Abs', 0, 1)],
                gate=[('SM', '<', 1), ('pBrakeF', '<', 1)]),

    # ── "between" / "outside" / "robust" gate operators ──────────────────────
    ScatterPlot("Mid Speed Rear Ride", "vCar", "hRideR",
                gate=[('vCar', 'between', (120, 240)),
                      ('gLong', 'outside', (-0.1, 0.1))]),
    ScatterPlot("Robust Filter Demo", "vCar", "PEngine",
                gate=('PEngine', 'robust', 3.0)),  # 3-MAD outlier reject

    # ── Custom axis_limits + reference_lines (horizontal y-guides) ───────────
    ScatterPlot("Yaw Rate Response", "aSteerWheel", "nYaw",
                axis_limits=[(-160, 160), (None, None)],
                best_fit=[('x', -20, 20)],
                reference_lines=[0]),

    # ── color_gate: highlight a matching subset in a different colour ────────
    ScatterPlot("Color Gate Demo", "vCar", "hRideF",
                best_fit=[('SM', 0, 0.5)],
                color_gate=('SM', '<', 0.3, '#FF00CC')),

    # ── annotate_fit_at: value read-outs at each x-position, per segment ─────
    ScatterPlot("Annotate Fit At Demo", "vCar", "hRideR",
                best_fit=1, gate=[('SM', '<', 0.5)],
                annotate_fit_at=(100, 200, 300)),

    # ── Robust fit: Theil–Sen slope + MAD outlier flag ───────────────────────
    ScatterPlot("Robust Fit Demo", "vCar", "PEngine",
                best_fit=1, robust=True, robust_threshold=3.0),

    # ── Vertical markers on a scatter ────────────────────────────────────────
    ScatterPlot("Scatter Markers Demo", "vCar", "gLong",
                markers=[
                    Marker(x=100, label="100 km/h"),
                    Marker(x=300, label="300 km/h", color="#FF6600"),
                ]),

    # ── Quadratic fit ────────────────────────────────────────────────────────
    ScatterPlot("Quadratic Fit Demo", "vCar", "hRideF",
                best_fit=2, show_equations=True, gate=[('SM', '<', 0.5)]),

    # ── dil_friendly: SM-split info-box relabels to "SM OFF" / "SM ON"
    # and the per-segment delta becomes an absolute y-intercept delta
    # (Δc = ±X.XX) instead of a slope factor. Used in DIL correlation reports.
    ScatterPlot("DIL Friendly Demo", "vCar", "hRideF",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)],
                annotate_fit_at=(100, 200, 300),
                dil_friendly=True),
]


# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
# nperseg: int, "auto", or None. None/"auto" uses the sample-rate-aware policy
#          in engine.datafunctions.auto_nperseg.
# lorentz_fit: (f_lo, f_hi) or a list of such tuples. Each window fits a
#              single-DOF Lorentzian + baseline and annotates f₀ and ζ (with
#              uncertainty when a bootstrap CI is enabled elsewhere).

PSD_PLOT_DEFINITIONS = [
    # ── Single channel ───────────────────────────────────────────────────────
    PsdPlot("Front Vertical Acceleration PSD", "gVertF",
            axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),

    # ── Multi-channel overlay (line style cycles per channel) ────────────────
    PsdPlot("Multi-Channel PSD Demo", ["FPRodAvgF", "FPRodAvgR"],
            axis_limits=[(0, 20), (1e-4, None)], annotate_at=(5, 15)),

    # ── Gated PSD: only spectrogram segments where the gate is True ──────────
    PsdPlot("Gated PSD Demo", "gVertF",
            axis_limits=[(0, 30), (1e-4, None)], nperseg=256,
            gate=[('vCar', '>', 100), ('SM', '<', 0.5)]),

    # ── show_envelope: ±1σ shading across runs ───────────────────────────────
    PsdPlot("PSD Envelope Demo", "FPRodAvgF",
            axis_limits=[(0, 20), (1e-4, None)],
            show_envelope=True, nperseg=512),

    # ── Static markers + horizontal reference_lines ──────────────────────────
    PsdPlot("PSD Markers Demo", "gVertF",
            axis_limits=[(0, 30), (1e-4, None)],
            reference_lines=[1e-2],
            markers=[
                Marker(x=5,  label="5 Hz"),
                Marker(x=15, label="15 Hz", color="#FF6600"),
            ]),

    # ── Linear-y PSD (log_scale=False) with automatic Welch window ───────────
    PsdPlot("Linear PSD Demo", "FPRodHeave",
            axis_limits=[(0, 20), (0, None)],
            log_scale=False, nperseg="auto"),

    # ── Lorentzian fit: one window per peak. f₀ is seeded at the in-window
    # argmax and free to roam across the full window. Damping ratio (ζ) is
    # annotated on each fitted curve.
    PsdPlot("Lorentz Fit Demo", "gVertF",
            axis_limits=[(0, 30), (1e-4, None)],
            lorentz_fit=[(4, 7), (13, 17)]),
]


# ─── HISTOGRAM PLOTS ──────────────────────────────────────────────────────────

HISTOGRAM_PLOT_DEFINITIONS = [
    HistogramPlot("Plank Power Distribution", "PPlank_F",
                  axis_limits=[(1, 51), (None, None)]),

    # ── Log-scale y-axis (long-tail distributions) ───────────────────────────
    HistogramPlot("Log-Scale Histogram Demo", "PPlank_F",
                  axis_limits=[(0, 100), (None, None)], log_scale=True),

    # ── Gated + reference_lines + markers ────────────────────────────────────
    HistogramPlot("Gated Histogram Demo", "vCar",
                  axis_limits=[(50, 350), (None, None)],
                  gate=[('SM', '<', 0.5), ('pBrakeF', '<', 1)],
                  reference_lines=[500],
                  markers=[
                      Marker(x=100, label="Low speed"),
                      Marker(x=250, label="High speed", color="#D70000"),
                  ]),
]


# ─── BAR PLOTS ────────────────────────────────────────────────────────────────
# metrics: tuple of "channel" or ("channel", "aggregation") entries.
# Valid aggregations:
#   integral abs_integral sum abs_sum mean median max min first last
# secondary_axis (default True): auto-splits very-different-scale metrics
#   onto a second y-axis; disable with secondary_axis=False.
# show_delta: append "Δ ±X.XX" on non-baseline bars vs the reference run.
# error_metrics: tuple parallel to metrics; each entry is a sigma-channel
#   name (or None) that becomes the errorbar half-width for that bar.

BAR_PLOT_DEFINITIONS = [
    # ── Multiple metrics, all with the same aggregation ──────────────────────
    BarPlot(
        name="Cumulative Metrics",
        metrics=(("dmInjector (kg/s)", "integral"),
                 ("PMGUK_Deploy (MJ)", "integral"),
                 ("PMGUK_Charge (MJ)", "integral")),
    ),

    # ── Single metric, one aggregation, with reference_lines (warn/fail) ────
    BarPlot(
        name="Plank Energy Target Demo",
        metrics=(("EPlank_F", "max"),),
        reference_lines=[300.0, 500.0],
    ),

    # ── default_aggregation + axis_limits + explicit secondary_axis ──────────
    BarPlot(
        name="Bar Axis Limits Demo",
        metrics=(("vCar",), ("nEngine",)),
        default_aggregation="mean",
        axis_limits=(0, 400),
        secondary_axis=False,
    ),

    # ── show_delta: baseline comparison (Δ vs reference run on each bar) ─────
    BarPlot(
        name="Lap Time Compare",
        metrics=(("tLap_Calc", "max"),),
        show_delta=True,
    ),

    # ── error_metrics: sigma channels drive per-bar errorbars ────────────────
    # Length must match metrics; None skips errorbars for that metric.
    BarPlot(
        name="Modal Frequencies",
        metrics=(("modal_heave_f0", "mean"),
                 ("modal_pitch_f0", "mean"),
                 ("modal_roll_f0",  "mean"),
                 ("modal_warp_f0",  "mean")),
        error_metrics=("modal_heave_f0_sigma",
                       "modal_pitch_f0_sigma",
                       "modal_roll_f0_sigma",
                       "modal_warp_f0_sigma"),
    ),

    # ── gate: pre-filter every run before aggregation ────────────────────────
    BarPlot(
        name="High Speed Ride Height Mean",
        metrics=(("hRideF", "mean"), ("hRideR", "mean")),
        gate=[('vCar', '>', 200), ('SM', '<', 0.5)],
    ),
]


# ─── BOX PLOTS ────────────────────────────────────────────────────────────────
# aggregation_mode:
#   per_run             — one box per (channel, run)
#   aggregated          — one box per channel with every run pooled
#   per_run_aggregated  — per-run boxes plus a pooled "ALL" box

BOX_PLOT_DEFINITIONS = [
    # ── Single channel, per-run boxes, gated ─────────────────────────────────
    BoxPlot(
        name="Low Speed Corner Distribution",
        channels="vCar",
        aggregation_mode="per_run",
        gate=[("gLong", "between", (-0.1, 0.1)), ("vCar", "<", 120)],
    ),

    # ── Multi-channel, per-run + pooled overlay ──────────────────────────────
    BoxPlot(
        name="Combined Ride Height Distribution",
        channels=("hRideF", "hRideR"),
        aggregation_mode="per_run_aggregated",
    ),

    # ── axis_limits + reference_lines + per-plot style overrides ─────────────
    # ``options`` overrides the shared BOX_PLOT_SETTINGS from channel_config.py.
    BoxPlot(
        name="Box Axis Limits Demo",
        channels="hRideF",
        aggregation_mode="per_run",
        axis_limits=(20, 80),
        reference_lines=[35, 55],
        options={"show_points": False, "box_edge_color": "#000000"},
    ),

    # ── BoxPlotGrid: rows × cols of AND-gated boxes ──────────────────────────
    # render_mode="grid"    — a single figure with a subplot matrix
    # render_mode="expand"  — one BoxPlot figure per cell (default)
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
        render_mode="grid",
    ),
]


# ─── HEATMAP PLOTS ────────────────────────────────────────────────────────────
# z_channel=None    → point-density (2-D histogram)
# z_channel="chan"  → per-bin aggregation of that channel
# Valid aggregations: mean median std count sum max min

HEATMAP_PLOT_DEFINITIONS = [
    # ── Density heatmap (no z_channel) ───────────────────────────────────────
    HeatmapPlot("gLat vs gLong Density", "gLat", "gLong", bins=100),

    # ── Aggregation heatmap (mean SM per bin) ────────────────────────────────
    HeatmapPlot("Ride Height vs Speed (mean SM)", "vCar", "hRideF",
                z_channel="SM", aggregation="mean", bins=100),

    # ── Custom cmap + z_limits + non-square bins ─────────────────────────────
    HeatmapPlot("Custom Cmap Demo", "vCar", "gLat",
                z_channel="hRideF", aggregation="median",
                bins=(80, 60), cmap="plasma", z_limits=(25, 70)),

    # ── Gated heatmap with min_count and markers ─────────────────────────────
    HeatmapPlot("Gated Heatmap Demo", "vCar", "hRideR",
                z_channel="gLong", aggregation="mean", bins=60,
                gate=[('SM', '<', 0.5)], min_count=5,
                axis_limits=[(50, 350), (20, 80)],
                markers=[Marker(x=200, label="200 km/h")]),
]


# ─── 3D SCATTER PLOTS ─────────────────────────────────────────────────────────
# Three channels against each other, one colour per run. Always writes a static
# PNG; also writes an interactive .html when plotly is installed.

SCATTER3D_PLOT_DEFINITIONS = [
    Scatter3DPlot("gLat gLong vCar", "gLat", "gLong", "vCar"),

    # ── Gated + explicit limits (exactly three (lo, hi) pairs) ───────────────
    # Scatter3DPlot("Ride Map", "vCar", "hRideF", "hRideR",
    #               gate=('SM', '<', 0.5),
    #               axis_limits=[(50, 350), (20, 80), (20, 120)]),
]


# ─── POWERPOINT EXPORT MAP (optional) ─────────────────────────────────────────
# Maps slides to generated plot images via the Slide() helper.
# Layouts: "main_plot" (one full-width image) | "double_plot" (two side by side)
# References use "type/Plot Name" — auto-converted to the on-disk filename.

POWERPOINT_EXPORT_MAP = [
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("double_plot", "scatter/GG Plot",         "scatter/Engine Efficiency"),
    Slide("double_plot", "scatter/Gear Ratios",     "scatter/Front Heave"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Multi-Channel PSD Demo"),
]

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    run_workflow(
        WORKFLOW_NAME,
        title=f"{WORKFLOW_NAME.upper()} PLOT GENERATION",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        psds=PSD_PLOT_DEFINITIONS,
        histograms=HISTOGRAM_PLOT_DEFINITIONS,
        bars=BAR_PLOT_DEFINITIONS,
        boxes=BOX_PLOT_DEFINITIONS,
        heatmaps=HEATMAP_PLOT_DEFINITIONS,
        scatter3d=SCATTER3D_PLOT_DEFINITIONS,
        powerpoint_output=POWERPOINT_OUTPUT,
        export_map=POWERPOINT_EXPORT_MAP,
        # ── Optional overrides (uncomment as needed) ───────────────────────────
        # verbose=True,               # debug-level logging
        # output_dpi=150,             # lower DPI for quick iteration (default 300)
        # scatter_max_points=30000,   # decimate scatter above this count
        # resample_rate=50,           # override channel_config.RESAMPLE_RATE (Hz)
        # open_output=False,          # don't auto-open the output folder
        # fig_size={"waveform": (14, 10), "scatter": (10, 8)},  # per-kind size
        # calculated_channels=CALCULATED_CHANNELS,  # workflow-specific overrides
        # filters=MY_FILTERS,         # override channel_config.FILTERS
        # cli_description="My custom --help text",  # override argparse text
        # ── Corporate template (cover slides + 16:9 body) ─────────────────────
        # from channel_config import resolve_template_path
        # powerpoint_template=resolve_template_path("template.pptx"),
        # powerpoint_start_slide=4,   # first slide to place a plot on
        # ── Multi-deck export (list of (template, output, map[, start_slide])) ─
        # powerpoint_exports=[
        #     (None,                          _OUTPUT_DIR / "Report.pptx",       POWERPOINT_EXPORT_MAP),
        #     (resolve_template_path("template.pptx"), _OUTPUT_DIR / "Modal.pptx", MODAL_EXPORT_MAP, 4),
        # ],
        # ── Modal fitting (per-run 4-DOF fit; injects modal_* channels) ───────
        # vibrations_fit={
        #     "method":         "lorentzian_combined",    # or "body4dof"
        #     "fmin":           2.0,
        #     "fmax":           13.0,
        #     "nperseg":        "auto",
        #     "expected_freqs": {"heave": (3, 7), "pitch": (7, 11),
        #                       "roll":  (4, 8), "warp":  (9, 13)},
        #     "bootstrap_ci":   True,
        #     "bootstrap_n":    400,
        # },
    )

    # ── Modal-parameter evolution across sessions/runs (call after workflow) ─
    # Requires vibrations_fit=… above so plotter.modal_results is populated.
    #
    # from engine import plot_modal_evolution
    # plotter = run_workflow(...)   # capture the returned DataPlotter
    # if plotter is not None and getattr(plotter, "modal_results", None):
    #     for mode in ("Heave", "Pitch", "Roll", "Warp"):
    #         plot_modal_evolution(
    #             plotter,
    #             modes=(mode,),
    #             group_by="group",        # run dict key, e.g. "RED"/"BLUE"
    #             compare_by="session",    # aligns runs by session token
    #             name_suffix=mode.lower(),
    #         )


if __name__ == "__main__":
    main()
