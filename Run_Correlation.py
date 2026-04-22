"""Correlation report entry point.

This file stays intentionally configuration-heavy, but only for the settings
that are specific to the correlation workflow.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from data_layout import (
    CORRELATION_OUTPUT_DIR,
    resolve_correlation_input_dir,
    resolve_template_path,
)
from plot_runtime import build_plot_groups, build_plotter as runtime_build_plotter, run_plot_job
from plot_shared_config import CHANNEL_MAPPINGS, UNITS_MAP

# ================================================================================
# CONFIGURATION: DATA FILES & LOCATIONS
# ================================================================================
# Root folder for input data and output plots

ROOT_FOLDER = resolve_correlation_input_dir()


# ================================================================================
# CONFIGURATION: RUN DEFINITIONS
# ================================================================================
# Load any number of telemetry runs. Each run specifies a data file, display name,
# color, and optional filtering criteria.
#
# Format:
#   {"name": "<run_id>", "file": "<relative_path_from_Data>", "color": "<#RRGGBB>", 
#    "nrun": <optional_ranked_run_index>, "nlap": <optional_lap_number>, "type": "<DATA_TYPE>"}
#
# Parameters:
#   - name:    Display name for this run in plots and reports
#   - file:    Filename relative to ROOT_FOLDER
#   - color:   Hex color code (#RRGGBB) for plotting this run
#   - nrun:    (Optional, parquet only) Rank-based run selection
#              nrun=1 -> lowest nRun value in file, nrun=2 -> next lowest, etc.
#   - nlap:    (Optional) Exact lap filter. Use nrun if both specified.
#   - type:    Data source type (OC, CAR, DLS, DIL) - determines mapping rules
#
# Notes:
#   - Set type correctly; it determines which shared mappings and
#     CHANNEL_TRANSFORMS are applied
#   - Use nrun for ranked selection when multiple runs exist in one file
#   - nlap filters on exact nLap value when available

RUNS = [
    # Example configurations (uncomment to activate):
    # {"name": "v37", "file": r"20260408-OC-VPG - Ref Data Set - CF1-26v037 - v1-BCN.parquet", 
    #  "color": "#0051FF", "nrun": 1, "type": "OC"},
    # {"name": "v38a", "file": r"20260408-OC-VPG - Ref Data Set - CF1-26v038a - v1-BCN.parquet", 
    #  "color": "#FF0000", "nrun": 1, "type": "OC"},
    
    # Active runs - modify these as needed:
    #{
    #    "name": "car",
    #    "file": "26R03SUZ_260328_MAC26-01_PER_Q_R02.txt",
    #    "color": "#FF8C00",
    #    "type": "CAR"
    #},
    {
        "name": "BSL",
        "file": r"26R03SUZ  77  Quali  Run 3 Q1R3 _V1_+2FRH STD SMv2_DLS.parquet",
        "color": "#FF3300",
        "nlap": 1,
        "type": "DLS"
    },
    {
        "name": "25m Less SM",
        "file": r"26R03SUZ  77  Quali  Run 3 Q1R3 _V1_+2FRH -25 SMv2_DLS.parquet",
        "color": "#00CCFF",
        "nlap": 1,
        "type": "DLS"
    },
        {
        "name": "50m Less SM",
        "file": r"26R03SUZ  77  Quali  Run 3 Q1R3 _V1_+2FRH -50 SMv2_DLS.parquet",
        "color": "#009FCB",
        "nlap": 1,
        "type": "DLS"
    },
    {
        "name": "100m Less SM",
        "file": r"26R03SUZ  77  Quali  Run 3 Q1R3 _V1_+2FRH -100m SMv2_DLS.parquet",
        "color": "#7300FF",
        "nlap": 1,
        "type": "DLS"
    },
    {
        "name": "100m Less SM - Corrected",
        "file": r"26R03SUZ  77  Quali  Run 3 Q1R3 _V1_+2p5FRH -100 SMv2_DLS.parquet",
        "color": "#3ADA00",
        "nlap": 1,
        "type": "DLS"
    }
]

# ================================================================================
# CONFIGURATION: POWERPOINT OUTPUT
# ================================================================================
# Template and output paths for PowerPoint report generation.
# Template must be a valid .pptx file with placeholders for images.

POWERPOINT_TEMPLATE = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT = CORRELATION_OUTPUT_DIR / "Correlation_Report.pptx"
EXPORT_TO_POWERPOINT = True


# ================================================================================
# CONFIGURATION: CHANNEL TRANSFORMS
# ================================================================================
# Apply mathematical transformations to channels (unit conversions, sign corrections).
# Each function receives raw channel values and returns transformed values.
#
# Common transformations:
#   - Unit conversion (W → kW): lambda x: x / 1000
#   - Sign correction: lambda x: -x
#   - Offset adjustment: lambda x: x - offset_value
#   - Conditional: lambda x: x * condition_array

CHANNEL_TRANSFORMS = {
    'OC': None,
    
    'DLS': {
        # DLS simulator load sign corrections and acceleration offsets
        "FPRodFL": lambda x: -x,
        "FPRodFR": lambda x: -x,
        "FPRodRL": lambda x: -x,
        "FPRodRR": lambda x: -x,
        "aRoll": lambda x: -x,
        "gVert": lambda x: x - 1,
    },
    
    'DIL': None,
    
    "CAR": {
        # Vehicle ECU unit conversions and adjustments
        "PMGUK": lambda x: x / 1000,              # W → kW
        "sLap": lambda x: x - 15,                 # GPS alignment shift
    }
}


# ================================================================================
# CONFIGURATION: CALCULATED CHANNELS
# ================================================================================
# Define derived channels computed from raw channel data.
# Each entry is a lambda function receiving the dataframe and returning computed values.
# These derived channels can be used in plots just like raw channels. 
# Can duplicated channels if different levels of filtering or processing are needed (e.g. raw vs filtered ride height).

CALCULATED_CHANNELS = {
    # Pushrod load differentials and averages
    "FPRodDeltaF": lambda df: df["FPRodFL"] - df["FPRodFR"],
    "FPRodDeltaR": lambda df: df["FPRodRL"] - df["FPRodRR"],
    "FPRodAvgF": lambda df: (df["FPRodFL"] + df["FPRodFR"]) / 2,
    "FPRodAvgR": lambda df: (df["FPRodRL"] + df["FPRodRR"]) / 2,
    
    # Damper travel differentials and averages
    "xDamperDeltaF": lambda df: df["xDamperFL"] - df["xDamperFR"],
    "xDamperDeltaR": lambda df: df["xDamperRL"] - df["xDamperRR"],
    "xDamperAvgF": lambda df: (df["xDamperFL"] + df["xDamperFR"]) / 2,
    "xDamperAvgR": lambda df: (df["xDamperRL"] + df["xDamperRR"]) / 2,
    
    # Lateral acceleration derived channels
    "gLat_Abs": lambda df: df["gLat"].abs(),
    "gLong (raw)": lambda df: df["gLong"],
    
    # Ride height channels
    "hRideF (raw)": lambda df: df["hRideF"],
    "hRideR (raw)": lambda df: df["hRideR"],
    
    # Combined power channels
    "PPUTotal": lambda df: df["PMGUK"] + df["PEngine"],
    
    # Wheel speed (rear average)
    "nWheelAvg_R": lambda df: (df["nWheelRL"] + df["nWheelRR"]) / 2,
    
    # Fuel/injector consumption (unit conversion)
    "dmInjector (kg/s)": lambda df: df["dmInjector"] / 3600,  # Convert kg/h to kg/s
    
    # MGUK deployment modes
    "PMGUK_Deploy": lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] > 0).astype(float)).abs(),
    "PMGUK_Charge": lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] < 0).astype(float)).abs(),
    
    # Plank energy (integrated power)
    "PPlank_F": lambda df: 0.001 * np.maximum(0.1 * df["FzPlankF"] * (df["vCar"] / 3.6), 0),
    "EPlank_F": lambda df: cumulative_trapezoid(df["PPlank_F"], dx=0.01, initial=0),
}


# ================================================================================
# CONFIGURATION: LOW-PASS FILTER SETTINGS
# ================================================================================
# Apply low-pass filtering to smooth noisy channels.
# cutoff=0 disables filtering for that channel.
#
# Parameters:
#   - cutoff: Cutoff frequency in Hz (0 = no filter)
#   - order:  Filter order (higher = steeper rolloff, more lag)
#   - "all":  Fallback filter applied to any channel not explicitly listed. 
#             Be careful with this as it may unintentionally filter channels that should be left raw.

LOW_PASS_FILTERS = {
    # Acceleration channels
    "gVertF": {"cutoff": 0, "order": 2},
    "gVertR": {"cutoff": 0, "order": 2},
    "gVert": {"cutoff": 0, "order": 2},
    
    # Plank and power channels
    "FzPlankF": {"cutoff": 0, "order": 2},
    "PMGUK": {"cutoff": 0, "order": 2},
    
    # Discrete channels (no filtering needed)
    "SM": {"cutoff": 0, "order": 2},
    "NGear": {"cutoff": 0, "order": 2},
    
    # Engine and drivetrain
    "nEngine": {"cutoff": 0, "order": 2},
    "nWheelAvg_R": {"cutoff": 0, "order": 2},
    
    # Plank wear
    "EPlank_F": {"cutoff": 0, "order": 2},
    "PPlank_F": {"cutoff": 0, "order": 2},
    
    # Driver inputs and vehicle state
    "rThrottle": {"cutoff": 0, "order": 2},
    "gLong (raw)": {"cutoff": 0, "order": 2},
    "hRideF (raw)": {"cutoff": 0, "order": 2},
    "hRideR (raw)": {"cutoff": 0, "order": 2},
    "dmInjector": {"cutoff": 0, "order": 2},
    "PPUTotal": {"cutoff": 0, "order": 2},
    "vCar": {"cutoff": 0, "order": 2},
    
    # Default filter for any unlisted channels
    "all": {"cutoff": 5, "order": 2},
}


# ================================================================================
# CONFIGURATION: SCATTER PLOT RENDERING
# ================================================================================
# Control how scatter plots are rendered. For large datasets, switches to
# hexagonal binning to avoid overplotting and improve performance.
#
# Rendering modes:
#   - "auto":     Automatically switches to hexbin if point count exceeds threshold
#   - "scatter":  Always use scatter plot (slower for large datasets)
#   - "hexbin":   Always use hexagonal binning (faster, shows density)

SCATTER_RENDER_MODE = "auto"                    # "auto", "scatter", "hexbin"
SCATTER_DENSITY_THRESHOLD = 25000                # Switch to hexbin above this point count
SCATTER_MAX_POINTS = 45000                       # Maximum points to render (down-samples if exceeded)
SCATTER_HEXBIN_GRIDSIZE = 70                     # Hexbin resolution (higher = more bins)
BAR_SECONDARY_AXIS_RATIO = 20.0                  # Secondary axis scale factor for bar plots

# ================================================================================
# CONFIGURATION: PLOT DEFINITIONS
# ================================================================================
# Define all plots to generate. Supported plot types:
#   - Waveform:  Time-series multi-channel plots with overlays
#   - Scatter:   XY correlation plots with optional trend lines
#   - PSD:       Power spectral density (frequency domain analysis)
#   - Histogram: Distribution plots
#   - Bar:       Bar charts with aggregations


# ================================================================================
# WAVEFORM PLOT DEFINITIONS
# ================================================================================
# Multi-panel time-series plots with no limit on amount of lots, but recommended maximum of 7 rows per plot.
#
# Format:
#   ["Plot Name", (row_specs...), (axis_limits...), (reference_lines...), (height_ratios...)]
#
# row_specs (one per row):
#   "channel"                       → single Y-axis
#   ("left_channel", "right_channel") → dual Y-axis overlay
#
# axis_limits (tuple matching rows):
#   Single row   → (ymin, ymax) or None
#   Dual row     → ((left_min, left_max), (right_min, right_max)) or None
#
# reference_lines (tuple matching rows):
#   Scalar or tuple of values to plot as horizontal lines, or None
#
# height_ratios (optional):
#   Tuple of relative heights for each row. Default: equal heights.
#
# Example:
#   ["Example", ("Ch1", ("Ch2", "Ch3")), (None, ((0, 10), (100, 200))), (None, (5, 10)), (0.4, 0.8)]

WAVEFORM_PLOT_DEFINITIONS = [
    [
        "Driver Input",
        ('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        (None, ((60, 400), (-1, 9)), (-160, 160), None, ((0, 105), (0, 1.3))),
        ((-350, 0, 350), None, (0), None, None),
        (0.4, 0.8, 0.4, 0.4, 0.4)
    ],
    
    [
        "Power Unit",
        ('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        (None, None, ((60, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        ((-350, 0, 350), (0), None, (10000), None, None),
        (0.4, 0.4, 0.6, 0.4, 0.4, 0.4)
    ],
    
    [
        "Plank Wear",
        ('PMGUK', 'vCar', 'FzPlankF', 'EPlank_F', 'pBrakeF', ('rThrottle', 'SM')),
        (None, None, None, None, None, ((0, 105), (0, 1.3))),
        ((-350, 0, 350), None, (0, 7500), (0), None, None),
        (0.4, 0.6, 0.4, 0.6, 0.4, 0.4)
    ],
    
    [
        "DIL TELEM",
        ('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
        ((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-160, 160), ((None, None), (None, None))),
        (None, None, (-350, 0, 350), None, (0), (0), None),
        (0.15, 0.2, 0.3, 0.5, 0.3, 0.3)
    ]
]


# ================================================================================
# SCATTER PLOT DEFINITIONS
# ================================================================================
# XY correlation plots with optional trend lines, gates, and annotations.
#
# Format:
#   ["Name", (x_channel, y_channel), [(xmin, xmax), (ymin, ymax)], best_fit, gate_spec, show_equations, show_error]
#
# best_fit (trend line mode):
#   0 or None        → no trend line
#   1                → single fit line across all points
#   List             → segmented fits by condition
#     Example: [('x', 0, 10), ('x', 10, 20)]  → separate lines for x: 0-10, 10-20
#     Example: [('SM', 0, 0.5), ('SM', 0.5, 1.0)]  → fits by SM channel (plotted but not shown separately)
#
# gate_spec (optional data filter):
#   Single gate:   ('channel', 'operator', value)
#   Multiple:      [('ch1', '>', val1), ('ch2', '<', val2)]  # ALL conditions must match
#   Operators:     '>', '<', '>=', '<=', '==', 'between'
#   Example: ('vCar', '>', 120)  or  ('gLong', 'between', (-0.5, 0))
#
# show_equations (optional, default True):
#   Display trend line equation on plot
#
# show_error (optional, default True):
#   Display percentage error information box

SCATTER_PLOT_DEFINITIONS = [
    # Drivetrain
    [
        "Gear Ratios",
        ('nWheelAvg_R', 'nEngine'),
        None,
        [('NGear', 1.5, 2.5), ('NGear', 2.5, 3.5), ('NGear', 3.5, 4.5), ('NGear', 4.5, 5.5), ('NGear', 5.5, 6.5), ('NGear', 6.5, 7.5), ('NGear', 7.5, 8.5)],
        False,
        True
    ],
    [
        "Engine Power",
        ('nEngine', 'PEngine'),
        None,
        0,
        True,
        True
    ],
    [
        "Long Acceleration",
        ('vCar', 'gLong'),
        None,
        None,
        True,
        True
    ],
    [
        "Lat Acceleration",
        ('vCar', 'gLat_Abs'),
        None,
        0,
        True,
        True
    ],
    [
        "GG Plot",
        ('gLat', 'gLong'),
        None,
        0,
        True,
        True
    ],
    [
        "Braking Efficiency",
        ('pBrakeF', 'gLong'),
        None,
        [('y', None, -0.2)],
        ('gLong', '<', 0),
        True,
        True
    ],
    [
        "Understeer Plot",
        ('vCar', 'aUndersteerFromSlip'),
        None,
        None,
        ("rThrottle", '<', 95),
        True,
        True
    ],
    [
        "Yaw Rate Response",
        ('aSteerWheel', 'nYaw'),
        [(-160, 160), (None, None)],
        [('x', -20, 20)],
        True,
        True
    ],
    [
        "Lateral Acceleration Response",
        ('aSteerWheel', 'gLat'),
        [(-160, 160), (None, None)],
        [('x', -20, 20)],
        True,
        True
    ],
    [
        "Steering Moment",
        ('aSteerWheel', 'MSteerWheel'),
        [(-160, 160), (None, None)],
        0,
        True,
        True
    ],
    [
        "Damper gLat front",
        ('gLat', 'xDamperDeltaF'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Damper gLat rear",
        ('gLat', 'xDamperDeltaR'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Pushrod gLat front",
        ('gLat', 'FPRodDeltaF'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Pushrod gLat rear",
        ('gLat', 'FPRodDeltaR'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Front Heave",
        ('xDamperAvgF', 'FPRodAvgF'),
        None,
        [('y', None, 10000), ('y', 10000, None)],
        True,
        True
    ],
    [
        "Front Roll",
        ('xDamperDeltaF', 'FPRodDeltaF'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Rear Heave",
        ('xDamperAvgR', 'FPRodAvgR'),
        None,
        [('y', None, -20000), ('y', -20000, None)],
        True,
        True
    ],
    [
        "Rear Roll",
        ('xDamperDeltaR', 'FPRodDeltaR'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Roll angle gLat",
        ('gLat', 'aRoll'),
        None,
        [('x', None, None)],
        True,
        True
    ],
    [
        "Front Pushrod vCar",
        ('vCar', 'FPRodAvgF'),
        None,
        [('gLat_Abs', 0, 1)],
        [('SM', '<', 1)],
        True,
        True
    ],
    [
        "Rear Pushrod vCar",
        ('vCar', 'FPRodAvgR'),
        None,
        [('gLat_Abs', 0, 1)],
        [('SM', '<', 1)],
        True,
        True
    ],
    [
        "Front Ride vCar",
        ('vCar', 'hRideF'),
        None,
        [('SM', 0, 0.5)],
        True,
        True
    ],
    [
        "Rear Ride vCar",
        ('vCar', 'hRideR'),
        None,
        [('SM', 0, 0.5)],
        True,
        True
    ],
    [
        "Ride Height Compare",
        ('hRideF', 'hRideR'),
        None,
        0,
        True,
        True
    ],
    [
        "Ride Height Compare Gated",
        ('hRideF', 'hRideR'),
        None,
        0,
        ('SM', '<', 1),
        True,
        True
    ],
    [
        "Plank power acceleration",
        ('gLong (raw)','PPlank_F'),
        None,
        0,
        True,
        True
    ],
    [
        "engine efficiency",
        ('dmInjector', 'PEngine'),
        None,
        [('x', None, None)],
        True,
        True
    ]
]


# ================================================================================
# POWER SPECTRAL DENSITY (PSD) PLOT DEFINITIONS
# ================================================================================
# Frequency-domain analysis plots showing noise/vibration characteristics.
#
# Format:
#   ["Plot Name", 'channel', [(xmin, xmax), (ymin, ymax)], log_scale, nperseg(optional)]
#
# Parameters:
#   - channel:  Channel to analyze
#   - xmin/xmax: Frequency range in Hz (Hz on X-axis)
#   - ymin/ymax: Power range (usually log scale)
#   - log_scale: Use logarithmic Y-axis
#   - nperseg:  (optional) FFT window size; larger = higher frequency resolution

PSD_PLOT_DEFINITIONS = [
    [
        "Front Vertical Acceleration PSD", 'gVertF', [(0, 50), (1e-4, None)], True
    ],
    [
        "Rear Vertical Acceleration PSD", 'gVertR', [(0, 50), (1e-4, None)], True
    ],
    [
        "Front Ride PSD", 'hRideF (raw)', [(0, 50), (1e-4, None)], True
    ],
    [
        "Rear Ride PSD", 'hRideR (raw)', [(0, 50), (1e-4, None)], True
    ]
]


# ================================================================================
# HISTOGRAM PLOT DEFINITIONS
# ================================================================================
# Distribution plots showing value frequency across the dataset.
#
# Format:
#   ["Plot Name", 'channel', [(xmin, xmax), (ymin, ymax)], log_scale]

HISTOGRAM_PLOT_DEFINITIONS = [
    [
        "Plank Power Distribution", 'PPlank_F', [(1, 51), (None, None)], False
    ]
]


# ================================================================================
# BAR PLOT DEFINITIONS
# ================================================================================
# Aggregate bar charts (max, min, mean, integral, etc. per run).
#
# Format:
#   ["Name", ("channel1", "channel2", ...)]
#   ["Name", (("channel1", "integral"), ("channel2", "sum")), default_agg, (ymin, ymax)]
#
# Aggregation modes:
#   - "integral": Integrate over time (area under curve)
#   - "sum": Sum all values
#   - "last": Use final value
#   - "mean": Average value
#   - "max": Maximum value
#   - "min": Minimum value

BAR_PLOT_DEFINITIONS = [
   ["Cumulative Metrics", (("dmInjector (kg/s)", "integral"),)],
]

BOX_PLOT_DEFINITIONS = [
    # Example box plot definition:
    # ["Box Plot Name", 'channel', [(ymin, ymax)], log_scale]
]
# ================================================================================
# CONFIGURATION: POWERPOINT EXPORT MAP
# ================================================================================
# Map plot images to PowerPoint slide layouts and positions.
# Each entry specifies which slide gets which plots.
#
# Format:
#   slide_number: {
#       'layout': 'layout_name',      # 'main_plot' (full-width) or 'double_plot' (two side-by-side)
#       'images': ['image1.png', 'image2.png']  # Files generated by plot_data()
#   }
#
# The 'images' list order matters for 'double_plot' layout:
#   - First image goes to left panel
#   - Second image goes to right panel

POWERPOINT_EXPORT_MAP = {
    4: {'layout': 'main_plot', 'images': ['waveform_Driver_Input.png']},
    5: {'layout': 'main_plot', 'images': ['waveform_Power_Unit.png']},
    
    6: {'layout': 'double_plot', 'images': ['scatter_Gear_Ratios.png', 'scatter_Engine_Power.png']},
    7: {'layout': 'double_plot', 'images': ['bar_Cumulative_Metrics.png', 'scatter_engine_efficiency.png']},
    
    8: {'layout': 'double_plot', 'images': ['scatter_Long_Acceleration.png', 'scatter_Lat_Acceleration.png']},
    9: {'layout': 'double_plot', 'images': ['scatter_GG_Plot.png', 'scatter_Understeer_Plot.png']},
    
    10: {'layout': 'double_plot', 'images': ['scatter_Yaw_Rate_Response.png', 'scatter_Lateral_Acceleration_Response.png']},
    11: {'layout': 'double_plot', 'images': ['scatter_Braking_Efficiency.png', 'scatter_Steering_Moment.png']},
    
    12: {'layout': 'double_plot', 'images': ['scatter_Damper_gLat_front.png', 'scatter_Damper_gLat_rear.png']},
    13: {'layout': 'double_plot', 'images': ['scatter_Pushrod_gLat_front.png', 'scatter_Pushrod_gLat_rear.png']},
    
    14: {'layout': 'double_plot', 'images': ['scatter_Front_Heave.png', 'scatter_Rear_Heave.png']},
    15: {'layout': 'double_plot', 'images': ['scatter_Front_Roll.png', 'scatter_Rear_Roll.png']},
    
    16: {'layout': 'double_plot', 'images': ['scatter_Front_Pushrod_vCar.png', 'scatter_Rear_Pushrod_vCar.png']},
    17: {'layout': 'double_plot', 'images': ['scatter_Front_Ride_vCar.png', 'scatter_Rear_Ride_vCar.png']},
    
    18: {'layout': 'double_plot', 'images': ['scatter_Ride_Height_Compare.png', 'scatter_Roll_angle_gLat.png']},
    
    19: {'layout': 'double_plot', 'images': ['psd_Front_Vertical_Acceleration_PSD.png', 'psd_Rear_Vertical_Acceleration_PSD.png']},
    20: {'layout': 'double_plot', 'images': ['psd_Front_Ride_PSD.png', 'psd_Rear_Ride_PSD.png']},
    
    21: {'layout': 'main_plot', 'images': ['waveform_Plank_Wear.png']},
    22: {'layout': 'double_plot', 'images': ['scatter_Plank_Power_Acceleration.png', 'histogram_Plank_Power_Distribution.png']},
}


# ================================================================================
# PLOT DEFINITIONS AGGREGATOR
# ================================================================================
# Combine all plot type definitions into a single tuple for processing.

PLOT_DEFINITIONS = build_plot_groups(
    WAVEFORM_PLOT_DEFINITIONS,
    SCATTER_PLOT_DEFINITIONS,
    PSD_PLOT_DEFINITIONS,
    HISTOGRAM_PLOT_DEFINITIONS,
    BAR_PLOT_DEFINITIONS,
    BOX_PLOT_DEFINITIONS,
)


def build_plotter():
    """Build the configured DataPlotter instance for the correlation report."""
    return runtime_build_plotter(
        root_folder=ROOT_FOLDER,
        output_dir=CORRELATION_OUTPUT_DIR,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS,
        units_map=UNITS_MAP,
        template_path=POWERPOINT_TEMPLATE,
        export_map=POWERPOINT_EXPORT_MAP,
        scatter_render_mode=SCATTER_RENDER_MODE,
        scatter_density_threshold=SCATTER_DENSITY_THRESHOLD,
        scatter_max_points=SCATTER_MAX_POINTS,
        scatter_hexbin_gridsize=SCATTER_HEXBIN_GRIDSIZE,
        bar_secondary_axis_ratio=BAR_SECONDARY_AXIS_RATIO,
    )

# ================================================================================
# MAIN EXECUTION
# ================================================================================
# This section runs when the script is executed directly.
if __name__ == "__main__":
    run_plot_job(
        title="CORRELATION PLOT GENERATION",
        plotter=build_plotter(),
        plot_method="plot_data",
        generate_message="Generating plots...",
        powerpoint_template=POWERPOINT_TEMPLATE if EXPORT_TO_POWERPOINT else None,
        powerpoint_output=POWERPOINT_OUTPUT if EXPORT_TO_POWERPOINT else None,
        export_map=POWERPOINT_EXPORT_MAP if EXPORT_TO_POWERPOINT else None,
    )

