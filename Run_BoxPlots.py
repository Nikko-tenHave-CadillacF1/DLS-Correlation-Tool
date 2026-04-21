"""
================================================================================
BOX PLOT ANALYSIS TOOL
================================================================================
Workflow:
    1. Define run data files and box plot specifications
    2. DataPlotter loads and processes telemetry data from each run
    3. Applies channel transforms (unit conversions, sign corrections, etc)
    4. Generates configured box plots with aggregation modes:
       - per_run:     One box per run (compare distributions across runs)
       - aggregated:  Single aggregated box across all runs
    5. Supports optional gates for conditional analysis

Key Configuration Sections:
    - RUNS: Define which data files to load and their visual properties
    - BOX_PLOT_DEFINITIONS: Specify channels, aggregation mode, gates, limits
    - BOX_PLOT_SETTINGS: Control rendering options (colors, point overlay, etc)
================================================================================
"""

from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid

from dataplotter import DataPlotter

# ================================================================================
# CONFIGURATION: DATA FILES & LOCATIONS
# ================================================================================
# Root folder for input data and output plots

ROOT_FOLDER = Path(r"C:\\GitHub_Local\\DLS-Correlation-Tool\\Data\\Run TXT Files\\Fuel Investigation\\")

# ================================================================================
# CONFIGURATION: RUN DEFINITIONS
# ================================================================================
# Load any number of telemetry runs. Each run specifies a data file, display name,
# color, and optional filtering criteria.
#
# Format:
#   {"name": "<run_id>", "file": "<relative_path_from_Data>", "color": "<#RRGGBB>", 
#    "nrun": <optional_ranked_run_index>, "nlap": <optional_lap_number>, "type": "<DATA_TYPE>"}

RUNS = [
    {
        "name": "T01BCN - R4",
        "file": "26T01BCN_260129_MAC26-01_PER_R04PARTIAL.txt",
        "color": "#D70000",
        "type": "CAR"
    },
    {
        "name": "T01BCN - R5",
        "file": "26T01BCN_260129_MAC26-01_PER_R05PARTIAL.txt",
        "color": "#06B300",
        "type": "CAR"
    },
    {
        "name": "T01BCN - R6",
        "file": "26T01BCN_260129_MAC26-01_PER_R06PARTIAL.txt",
        "color": "#008CFF",
        "type": "CAR"
    },
    {
        "name": "T01BCN - R7",
        "file": "26T01BCN_260129_MAC26-01_PER_R07PARTIAL.txt",
        "color": "#EA00FF",
        "type": "CAR"
    },

    # {
    #     "name": "T03BAH",
    #     "file": "26T03BAH_260220_MAC26-03_BOT_D3_R10.txt",
    #     "color": "#D77300",
    #     "type": "CAR"
    # },
    # {
    #     "name": "R01MEL - Q",
    #     "file": "26R01MEL_260307_MAC26-01_PER_Q_R03.txt",
    #     "color": "#48CB01",
    #     "type": "CAR"
    # },
    # {
    #     "name": "R01MEL - R",
    #     "file": "26R01MEL_260308_MAC26-01_PER_GP_R02.txt",
    #     "color": "#007BBD",
    #     "type": "CAR"
    # },
    #     {
    #     "name": "R02SHA - Q",
    #     "file": "26R02SHA_260314_MAC26-02_BOT_Q_R03.txt",
    #     "color": "#1100FF",
    #     "type": "CAR"
    # },
    # {
    #     "name": "R02SHA - R",
    #     "file": "26R02SHA_260315_MAC26-02_BOT_GP_R02.txt",
    #     "color": "#3002E9",
    #     "type": "CAR"
    # },
    #     {
    #     "name": "R03SUZ - Q",
    #     "file": "26R03SUZ_260328_MAC26-01_PER_Q_R02.txt",
    #     "color": "#FF00FB",
    #     "type": "CAR"
    # },
    # {
    #     "name": "R03SUZ - R",
    #     "file": "26R03SUZ_260329_MAC26-01_PER_GP_R02.txt",
    #     "color": "#FF006F",
    #     "type": "CAR"
    # },
]


# ================================================================================
# CONFIGURATION: CHANNEL MAPPINGS
# ================================================================================
# Map source channel names to standardized names for consistency across data types.

CHANNEL_MAPPINGS = {
    'CAR': {
        'rThrottlePedal': 'rThrottle',
    }
}


# ================================================================================
# CONFIGURATION: CHANNEL TRANSFORMS
# ================================================================================
# Apply mathematical transformations to channels (unit conversions, sign corrections).

CHANNEL_TRANSFORMS = {
    "CAR": None
}


# ================================================================================
# CONFIGURATION: CALCULATED CHANNELS
# ================================================================================
# Define derived channels computed from raw channel data.

CALCULATED_CHANNELS = {
    "dmInjector (kg/s)": lambda df: df["dmInjector"] / 3600,
    "rLambda_avg (%)": lambda df: 100 * (df["rLambdaL"] + df["rLambdaR"]) / 2,
    "dmExhaust": lambda df: df["dmInjector"] * (1 + 13.23 * df["rLambda_avg (%)"] / 100)/3600,
    "dmExhaust_Estimated": lambda df: df["dmInjector"] * (1 + 13.23 * (0.155 * df["vCar"] + 109.329)/ 100)/3600,
    "Error_dmExhaust": lambda df: df["dmExhaust"] - df["dmExhaust_Estimated"],
    "CosPhi_Calc" : lambda df: df["gLong"] / np.sqrt(df["glat"]**2 + df["gLong"]**2) 
}


# ================================================================================
# CONFIGURATION: LOW-PASS FILTER SETTINGS
# ================================================================================
# Apply low-pass filtering to smooth noisy channels.

LOW_PASS_FILTERS = {
    "rLambdaL": {"cutoff": 5, "order": 2},
    "rLambdaR": {"cutoff": 5, "order": 2},
    "rLambda_avg": {"cutoff": 3, "order": 2},
    "dmInjector": {"cutoff": 5, "order": 2},
    "dmFFMFuel": {"cutoff": 5, "order": 2},
    "CosPhi": {"cutoff": 5, "order": 2},
    "CosPhi_Calc": {"cutoff": 5, "order": 2},
    "gLong": {"cutoff": 3, "order": 2},
    "gLat": {"cutoff": 5, "order": 2},
    #"dmExhaust": {"cutoff": 5, "order": 2},
}


# ================================================================================
# CONFIGURATION: UNITS MAPPING
# ================================================================================
# Map channel names to their physical units.

UNITS_MAP = {
    "glat": "g", "glong": "g", "gvertf": "g", "gvertr": "g", "glat_abs": "g",
    "gLong (raw)": "g", "gVert": "g",
    
    "vcar": "kph",
    
    "aroll": "deg", "asteer": "deg", "asteerwheel": "deg", "aundersteerfromslip": "deg",
    
    "xrh": "mm", "laser": "mm", "hrider": "mm", "hridef": "mm",
    "damper": "mm", "xdamper": "mm",
    
    "fprod": "N", "fpushrod": "N", "pushrod": "N",
    "fprodfl": "N", "fprodfr": "N", "fprodrl": "N", "fprodrr": "N",
    "fprodavgf": "N", "fprodavgr": "N",
    "fproddeltaf": "N", "fproddeltar": "N",
    "trackrod": "N",
    
    "xdamperavgf": "mm", "xdamperavgr": "mm",
    "xdamperdeltaf": "mm", "xdamperdeltar": "mm",
    
    "nengine": "rpm", "mengine": "Nm", "msteerwheel": "Nm",
    
    "brake": "bar", "throttle": "%", "pmguk": "kW", "pengine": "kW",
    "nwheelr_avg": "rpm",
    
    "nyaw": "deg/s",
    "pbrakef": "bar",
    "rthrottle": "%",
    
    "EPlank_F": "kJ",
    "PPlank_F": "kW",
    "FzPlankF": "N",

    "dmInjector": "kg/h",
    "PMGUK_Deploy": "kW",
    "PMGUK_Charge": "kW",
    "dmInjector (kg/s)": "kg/s",
}


# ================================================================================
# CONFIGURATION: BOX PLOT DEFINITIONS
# ================================================================================
# Define box plots to generate.
#
# Format:
#   ["Plot Name", "channel" or ("ch1", "ch2", ...), aggregation_mode, axis_limits, gate_spec, options]
#
# Parameters:
#   - channels:         Single channel (string) or tuple of channels
#   - aggregation_mode: "per_run" (boxes by run) or "aggregated" (one box all runs)
#   - axis_limits:      (ymin, ymax) or None
#   - gate_spec:        Filter condition, e.g., ('vCar', '>', 100) or None
#   - options:          Dict with rendering options (show_points, jitter, etc)
#
# Example per_run (compare distributions across runs):
#   ["vCar Distribution", "vCar", "per_run", (50, 350), None, {}]
#
# Example aggregated (overall distribution from all runs):
#   ["Overall vCar", "vCar", "aggregated", None, ("vCar", ">", 100), {}]
#
# Example multi-channel aggregated (multiple subplots):
#   ["Dynamics", ("gLat", "nYaw", "aSteerWheel"), "aggregated", None, None, {}]


WAVEFORM_DEFINITIONS = [
    [
        "vCar Comparison",
        ('vCar', 'rLambda_avg (%)', 'gLat', 'CosPhi' ,'pBrakeF' , 'rThrottle', 'SM'),
        (None, None, None, None, None, None, None),
        (None, None, None, None, None, None, None),
        (0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4)
    ],
]

# Format:
#   ["Name", (x_channel, y_channel), [(xmin, xmax), (ymin, ymax)], best_fit, gate_spec, show_equations, show_error]
#

SCATTER_DEFINITIONS = [
    # [
    #     "Lambda Ratios",
    #     ('vCar', 'rLambda_avg (%)',), 
    #     None, 
    #     [('x', None, 200), ('x', 200, None)],
    #     [("CosPhi", "between", (0, 0.9))],
    #     True,
    #     True
    # ],
    [
        "dmInjector ",
        ('vCar', 'gLat'), 
        None, 
        [('x', None, 200), ('x', 200, None)],
        [("CosPhi", "between", (-1, 0.9))],
        True,
        True
    ],
    [
        "vCar vs dmExhaust - T5",
        ('vCar', 'dmExhaust',), 
        ((None, None), (None, None)), 
        [("x", None, 125)],
        [("CosPhi", "between", (-0.9, 0.9))],
        True,
        True
    ],
    # [
    #     "dmExhaust - T1-2",
    #     ('CosPhi', 'dmExhaust',), 
    #     ((-1, 1), (None, None)), 
    #     None,
    #     [("sLap", "between", (750, 1000))],
    #     True,
    #     True
    # ],
    # [
    #     "dmExhaust - T4",
    #     ('CosPhi', 'dmExhaust',), 
    #     ((-1, 1), (None, None)), 
    #     None,
    #     [("sLap", "between", (1600, 1850))],
    #     True,
    #     True
    # ],
    # [
    #     "dmExhaust - T5",
    #     ('CosPhi', 'dmExhaust',), 
    #     ((-1, 1), (None, None)), 
    #     None,
    #     [("sLap", "between", (2000, 2250))],
    #     True,
    #     True
    # ],
    # [
    #     "dmExhaust - T7-8",
    #     ('CosPhi', 'dmExhaust',), 
    #     ((-1, 1), (None, None)), 
    #     None,
    #     [("sLap", "between", (2450, 2650))],
    #     True,
    #     True
    # ],
    # [
    #     "dmExhaust - T10",
    #     ('CosPhi', 'dmExhaust',), 
    #     ((-1, 1), (None, None)), 
    #     None,
    #     [("sLap", "between", (3350, 3550))],
    #     True,
    #     True
    # ],
    # [
    #     "dmExhaust - T12",
    #     ('CosPhi', 'dmExhaust',), 
    #     ((-1, 1), (None, None)), 
    #     None,
    #     [("sLap", "between", (3650, 3850))],
    #     True,
    #     True
    # ],
    [
        "dmInjector vs CosPhi",
        ('CosPhi_Calc', 'dmInjector',), 
        None, 
        None,
        True,
        True
    ],

]

BOX_PLOT_DEFINITIONS = [
    # Per-run comparison: which run has more variable vCar?
    [
        "Low Speed Corner Distribution",
        "vCar",
        "per_run",
        (None, None),
        [("gLong", "between", (-0.1, 0.1)), ("vCar", "<", 120)],
        {}
    ],
]


# ================================================================================
# CONFIGURATION: BOX PLOT RENDERING SETTINGS
# ================================================================================
# Control appearance of box plots.

BOX_PLOT_SETTINGS = {
    # Display options
    "show_fliers": False,                   # Hide statistical outliers (keeps plots clean)
    
    # Box styling
    "box_width": 0.65,                      # Width of box relative to spacing
    "box_linewidth": 1.8,                   # Box border line width (professional thickness)
    "medianline_color": "#1A1A1A",          # Median line color (dark)
    "medianline_width": 2.5,                # Median line width (prominent)
    
    # Per-run mode: use run colors
    # Aggregated mode: use unified color scheme
    "aggregated_box_color": "#2E7D99",      # Professional teal-blue for aggregated boxes
    "aggregated_box_alpha": 0.75,
    "per_run_box_alpha": 0.75,
    
    # Figure sizing
    "figsize_single_channel": (10, 6),      # Single channel plot
    "figsize_multi_channel": (14, 10),      # Multiple channels (subplots)
}


# ================================================================================
# PLOT DEFINITIONS AGGREGATOR
# ================================================================================
# Combine plot definitions into tuple format expected by DataPlotter.
# Index 5 is reserved for box plots.

PLOT_DEFINITIONS = (
    WAVEFORM_DEFINITIONS if WAVEFORM_DEFINITIONS else [],  # 0: Waveforms
    SCATTER_DEFINITIONS if SCATTER_DEFINITIONS else [],  # 1: Scatter
    [],                                 # 2: PSD
    [],                                 # 3: Histogram
    [],                                 # 4: Bar
    BOX_PLOT_DEFINITIONS if BOX_PLOT_DEFINITIONS else [],  # 5: Box plots
)


# ================================================================================
# MAIN EXECUTION
# ================================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 25 + "BOX PLOT ANALYSIS - STARTING")
    print("=" * 80 + "\n")

    # Initialize data plotter with all configurations
    plotter = DataPlotter(
        root_folder=ROOT_FOLDER,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS,
        units_map=UNITS_MAP,
        box_plot_settings=BOX_PLOT_SETTINGS,
        fig_size=[(15.5, 6.4), (10, 8), (10, 8), (10, 8), (10, 6)],
    )

    # Generate box plots only
    print("\nGenerating box plots...")
    plotter.plot_all()

    print("\n" + "=" * 80)
    print(" " * 25 + "BOX PLOT GENERATION COMPLETE")
    print("=" * 80 + "\n")
