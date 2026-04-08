"""
Correlation Report Generator
--------------------------------
Loads multiple runs (any number), applies mapping/transform rules,
generates waveform / scatter / PSD plots, then exports to PowerPoint.
"""

import os
from pathlib import Path

from dataplotter import DataPlotter
from powerpointexporter import (
    export_report_to_powerpoint,
    get_template_plot_aspect_ratios
)

# =====================================================================
# CONFIGURATION
# =====================================================================

# Root folder for input data and output plots
ROOT_FOLDER = Path(r"C:\GitHub_Local\DLS-Correlation-Tool\Data")

# ---------------------------------------------------------------------
# RUN DEFINITIONS (MULTI‑RUN FRIENDLY)
# ---------------------------------------------------------------------

RUNS = [
    {"name": "car", "file": "Run TXT Files\\26R03SUZ_260328_MAC26-02_BOT_Q_R03.txt", "color": "#FF8C00"},
    {"name": "dls", "file": "Run TXT Files\\26R03SUZ  11  FP3  Run 3 3  Baseline_LTS_Iteration_4.txt", "color": "#002FFF"},
    {"name": "oc", "file": "Run TXT Files\\Lap001_20260406-OC-VPG - Correlation - DiL 2603256 FIT SUZ Support Run 7 - Multi - v1-SUZ.oc.txt", "color": "#37FF00"},
    # Add additional runs here:
    # {"name": "run2", "file": "somefile.csv", "color": "#0000FF"},
]

# ---------------------------------------------------------------------
# POWERPOINT TEMPLATE
# ---------------------------------------------------------------------
POWERPOINT_TEMPLATE = ROOT_FOLDER / "template.pptx"
POWERPOINT_OUTPUT = ROOT_FOLDER / "DLS_Correlation_Report.pptx"
EXPORT_TO_POWERPOINT = True

# =====================================================================
# CHANNEL MAPPINGS
CHANNEL_MAPPINGS = {
    'oc': {
    # Example mappings - adjust based on your data:
    'rSLMActive' : 'SM'
    },
    'dil': {
    # Example mappings - adjust based on your data:
    'BSLMActiveCan': 'SM',
    'FPushrodFL': 'FPRodFL',
    'FPushrodFR': 'FPRodFR',
    'FPushrodRL': 'FPRodRL',
    'FPushrodRR': 'FPRodRR',
    'EPlankWearLapF' : 'EPlankF',
    'PPlankWearF' : 'PPlankF',
    'pBrakeF1' : 'pBrakeF',
    'CAN_6_632_aSteerWheel_Can' : 'aSteerWheel',
    },
    'dls': {
    # Example mappings - adjust based on your data:
    'aRollCarTrack': 'aRoll',
    'FPushrodFL': 'FPRodFL',
    'FPushrodFR': 'FPRodFR',
    'FPushrodRL': 'FPRodRL',
    'FPushrodRR': 'FPRodRR',
    'aUndersteer_aSlip': 'aUndersteerFromSlip',
    'BAeroModeXDriver': 'SM',
    'rThrottlePedal': 'rThrottle',
    'EPlankLTS_Lap' : 'EPlankF',
    'PPlankWearF' : 'PPlankF'
    },
    'car': {
    'BNSLMEnablingStatusEnabled': 'SM',
    'PMGUKActual': 'PMGUK',
    'rThrottlePedal': 'rThrottle',
    'xDamperPotFL': 'xDamperFL',
    'xDamperPotFR': 'xDamperFR',
    'xDamperPotRL': 'xDamperRL',
    'xDamperPotRR': 'xDamperRR',
    'nYawSlipSensor': 'nYaw',
    'EPlankWearLapF': 'EPlankF',
    'PPlankWearF': 'PPlankF',
    }
    ## Add more runs and their channel mappings as needed
}

# =====================================================================
# CHANNEL TRANSFORMS
# =====================================================================
CHANNEL_TRANSFORMS = {
    'oc' : None,
    'dil' : None,
    "dls": {
        # DLS load sign corrections
        "FPRodFL": lambda x: -x,
        "FPRodFR": lambda x: -x,
        "FPRodRL": lambda x: -x,
        "FPRodRR": lambda x: -x,
        "aRoll": lambda x: -x,
    },
    "car": {
        "PMGUK": lambda x: x / 1000,   # W → kW
        "sLap": lambda x: x - 10,      # GPS alignment shift
    }
}

# =====================================================================
# UNITS MAPPING
# =====================================================================
UNITS_MAP = {
    "glat": "g", "glong": "g", "gvertf": "g", "gvertr": "g", "glat_abs": "g",
    "gLong (unsmoothed)": "g",
    "vcar": "kph",
    "aroll": "°", "asteer": "°", "asteerwheel": "°", "aundersteerfromslip": "°",
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
    "nyaw": "°/s",
    "pbrakef": "bar",
    "rthrottle": "%",
    "EPlankF": "kJ",
    "PPlankF": "kW",
    "FzPlankF": "N",
}

# =====================================================================
# CALCULATED CHANNELS
# =====================================================================
CALCULATED_CHANNELS = {
    "FPRodDeltaF": lambda df: df["FPRodFL"] - df["FPRodFR"],
    "FPRodDeltaR": lambda df: df["FPRodRL"] - df["FPRodRR"],
    "FPRodAvgF": lambda df: (df["FPRodFL"] + df["FPRodFR"]) / 2,
    "FPRodAvgR": lambda df: (df["FPRodRL"] + df["FPRodRR"]) / 2,
    "xDamperDeltaF": lambda df: df["xDamperFL"] - df["xDamperFR"],
    "xDamperDeltaR": lambda df: df["xDamperRL"] - df["xDamperRR"],
    "xDamperAvgF": lambda df: (df["xDamperFL"] + df["xDamperFR"]) / 2,
    "xDamperAvgR": lambda df: (df["xDamperRL"] + df["xDamperRR"]) / 2,
    "gLat_Abs": lambda df: df["gLat"].abs(),
    "gLong (unsmoothed)": lambda df: df["gLong"],
}

# =====================================================================
# LOW-PASS FILTER SETTINGS
# =====================================================================
LOW_PASS_FILTERS = {
    "gVertF": {"cutoff": 0, "order": 2},
    "gVertR": {"cutoff": 0, "order": 2},
    "FzPlankF": {"cutoff": 0, "order": 2},
    "PMGUK": {"cutoff": 0, "order": 2},
    "SM": {"cutoff": 0, "order": 2},
    "NGear": {"cutoff": 0, "order": 2},
    "nEngine": {"cutoff": 0, "order": 2},
    "nWheelR_Avg": {"cutoff": 0, "order": 2},
    "EPlankF": {"cutoff": 0, "order": 2},
    "PPlankF": {"cutoff": 0, "order": 2},
    "rThrottle": {"cutoff": 0, "order": 2},
    "all": {"cutoff": 5, "order": 2},
}

# =====================================================================
# PLOT DEFINITIONS
# =====================================================================

WAVEFORM_PLOT_DEFINITIONS = [
   # ["Name", (channels...), ((ymin,ymax)...), (reference lines...), (subplot height ratios...)]
   # `subplot height ratios` is optional; omit it to give every channel the same height.
    [
        "Driver Input", ('SM','PMGUK', 'NGear','vCar', 'aSteerWheel' , 'pBrakeF', 'rThrottle'),
        ((-0.2, 1.2), (-360, 360), (1, 9), (60, 360), (-160, 160), (-10, 80), (-5, 105)), # y-axis limits for each channel
        (None, (-350, 0, 350), None, None, (0), None, None), # reference lines for each channel
        (0.1, 0.7, 0.6, 1,0.6, 0.35, 0.35) # subplot height ratios (optional)
    ],
    [
        "Power Unit", ('PMGUK', 'PEngine','NGear','vCar', 'nEngine', 'gLong' , 'pBrakeF', 'rThrottle'),
        ((-360, 360), (-100, 500), (1, 9), (60, 360), (7000, 13000), None, (-10, 80), (-5,105)), # y-axis limits for each channel
        ((-350, 0, 350), (0), None, None, (10000), (0), None, None), # reference lines for each channel
        (0.4, 0.4, 0.3, 0.7, 0.5, 0.5, 0.35, 0.35) # subplot height ratios (optional)
    ],
    [
        "Plank Wear", ('SM', 'PMGUK','vCar','FzPlankF', 'EPlankF' , 'pBrakeF', 'rThrottle'),
        ((-0.1, 1.1), (-360, 360), (60, 360), (0, 8000), (0, 100), (-10, 80), (-5,105)), # y-axis limits for each channel
        (None, (-350, 0, 350), (0,7500), (0), None, None), # reference lines for each channel
        (0.1, 0.6, 0.8, 0.7, 0.6, 0.35, 0.35) # subplot height ratios (optional)
    ],
] 

SCATTER_PLOT_DEFINITIONS = [
    ["Gear Ratios", ('nWheelR_Avg', 'nEngine'), None, 0],
    ["Engine Power", ('nEngine', 'PEngine'), None, 0],
    ["Long Acceleration", ('vCar', 'gLong'), [(60,360),(None,None)], 0],
    ["Lat Acceleration", ('vCar', 'gLat_Abs'), [(60,360),(None,None)], 0],
    ["GG Plot", ('gLat', 'gLong'), None , 0],
    ["Understeer Plot", ('vCar', 'aUndersteerFromSlip'), None , 0],
    ["Yaw Rate Response", ('aSteerWheel', 'nYaw'), None , 0],
    ["Lateral Acceleration Response", ('aSteerWheel', 'gLat'), None , 0],
    ["Braking Efficiency", ('pBrakeF', 'gLong'), [(None,None),(-5,0)] , [('y', None, -0.3)]],
    ["Damper gLat front", ('gLat', 'xDamperDeltaF'), None , [('x', None, None)]],
    ["Damper gLat rear", ('gLat', 'xDamperDeltaR'), None , [('x', None, None)]],
    ["Pushrod gLat front", ('gLat', 'FPRodDeltaF'), None , [('x', None, None)]],
    ["Pushrod gLat rear", ('gLat', 'FPRodDeltaR'), None , [('x', None, None)]],
    ["Front Heave", ('xDamperAvgF', 'FPRodAvgF'), None, [('y', None, 10000), ('y', 10000, None)]],
    ["Front Roll", ('xDamperDeltaF', 'FPRodDeltaF'), None, [('x', None, None)]],
    ["Rear Heave", ('xDamperAvgR', 'FPRodAvgR'), None, [('x', None, None)]],
    ["Rear Roll", ('xDamperDeltaR', 'FPRodDeltaR'), None, [('x', None, None)]],
    ["Front Pushrod vCar", ('vCar', 'FPRodAvgF'), None, [('x', None, None)]],
    ["Rear Pushrod vCar", ('vCar', 'FPRodAvgR'), None, [('x', None, None)]],
    ["Front Ride vCar", ('vCar', 'hRideF'), [(None,None), (0,40)], [('x', None, None)]],
    ["Rear Ride vCar", ('vCar', 'hRideR'), [(None,None),(20,80)], [('x', None, None)]],
    ["Ride Height Compare", ('hRideF', 'hRideR'), [(0, 40),(20, 70)], 0],
    ["Roll angle gLat", ('gLat', 'aRoll'), None, [('x', None, None)]],
    ["Steering Moment", ('aSteerWheel', 'MSteerWheel'), None, 0],
    ["Plank power acceleration", ('gLong (unsmoothed)', 'PPlankF'), None, 0],
] 

PSD_PLOT_DEFINITIONS = [
   # ["Name of Plot", 'channel', [(xmin, xmax), (ymin, ymax)], log_scale, nperseg(optional)]
    ["Front Vertical Acceleration PSD", 'gVertF', [(0, 50), (None, None)], False],
    ["Rear Vertical Acceleration PSD", 'gVertR', [(0, 50), (None, None)], False],
]

HISTOGRAM_PLOT_DEFINITIONS = [
   # ["Name of Plot", 'channel', [(xmin, xmax), (ymin, ymax)], log_scale]
    ["Plank Power Distribution", 'PPlankF', [(1,45), (None, None)], False],
]

POWERPOINT_EXPORT_MAP = {
    4: {'layout': 'main_plot', 'images': ['waveform_Driver_Input.png']},
    5: {'layout': 'main_plot', 'images': ['waveform_Power_Unit.png']},
    6: {'layout': 'double_plot', 'images': ['scatter_Gear_Ratios.png', 'scatter_Engine_Power.png']},
    7: {'layout': 'double_plot', 'images': ['scatter_Long_Acceleration.png', 'scatter_Lat_Acceleration.png']},
    8: {'layout': 'double_plot', 'images': ['scatter_GG_Plot.png', 'scatter_Understeer_Plot.png']},
    9: {'layout': 'double_plot', 'images': ['scatter_Yaw_Rate_Response.png', 'scatter_Lateral_Acceleration_Response.png']},
    10: {'layout': 'double_plot', 'images': ['scatter_Braking_Efficiency.png', 'scatter_Steering_Moment.png']},
    11: {'layout': 'double_plot', 'images': ['scatter_Damper_gLat_front.png', 'scatter_Damper_gLat_rear.png']},
    12: {'layout': 'double_plot', 'images': ['scatter_Pushrod_gLat_front.png', 'scatter_Pushrod_gLat_rear.png']},
    13: {'layout': 'double_plot', 'images': ['scatter_Front_Heave.png', 'scatter_Rear_Heave.png']},
    14: {'layout': 'double_plot', 'images': ['scatter_Front_Roll.png', 'scatter_Rear_Roll.png']},
    15: {'layout': 'double_plot', 'images': ['scatter_Front_Pushrod_vCar.png', 'scatter_Rear_Pushrod_vCar.png']},
    16: {'layout': 'double_plot', 'images': ['scatter_Front_Ride_vCar.png', 'scatter_Rear_Ride_vCar.png']},
    17: {'layout': 'double_plot', 'images': ['scatter_Ride_Height_Compare.png', 'scatter_Roll_angle_gLat.png']},
    18: {'layout': 'double_plot', 'images': ['psd_Front_Vertical_Acceleration_PSD.png', 'psd_Rear_Vertical_Acceleration_PSD.png']},
    19: {'layout': 'main_plot', 'images': ['waveform_Plank_Wear.png']},
    20: {'layout': 'double_plot', 'images': ['scatter_Plank_Power_Acceleration.png', 'histogram_Plank_Power_Distribution.png']},
}

PLOT_DEFINITIONS = (
    WAVEFORM_PLOT_DEFINITIONS if WAVEFORM_PLOT_DEFINITIONS else [],
    SCATTER_PLOT_DEFINITIONS if SCATTER_PLOT_DEFINITIONS else [],
    PSD_PLOT_DEFINITIONS if PSD_PLOT_DEFINITIONS else [],
    HISTOGRAM_PLOT_DEFINITIONS if HISTOGRAM_PLOT_DEFINITIONS else []
)

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("              CORRELATION PLOT GENERATION")
    print("=" * 60)

    # Extract PPT-defined target aspect ratios
    plot_aspect_ratios = get_template_plot_aspect_ratios(
        POWERPOINT_TEMPLATE,
        POWERPOINT_EXPORT_MAP
    )

    # Initialise main plotter
    plotter = DataPlotter(
        root_folder=ROOT_FOLDER,
        runs=RUNS,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS,
        units_map=UNITS_MAP,
        plot_aspect_ratios=plot_aspect_ratios,
    )

    # Generate all plots
    plotter.plot_all()

    # Optional: export to PowerPoint
    if EXPORT_TO_POWERPOINT:
        export_report_to_powerpoint(
            template_path=POWERPOINT_TEMPLATE,
            output_path=POWERPOINT_OUTPUT,
            plots_dir=plotter.plots_dir,
            export_map=POWERPOINT_EXPORT_MAP,
            visible=False,
        )

        os.startfile(POWERPOINT_OUTPUT)
