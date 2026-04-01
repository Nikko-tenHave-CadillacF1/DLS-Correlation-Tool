"""
Correlation Scatterplots - Multi-Run Data Plotter for Track and DiL Data
Allows flexible configuration of multiple runs, channel mappings, transformations, and plotting.
"""
from pathlib import Path
from dataplotter import DataPlotter
from powerpointexporter import export_report_to_powerpoint

# ======== CONFIGURATION ========

# Root folder containing data files
ROOT_FOLDER = r'C:\GitHub_Local\DLS-Correlation-Tool\Data'

DLS_RUN = {'name': 'DLS', 'file': '26R03SUZ  77  FP2  Run 2 2 P2R2 NC3 Qsim  Stint 1 2 P2R2 NC3 Qsim_DLS_1.txt', 'color': '#0000FF'}

TRACK_RUN = {'name': 'Track', 'file': '26R03SUZ_260327_MAC26-02_BOT_P2_R02_1.txt', 'color': "#FF9100"}

POWERPOINT_TEMPLATE = Path(ROOT_FOLDER) / 'template.pptx'
POWERPOINT_OUTPUT = Path(ROOT_FOLDER) / 'DLS_Correlation_Report.pptx'
EXPORT_TO_POWERPOINT = True

# ========= CHANNEL MAPPINGS ========

CHANNEL_MAPPINGS = {
    'dls': {
    # Example mappings - adjust based on your data:
    'FPushrodFL': 'FProdFL',  # DLS has 'aRollCarTrack', Track has 'aRoll'
    'FPushrodFR': 'FProdFR',
    'FPushrodRL': 'FProdRL',
    'FPushrodRR': 'FProdRR',
    'aUndersteer_aSlip': 'aUndersteerFromSlip',
    'BAeroModeXDriver': 'SM',
    'rThrottlePedal': 'rThrottle'
    },
    'track': {
    'BNSLMEnablingStatusEnabled': 'SM',
    'PMGUKActual': 'PMGUK',
    'rThrottlePedal': 'rThrottle',
    'FPRodFL': 'FProdFL',
    'FPRodFR': 'FProdFR',
    'FPRodRL': 'FProdRL',
    'FPRodRR': 'FProdRR',
    'xDamperPotFL': 'xDamperFL',
    'xDamperPotFR': 'xDamperFR',
    'xDamperPotRL': 'xDamperRL',
    'xDamperPotRR': 'xDamperRR',
    'nYawSlipSensor': 'nYaw'
    }
}

CHANNEL_TRANSFORMS = {
    'dls': {
        'FProdFL': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'FProdFR': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'FProdRL': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'FProdRR': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
    },
    'track': {
        'PMGUK': lambda x: x/1000,  #convert from W to kW
        'sLap': lambda x: x - 10, # Shift distance to correct for GPS
    }
}

# ========= CALCULATED CHANNELS ========

CALCULATED_CHANNELS = {
    # Example:
    # 'BrakeBalanceDelta': lambda df: df['pBrakeF'] - df['pBrakeR'],
    'FProdDeltaF': lambda df: df['FProdFL'] - df['FProdFR'], # Calculate front left-right pushrod load difference
    'FProdDeltaR': lambda df: df['FProdRL'] - df['FProdRR'], # Calculate rear left-right pushrod load difference
    'FProdAvgF': lambda df: (df['FProdFL'] + df['FProdFR']) / 2, # Calculate front average pushrod load
    'FProdAvgR': lambda df: (df['FProdRL'] + df['FProdRR']) / 2, # Calculate rear average pushrod load
    'xDamperDeltaF': lambda df: df['xDamperFL'] - df['xDamperFR'], # Calculate front left-right damper displacement difference
    'xDamperDeltaR': lambda df: df['xDamperRL'] - df['xDamperRR'], # Calculate rear left-right damper displacement difference
    'xDamperAvgF': lambda df: (df['xDamperFL'] + df['xDamperFR']) / 2, # Calculate front average damper displacement
    'xDamperAvgR': lambda df: (df['xDamperRL'] + df['xDamperRR']) / 2, # Calculate rear average damper displacement
    'gLat_Abs': lambda df: df['gLat'].abs(), # Absolute value of lateral acceleration
}

# ========= LOW PASS FILTERS ========

LOW_PASS_FILTERS = {
    'gVertF': {'cutoff': 0, 'order': 2},
    'gVertR': {'cutoff': 0, 'order': 2},
    'SM': {'cutoff': 0, 'order': 2},
    'NGear': {'cutoff': 0, 'order': 2},
    'all': {'cutoff': 5, 'order': 2},
}

# ======== PLOT DEFINITIONS ========

WAVEFORM_PLOT_DEFINITIONS = [
   # ["Name", (channels...), ((ymin,ymax)...), (reference lines...), (subplot height ratios...)]
   # `subplot height ratios` is optional; omit it to give every channel the same height.
    [
        "Driver Input", ('SM','NGear','vCar', 'PMGUK', 'aSteerWheel' , 'pBrakeF', 'rThrottle'),
        ((-0.1, 1.1), (1, 9), (60, 360), (-351, 351), (-160, 160), (0, 80), (-1, 101)), # y-axis limits for each channel
        (None, None, None, (-350, 0, 350), (0), None, None), # reference lines for each channel
        (0.2, 0.6, 1, 0.6,0.6, 0.4, 0.4) # subplot height ratios (optional)
    ],
    [
        "Power Unit", ('PMGUK', 'PEngine','NGear','vCar', 'nEngine', 'gLong' , 'pBrakeF', 'rThrottle'),
        ((-351, 351), (-100, 500), (1, 9), (60, 360), (7000, 13000), None, (0, 80), (0,101)), # y-axis limits for each channel
        ((-350, 0, 350), (0), None, None, (10000), (0), None, None), # reference lines for each channel
        (0.4, 0.4, 0.2, 0.8, 0.5, 0.5, 0.4, 0.4) # subplot height ratios (optional)
    ],
    [
        "Plank Wear", ('SM', 'PMGUK','vCar','FzPlankF', 'EPlankF' , 'pBrakeF', 'rThrottle'),
        ((-0.1, 1.1), (-351, 351), (60, 360), (0, 8000), (0, 100), (0, 80), (0,101)), # y-axis limits for each channel
        (None, (-350, 0, 350), (0,7500), (0), None, None), # reference lines for each channel
        (0.2, 0.6, 0.8, 0.6, 0.6, 0.4, 0.4) # subplot height ratios (optional)
    ],
] 

SCATTER_PLOT_DEFINITIONS = [
   #["Name of Plot", ('x Axis', 'y Axis'), [(xmin, xmax), (ymin, ymax)], Best Fit T/F, ('x', x_crossing) or ('y', y_crossing)],
    ["Gear Ratios", ('nWheelR_Avg', 'nEngine'), [(0,400),(None,None)], 0, None],
    ["Engine Power", ('nEngine', 'PEngine'), None, 0, None],
    ["Long Acceleration", ('vCar', 'gLong'), [(60,360),(None,None)], 0, None],
    ["Lat Acceleration", ('vCar', 'gLat_Abs'), [(60,360),(None,None)], 0, None],
    ["GG Plot", ('gLat', 'gLong'), None , 0, None],
    ["Understeer Plot", ('vCar', 'aUndersteerFromSlip'), None , 0, None],
    ["Yaw Rate Response", ('aSteerWheel', 'nYaw'), None , 0, None],
    ["Lateral Acceleration Response", ('aSteerWheel', 'gLat'), None , 0, None],
    ["Braking Efficiency", ('pBrakeF', 'gLong'), [(None,None),(-5,0)] , 0, None],
    ["Damper gLat front", ('gLat', 'xDamperDeltaF'), None , 1, None],
    ["Damper gLat rear", ('gLat', 'xDamperDeltaR'), None , 1, None],
    ["Pushrod gLat front", ('gLat', 'FProdDeltaF'), None , 1, None],
    ["Pushrod gLat rear", ('gLat', 'FProdDeltaR'), None , 1, None],
    ["Front Heave", ('xDamperAvgF', 'FProdAvgF'), None, 2, ('y', 10000)],
    ["Front Roll", ('xDamperDeltaF', 'FProdDeltaF'), None, 1, None],
    ["Rear Heave", ('xDamperAvgR', 'FProdAvgR'), None, 1, None],
    ["Rear Roll", ('xDamperDeltaR', 'FProdDeltaR'), None, 1, None],
    ["Front Pushrod vCar", ('vCar', 'FProdAvgF'), None, 1, None],
    ["Rear Pushrod vCar", ('vCar', 'FProdAvgR'), None, 1, None],
    ["Front Ride vCar", ('vCar', 'hRideF'), None, 1, None],
    ["Rear Ride vCar", ('vCar', 'hRideR'), None, 1, None],
    ["Ride Height Compare", ('hRideF', 'hRideR'), [(0, 40),(20, 70)], 0, None],
    ["Roll angle gLat", ('gLat', 'aRoll'), None, 1, None],
] 

PSD_PLOT_DEFINITIONS = [
   # ["Name of Plot", 'channel', [(xmin, xmax), (ymin, ymax)], nperseg(optional)]
    ["Front Vertical Acceleration PSD", 'gVertF', [(0, 50), (None, None)]],
    ["Rear Vertical Acceleration PSD", 'gVertR', [(0, 50), (None, None)]],
]

PLOT_DEFINITIONS = (WAVEFORM_PLOT_DEFINITIONS, SCATTER_PLOT_DEFINITIONS, PSD_PLOT_DEFINITIONS)

POWERPOINT_EXPORT_MAP = {
    4: {'layout': 'main_plot', 'images': ['waveform_Driver_Input.png']},
    5: {'layout': 'main_plot', 'images': ['waveform_Power_Unit.png']},
    6: {'layout': 'double_plot', 'images': ['scatter_Gear_Ratios.png', 'scatter_Engine_Power.png']},
    7: {'layout': 'double_plot', 'images': ['scatter_Long_Acceleration.png', 'scatter_Lat_Acceleration.png']},
    8: {'layout': 'double_plot', 'images': ['scatter_GG_Plot.png', 'scatter_Understeer_Plot.png']},
    9: {'layout': 'double_plot', 'images': ['scatter_Yaw_Rate_Response.png', 'scatter_Lateral_Acceleration_Response.png']},
    10: {'layout': 'main_plot', 'images': ['scatter_Braking_Efficiency.png']},
    11: {'layout': 'double_plot', 'images': ['scatter_Damper_gLat_front.png', 'scatter_Damper_gLat_rear.png']},
    12: {'layout': 'double_plot', 'images': ['scatter_Pushrod_gLat_front.png', 'scatter_Pushrod_gLat_rear.png']},
    13: {'layout': 'double_plot', 'images': ['scatter_Front_Heave.png', 'scatter_Rear_Heave.png']},
    14: {'layout': 'double_plot', 'images': ['scatter_Front_Roll.png', 'scatter_Rear_Roll.png']},
    15: {'layout': 'double_plot', 'images': ['scatter_Front_Pushrod_vCar.png', 'scatter_Rear_Pushrod_vCar.png']},
    16: {'layout': 'double_plot', 'images': ['scatter_Front_Ride_vCar.png', 'scatter_Rear_Ride_vCar.png']},
    17: {'layout': 'double_plot', 'images': ['scatter_Ride_Height_Compare.png', 'scatter_Roll_angle_gLat.png']},
    18: {'layout': 'double_plot', 'images': ['psd_Front_Vertical_Acceleration_PSD.png', 'psd_Rear_Vertical_Acceleration_PSD.png']},
    19: {'layout': 'main_plot', 'images': ['waveform_Plank_Wear.png']},
}

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    print("=" * 60)
    print("Correlation Scatterplots")
    print("=" * 60)
    
    plotter = DataPlotter(
        ROOT_FOLDER, 
        dls_run=DLS_RUN,
        track_run=TRACK_RUN,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS
    )
    plotter.plot_all()

    if EXPORT_TO_POWERPOINT:
        export_report_to_powerpoint(
            template_path=POWERPOINT_TEMPLATE,
            output_path=POWERPOINT_OUTPUT,
            plots_dir=plotter.plots_dir,
            export_map=POWERPOINT_EXPORT_MAP,
            visible=False
        )
    
