"""
Correlation Scatterplots - Multi-Run Data Plotter for Track and DiL Data
Allows flexible configuration of multiple runs, channel mappings, transformations, and plotting.
"""
import os
from pathlib import Path
from dataplotter import DataPlotter
from powerpointexporter import export_report_to_powerpoint, get_template_plot_aspect_ratios

# ======== CONFIGURATION ========

# Root folder containing data files
ROOT_FOLDER = r'C:\GitHub_Local\DLS-Correlation-Tool\Data'

DLS_RUN = {'name': 'DLS', 'file': 'BOT Q1R3 - OG_DLS_2.txt', 'color': "#0073FF"} #Plotted First

TRACK_RUN = {'name': 'CAR', 'file': '26R02SHA_260314_MAC26-02_BOT_Q_R02.txt', 'color': "#FF9100"} # Plotted Second (overlaid on top, swap order if necessary)

POWERPOINT_TEMPLATE = Path(ROOT_FOLDER) / 'template.pptx'
POWERPOINT_OUTPUT = Path(ROOT_FOLDER) / 'DLS_Correlation_Report.pptx'
EXPORT_TO_POWERPOINT = True

# ========= CHANNEL MAPPINGS ========

CHANNEL_MAPPINGS = {
    'dls': {
    # Example mappings - adjust based on your data:
    'aRollCarTrack': 'aRoll',
    'FPushrodFL': 'FProdFL',
    'FPushrodFR': 'FProdFR',
    'FPushrodRL': 'FProdRL',
    'FPushrodRR': 'FProdRR',
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
    'FPRodFL': 'FProdFL',
    'FPRodFR': 'FProdFR',
    'FPRodRL': 'FProdRL',
    'FPRodRR': 'FProdRR',
    'xDamperPotFL': 'xDamperFL',
    'xDamperPotFR': 'xDamperFR',
    'xDamperPotRL': 'xDamperRL',
    'xDamperPotRR': 'xDamperRR',
    'nYawSlipSensor': 'nYaw',
    'EPlankWearLapF': 'EPlankF',
    'PPlankWearF': 'PPlankF',
    }
}

CHANNEL_TRANSFORMS = {
    'dls': {
        'FProdFL': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'FProdFR': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'FProdRL': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'FProdRR': lambda x: -x, # DLS pushrod loads are negative, so invert to match track convention
        'aRoll': lambda x: -x, # DLS roll angle is opposite sign to track data, so invert
    },
    'car': {
        'PMGUK': lambda x: x/1000,  #convert from W to kW
        'sLap': lambda x: x - 10, # Shift distance to correct for GPS
    }
}

UNITS_MAP = {
    'glat': 'g', 'glong': 'g', 'gvertF': 'g', 'gvertR': 'g', 'glat_Abs': 'g', 'gLong (unsmoothed)': 'g',
    'vcar': 'kph',
    'aroll': '°', 'asteer': '°', 'asteerwheel': '°','aundersteerfromslip': '°',
    'xrh': 'mm', 'laser': 'mm', 'hrider': 'mm', 'hridef': 'mm',
    'damper': 'mm', 'xdamper': 'mm',
    'fprod': 'N', 'fpushrod': 'N', 'pushrod': 'N', 'fprodfl': 'N', 'fprodfr': 'N', 'fprodrL': 'N', 'fprodrR': 'N',
    'fprodavgf': 'N', 'fprodavgR': 'N', 'fproddeltaf': 'N', 'fproddeltaR': 'N','trackrod': 'N',
    'xdamperavgf': 'mm', 'xDamperAvgR': 'mm', 'xDamperDeltaF': 'mm', 'xDamperDeltaR': 'mm',
    'nengine': 'rpm',
    'mengine': 'Nm', 'msteerWheel': 'Nm',
    'brake': 'bar',
    'throttle': '%',
    'pmguk': 'kW',
    'pengine': 'kW',
    'nwheelr_avg': 'rpm',
    'nyaw': '°/s',
    'pbrakeF': 'bar',
    'rthrottle': '%',
    'EPlankF': 'kJ',
    'PPlankF': 'kW',
    'FzPlankF': 'N',
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
    'gLong (unsmoothed)': lambda df: df['gLong'], # Unsmoothed longitudinal acceleration (for comparison with smoothed version)
}

# ========= LOW PASS FILTERS ========

LOW_PASS_FILTERS = {
    'gVertF': {'cutoff': 0, 'order': 2},
    'gVertR': {'cutoff': 0, 'order': 2},
    'FzPlankF': {'cutoff': 0, 'order': 2},
    'PMGUK': {'cutoff': 0, 'order': 2},
    'SM': {'cutoff': 0, 'order': 2},
    'NGear': {'cutoff': 0, 'order': 2},
    'nEngine': {'cutoff': 0, 'order': 2},
    'nWheelR_Avg': {'cutoff': 0, 'order': 2},
    'EPlankF': {'cutoff': 0, 'order': 2},
    'PPlankF': {'cutoff': 0, 'order': 2},
    'all': {'cutoff': 6, 'order': 3}
}

# ======== PLOT DEFINITIONS ========

WAVEFORM_PLOT_DEFINITIONS = [
   # ["Name", (channels...), ((ymin,ymax)...), (reference lines...), (subplot height ratios...)]
   # `subplot height ratios` is optional; omit it to give every channel the same height.
    [
        "Driver Input", ('SM','NGear','vCar', 'PMGUK', 'aSteerWheel' , 'pBrakeF', 'rThrottle'),
        ((-0.2, 1.2), (1, 9), (60, 360), (-360, 360), (-160, 160), (0, 80), (-1, 101)), # y-axis limits for each channel
        (None, None, None, (-350, 0, 350), (0), None, None), # reference lines for each channel
        (0.1, 0.7, 1, 0.6,0.6, 0.35, 0.35) # subplot height ratios (optional)
    ],
    [
        "Power Unit", ('PMGUK', 'PEngine','NGear','vCar', 'nEngine', 'gLong' , 'pBrakeF', 'rThrottle'),
        ((-360, 360), (-100, 500), (1, 9), (60, 360), (7000, 13000), None, (0, 80), (0,101)), # y-axis limits for each channel
        ((-350, 0, 350), (0), None, None, (10000), (0), None, None), # reference lines for each channel
        (0.4, 0.4, 0.3, 0.7, 0.5, 0.5, 0.35, 0.35) # subplot height ratios (optional)
    ],
    [
        "Plank Wear", ('SM', 'PMGUK','vCar','FzPlankF', 'EPlankF' , 'pBrakeF', 'rThrottle'),
        ((-0.1, 1.1), (-351, 351), (60, 360), (0, 8000), (0, 100), (0, 80), (0,101)), # y-axis limits for each channel
        (None, (-350, 0, 350), (0,7500), (0), None, None), # reference lines for each channel
        (0.1, 0.6, 0.8, 0.7, 0.6, 0.35, 0.35) # subplot height ratios (optional)
    ],
] 

SCATTER_PLOT_DEFINITIONS = [
   #["Name of Plot", ('x Axis', 'y Axis'), [(xmin, xmax), (ymin, ymax)], Best Fit T/F, ('x', x_crossing) or ('y', y_crossing)],
    ["Gear Ratios", ('nWheelR_Avg', 'nEngine'), None, 0, None],
    ["Engine Power", ('nEngine', 'PEngine'), None, 0, None],
    ["Long Acceleration", ('vCar', 'gLong'), [(60,360),(None,None)], 0, None],
    ["Lat Acceleration", ('vCar', 'gLat_Abs'), [(60,360),(None,None)], 0, None],
    ["GG Plot", ('gLat', 'gLong'), None , 0, None],
    ["Understeer Plot", ('vCar', 'aUndersteerFromSlip'), None , 0, None],
    ["Yaw Rate Response", ('aSteerWheel', 'nYaw'), None , 0, None],
    ["Lateral Acceleration Response", ('aSteerWheel', 'gLat'), None , 0, None],
    ["Braking Efficiency", ('pBrakeF', 'gLong'), [(None,None),(-5,0)] , 2, ('y', -0.3)],
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
    ["Steering Moment", ('aSteerWheel', 'MSteerWheel'), None, 0, None],
    ["Plank power acceleration", ('gLong (unsmoothed)', 'PPlankF'), None, 0, None],
] 

PSD_PLOT_DEFINITIONS = [
   # ["Name of Plot", 'channel', [(xmin, xmax), (ymin, ymax)], nperseg(optional), log_scale(optional, default True)]
    ["Front Vertical Acceleration PSD", 'gVertF', [(0, 50), (None, None)], 256, False],
    ["Rear Vertical Acceleration PSD", 'gVertR', [(0, 50), (None, None)], 256, False],
    ["Plank Force PSD", 'FzPlankF', [(0, 50), (None, None)], 256, False],
]

PLOT_DEFINITIONS = (WAVEFORM_PLOT_DEFINITIONS, SCATTER_PLOT_DEFINITIONS, PSD_PLOT_DEFINITIONS)

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
    20: {'layout': 'double_plot', 'images': ['scatter_Plank_Power_Acceleration.png', 'psd_Plank_Force_PSD.png']},
}

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    print("=" * 60)
    print("Correlation Scatterplots")
    print("=" * 60)
    
    plot_aspect_ratios = get_template_plot_aspect_ratios(POWERPOINT_TEMPLATE, POWERPOINT_EXPORT_MAP)

    plotter = DataPlotter(
        ROOT_FOLDER, 
        dls_run=DLS_RUN,
        track_run=TRACK_RUN,
        plot_definitions=PLOT_DEFINITIONS,
        channel_mappings=CHANNEL_MAPPINGS,
        channel_transforms=CHANNEL_TRANSFORMS,
        calculated_channels=CALCULATED_CHANNELS,
        low_pass_filters=LOW_PASS_FILTERS,
        units_map = UNITS_MAP,
        plot_aspect_ratios=plot_aspect_ratios
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
        os.startfile(POWERPOINT_OUTPUT)

