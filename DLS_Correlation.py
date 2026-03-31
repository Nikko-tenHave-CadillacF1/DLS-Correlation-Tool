"""
Correlation Scatterplots - Multi-Run Data Plotter for Track and DiL Data
Allows flexible configuration of multiple runs, channel mappings, transformations, and plotting.
"""
from dataplotter import DataPlotter

# ======== CONFIGURATION ========

# Root folder containing data files
ROOT_FOLDER = r'C:\GitHub_Local\DLS_Correlation\Data'

DLS_RUN = {'name': 'DLS', 'file': '26R03SUZ  11  FP2  Run 1 1  Stint 1 1 1 FP2 R1 nC3 Q Sim_DLS.txt', 'color': '#0000FF'}

TRACK_RUN = {'name': 'Track', 'file': '26R03SUZ_260327_MAC26-02_BOT_P2_R03.txt', 'color': "#FF9100"}

# ========= CHANNEL MAPPINGS ========

CHANNEL_MAPPINGS = {
    'dls': {
    # Example mappings - adjust based on your data:
    'FPushrodFL': 'FProdFL',  # DLS has 'aRollCarTrack', Track has 'aRoll'
    'FPushrodFR': 'FProdFR',
    'FPushrodRL': 'FProdRL',
    'FPushrodRR': 'FProdRR',
    'aUndersteer_aSlip': 'aUndersteerFromaSlip',
    'BAeroModeXDriver': 'SM',
    'rThrottlePedal': 'rThrottle'
    },
    'track': {
    'BNSLMEnablingStatusEnabled': 'SM',
    'PMGUKActual': 'PMGUK',
    'rThrottlePedal': 'rThrottle'
    }
}

CHANNEL_TRANSFORMS = {
    'dls': {
        'FPushrodFL': lambda x: -x,
        'FPushrodFR': lambda x: -x,
    },
    'track': {
        'PMGUK': lambda x: x/1000,  #convert from W to kW
        'sLap': lambda x: x - 10, # Shift distance to correct for GPS
    }
}

# ========= LOW PASS FILTERS ========

LOW_PASS_FILTERS = {
    # Filter all pushrod channels at 5 Hz (individual corners)
    'FPushrodFL': {'cutoff': 4, 'order': 2},
    'FPushrodFR': {'cutoff': 4, 'order': 2},
    'FPushrodRL': {'cutoff': 4, 'order': 2},
    'FPushrodRR': {'cutoff': 4, 'order': 2},

    # Track pushrod loads (FPRod notation)
    'FPRodFL': {'cutoff': 4, 'order': 2},
    'FPRodFR': {'cutoff': 4, 'order': 2},
    'FPRodRL': {'cutoff': 4, 'order': 2},
    'FPRodRR': {'cutoff': 4, 'order': 2},
    
    # Damper pot channels at 5 Hz (individual corners)
    'xDamperPotFL': {'cutoff': 4, 'order': 2},
    'xDamperPotFR': {'cutoff': 4, 'order': 2},
    'xDamperPotRL': {'cutoff': 4, 'order': 2},
    'xDamperPotRR': {'cutoff': 4, 'order': 2},
    
    # Ride height and laser channels at 5 Hz
    'xRHLaserF': {'cutoff': 4, 'order': 2},
    'xRHRollLaserL': {'cutoff': 4, 'order': 2},
    'xRHRollLaserR': {'cutoff': 4, 'order': 2},
    'hRideF': {'cutoff': 4, 'order': 2},
    'hRideR': {'cutoff': 4, 'order': 2},
    
    # Roll angle and lateral g at 5 Hz
    'aRoll': {'cutoff': 4, 'order': 2},
    'gLat': {'cutoff': 4, 'order': 2},
    
    # Engine channels - different cutoffs for dls (5 Hz) vs Track (2 Hz)
    'MEngine': {'cutoff': 5, 'order': 2},
    'PEngine': {'cutoff': 5, 'order': 2},
    'PMGUK': {'cutoff': 5, 'order': 2}
}


# ======== PLOT DEFINITIONS ========

WAVEFORM_PLOT_DEFINITIONS = [
   # ["Name", (channels...), ((ymin,ymax)...), (reference lines...), (subplot height ratios...)]
   # `subplot height ratios` is optional; omit it to give every channel the same height.
    [
        "Driver Input", ('SM','NGear','vCar', 'PMGUK', 'pBrakeF', 'rThrottle'),
                     ((-0.1, 1.1), (1, 9), (60, 360), (-351, 351), (0, 80), (-1, 101)), # y-axis limits for each channel
                     (None, None, None, (-350, 0, 350), 100, None), # reference lines for each channel
                     (0.2, 0.5, 1.4, 0.8, 0.5, 0.5) # subplot height ratios (optional)
    ],
] 

SCATTER_PLOT_DEFINITIONS = [
   #["Name of Plot", ('x Axis', 'y Axis'), [(xmin, ymin), (xmax, ymax)], Best Fit T/F, X crossing],
    ["Roll Angle1", ('gLat', 'aRoll'), None, 0, None],
    ["Roll Angle2", ('gLat', 'aRoll'), None, 1, None],
    ["Roll Angle3", ('gLat', 'aRoll'), None, 2, 0.1],
] 

PLOT_DEFINITIONS = (WAVEFORM_PLOT_DEFINITIONS, SCATTER_PLOT_DEFINITIONS)

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    print("=" * 60)
    print("Correlation Scatterplots")
    print("=" * 60)
    
    plotter = DataPlotter(ROOT_FOLDER, dls_run=DLS_RUN, track_run=TRACK_RUN, plot_definitions=PLOT_DEFINITIONS, channel_mappings=CHANNEL_MAPPINGS, channel_transforms=CHANNEL_TRANSFORMS, low_pass_filters=LOW_PASS_FILTERS) 
    plotter.plot_all()
    
