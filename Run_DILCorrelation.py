"""DIL correlation workflow — edit RUNS and plot definitions below."""

from channel_config import get_workflow_dirs
from engine import (
    BarPlot,
    PsdPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    run_workflow,
)

WORKFLOW_NAME = "correlation"
EVENT = "26R15BAK"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

NPERSEG = 120  # Welch segment length (samples) shared by every PsdPlot.


# ─── RUNS ─────────────────────────────────────────────────────────────────────
# Supported "type" values: "CAR", "OC", "DIL", "DLS", "FMIOpt".
# FMIOpt = LapSim/AVL-TR parquet (handled like DLS; see channel_config.py).

RUNS = [
    # {
    #     "name": "CAR - BLUE",
    #     "file": r"26R15BAK_260924_MAC26-01_BOT_P2_R02PARTIAL.txt",
    #     "color": "#0900B9",
    #     "type": "CAR",
    # },
    # {
    #     "name": "DLS - BLUE",
    #     "file": r"BOT FP2R2_-CORR_DLS.parquet",
    #    "color": "#00A830",
    #     "nlap": 1,
    #     "type": "DLS",
    # },
    {
        "name": "CAR - RED",
        "file": r"26R15BAK_260924_MAC26-02_PER_P2_R02PARTIAL.txt",
        "color": "#B90C00",
        "type": "CAR"
    },
    {
        "name": "DLS - FIT R22",
        "file": r"FIT BAK R22_-BSL_DLS.parquet",
        "color": "#00A149",
        "nlap": 1,
        "type": "DLS",
    },
    {
        "name": "DLS -50 CLF +30 CLR FIT R22",
        "file": r"FIT BAK R22_-50 CLF + 30 CLR_DLS.parquet",
        "color": "#0800A1",
        "nlap": 1,
        "type": "DLS",
    },
]


# ─── POWERPOINT EXPORT ────────────────────────────────────────────────────────
# Set POWERPOINT_OUTPUT = None to disable the .pptx export.
POWERPOINT_OUTPUT = _OUTPUT_DIR / "DIL_Offline_Checks.pptx"


# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────

WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="Driver Input",
        channels=('PMGUK', ('vCar', 'NGear'), 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, ((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0,), None, None),
        subplot_heights=(0.4, 0.7, 0.3, 0.3, 0.3),
        show_delta=(False, True, False, False, False),
    ),
    WaveformPlot(
        name="Power Unit",
        channels=('PMGUK', 'PEngine', ('vCar', 'NGear'), 'nEngine', 'dmInjector', ('rThrottle', 'SM')),
        axis_limits=(None, None, ((None, 400), (-1, 9)), None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), None, (10000,), None, None),
        subplot_heights=(0.4, 0.4, 0.6, 0.4, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Plank Wear",
        channels=('PMGUK', 'vCar', ('FzPlankF', 'FzPlankR'), 'EPlank_F', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), None, (0, 7500), None, (0, 100), None),
        subplot_heights=(0.4, 0.6, 0.4, 0.6, 0.4, 0.4),
    ),
    WaveformPlot(
        name="Ride Heights Waveform",
        channels=(('vCar', 'NGear'), 'hRideF', 'hRideR', 'aRoll', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, (0,), (0,), (0,), None, None),
        subplot_heights=(0.8, 0.8, 0.8, 0.5, 0.5, 0.5),
    ),
    WaveformPlot(
        name="Handling Metrics",
        channels=('PMGUK', 'aUndersteerFromSlip', 'aSteerWheel', 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=((-350, 0, 350), (0,), (0,), None, None),
        subplot_heights=(0.4, 0.4, 0.3, 0.3, 0.3),
    ),
    WaveformPlot(
        name="Yaw & Lateral Response",
        channels=(('vCar', 'NGear'), 'aSteerWheel', 'nYaw', ('gLat_Abs', 'gCombined'), 'pBrakeF', ('rThrottle', 'SM')),
        axis_limits=(((None, 400), (-1, 9)), None, None, None, None, ((0, 105), (0, 1.3))),
        reference_lines=(None, (0,), (0,), None, None, None),
        subplot_heights=(0.7, 0.3, 0.3, 0.4, 0.3, 0.3),
    ),
]


# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────

SCATTER_PLOT_DEFINITIONS = [
    # ── Powertrain ───────────────────────────────────────────────────────────
    ScatterPlot("Gear Ratios", "nWheelAvg_R", "nEngine",
                best_fit=[('NGear', g - 0.5, g + 0.5) for g in range(2, 9)],
                show_equations=False, error_as_factor=True),
    ScatterPlot("Engine Power",      "nEngine",    "PEngine"),
    ScatterPlot("Engine Efficiency", "dmInjector", "PEngine", best_fit=1, error_as_factor=True),

    # ── Vehicle dynamics ─────────────────────────────────────────────────────
    ScatterPlot("Long Acceleration", "vCar", "gLong"),
    ScatterPlot("Lat Acceleration",  "vCar", "gLat_Abs"),
    ScatterPlot("GG Plot",           "gLat", "gLong"),
    ScatterPlot("Braking Efficiency", "pBrakeF", "gLong",
                best_fit=[('y', None, -0.2)], gate=('gLong', '<', 0), error_as_factor=True),

    # ── Handling / steering ──────────────────────────────────────────────────
    ScatterPlot("Understeer Plot", "vCar", "aUndersteerFromSlip",
                axis_limits=[(None, None), (-3, 3)]),
    ScatterPlot("Yaw Rate Response", "aSteerWheel", "nYaw",
                axis_limits=[(-160, 160), (None, None)],
                best_fit=[('x', -20, 20)], error_as_factor=True),
    ScatterPlot("Lateral Acceleration Response", "aSteerWheel", "gLat",
                axis_limits=[(-160, 160), (None, None)],
                best_fit=[('x', -20, 20)], error_as_factor=True),
    ScatterPlot("Steering Moment", "aSteerWheel", "MSteerWheel",
                axis_limits=[(-160, 160), (None, None)]),

    # ── Absolute suspension offsets — used to align DIL setup with CAR ──────
    ScatterPlot("Front Heave", "xDamperAvgF", "FPRodAvgF",
                best_fit=[('x', 105, None)], error_as_factor=True),
    ScatterPlot("Front Roll",  "xDamperDeltaF", "FPRodDeltaF",
                best_fit=[('x', None, None)], error_as_factor=True),
    ScatterPlot("Rear Heave",  "xDamperAvgR", "FPRodAvgR",
                best_fit=[('x', None, 145), ('x', 146, None)], error_as_factor=True),
    ScatterPlot("Rear Roll",   "xDamperDeltaR", "FPRodDeltaR",
                best_fit=[('x', None, None)], error_as_factor=True),
    ScatterPlot("Roll angle gLat", "gLat", "aRoll",
                best_fit=[('x', None, None)], error_as_factor=True),

    # ── Aero / ride vs vCar (SM-split fits) ──────────────────────────────────
    ScatterPlot("Front Pushrod vCar", "vCar", "FPRodAvgF",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                error_as_factor=True, gate=(('pBrakeF', '<', 1), ('gLat_Abs', '<', 1))),
    ScatterPlot("Rear Pushrod vCar", "vCar", "FPRodAvgR",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                error_as_factor=True, gate=(('pBrakeF', '<', 1), ('gLat_Abs', '<', 1))),
    ScatterPlot("Front Ride vCar", "vCar", "hRideF",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                axis_limits=[(None, None), (None, 40)],
                error_as_factor=True, dil_friendly=True),
    ScatterPlot("Rear Ride vCar", "vCar", "hRideR",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                axis_limits=[(None, None), (None, 75)],
                error_as_factor=True, dil_friendly=True),
    ScatterPlot("Front Ride vCar - SM OFF", "vCar", "hRideF",
                best_fit=[('SM', 0, 0.5)], annotate_fit_at=(100, 200, 300),
                axis_limits=[(None, None), (None, 40)],
                error_as_factor=True, dil_friendly=True),
    ScatterPlot("Ride Height Compare", "hRideF", "hRideR"),

    # ── CL proxies (F_pushrod / vCar² plateau ≈ per-axle CL offset) ─────────
    ScatterPlot("Front CL Proxy vCar - SM OFF", "vCar", "CLF_Proxy",
                best_fit=[('x', 150, None)], annotate_fit_at=(150, 200, 250, 300),
                error_as_factor=True,
                gate=(('pBrakeF', '<', 1), ('gLat_Abs', '<', 1), ('SM', '<', 0.5))),
    ScatterPlot("Rear CL Proxy vCar - SM OFF", "vCar", "CLR_Proxy",
                best_fit=[('x', 150, None)], annotate_fit_at=(150, 200, 250, 300),
                error_as_factor=True,
                gate=(('pBrakeF', '<', 1), ('gLat_Abs', '<', 1), ('SM', '<', 0.5))),
    ScatterPlot("Front CL Proxy vCar - SM ON", "vCar", "CLF_Proxy",
                best_fit=[('x', 150, None)], annotate_fit_at=(150, 200, 250, 300),
                error_as_factor=True,
                gate=(('pBrakeF', '<', 1), ('gLat_Abs', '<', 1), ('SM', '>', 0.5))),
    ScatterPlot("Rear CL Proxy vCar - SM ON", "vCar", "CLR_Proxy",
                best_fit=[('x', 150, None)], annotate_fit_at=(150, 200, 250, 300),
                error_as_factor=True,
                gate=(('pBrakeF', '<', 1), ('gLat_Abs', '<', 1), ('SM', '>', 0.5))),

    # ── Raw laser ride heights (fallback when calibrated hRide* missing) ────
    ScatterPlot("Front Laser vCar",      "vCar", "xRHLaserF",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                error_as_factor=True),
    ScatterPlot("Rear Laser Left vCar",  "vCar", "xRHRollLaserL",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                error_as_factor=True),
    ScatterPlot("Rear Laser Right vCar", "vCar", "xRHRollLaserR",
                best_fit=[('SM', 0, 0.5), ('SM', 0.5, 1)], annotate_fit_at=(100, 200, 300),
                error_as_factor=True),
]


# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
# `lorentz_fit=(f0_lo, f0_hi)` fits a Lorentzian peak in the given Hz range;
# use a list of tuples for multi-peak fits (e.g. Roll = body + wheel-hop).

PSD_PLOT_DEFINITIONS = [
    # ── Ride-mode PSDs (linear) ──────────────────────────────────────────────
    PsdPlot("Heave Mode PSD - abs", "FPRodHeave", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(3, 7)),
    PsdPlot("Pitch Mode PSD - abs", "FPRodPitch", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(7, 11)),
    PsdPlot("Roll Mode PSD - abs",  "FPRodRoll",  nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)]),
    PsdPlot("Warp Mode PSD - abs",  "FPRodWarp",  nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(10, 15)),

    # ── Corner pushrod PSDs (linear) ─────────────────────────────────────────
    PsdPlot("FPushrodFL PSD", "FPushrodFL_High", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(4, 7)),
    PsdPlot("FPushrodFR PSD", "FPushrodFR_High", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(4, 7)),
    PsdPlot("FPushrodRL PSD", "FPushrodRL_High", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)]),
    PsdPlot("FPushrodRR PSD", "FPushrodRR_High", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)]),

    # ── Chassis vertical acceleration PSDs (log + linear) ────────────────────
    PsdPlot("Front Vertical Acceleration PSD",       "gVertF", nperseg=NPERSEG,
            axis_limits=[(0, 20), (None, None)], lorentz_fit=(5, 11)),
    PsdPlot("Rear Vertical Acceleration PSD",        "gVertR", nperseg=NPERSEG,
            axis_limits=[(0, 20), (None, None)], lorentz_fit=(5, 11)),
    PsdPlot("Front Vertical Acceleration PSD - ABS", "gVertF", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (None, None)], lorentz_fit=(5, 11)),
    PsdPlot("Rear Vertical Acceleration PSD - ABS",  "gVertR", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (None, None)], lorentz_fit=(5, 11)),

    # ── Ride-height PSDs (log + linear) ──────────────────────────────────────
    PsdPlot("hRideF PSD",       "hRideF (raw)",  nperseg=NPERSEG,
            axis_limits=[(0, 20), (None, None)],   lorentz_fit=(3, 8)),
    PsdPlot("hRideR PSD",       "hRideR (raw)",  nperseg=NPERSEG,
            axis_limits=[(0, 20), (None, None)],   lorentz_fit=(4, 9)),
    PsdPlot("hRideF PSD - abs", "hRideF (high)", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e-4, None)],   lorentz_fit=(3, 8)),
    PsdPlot("hRideR PSD - abs", "hRideR (high)", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e-4, None)],   lorentz_fit=(4, 9)),

    # ── Front/Rear pushrod average PSDs (linear) ─────────────────────────────
    PsdPlot("FPRodAvgF - PSD", "FPRodAvgF_High", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(3, 7)),
    PsdPlot("FPRodAvgR - PSD", "FPRodAvgR_High", nperseg=NPERSEG, log_scale=False,
            axis_limits=[(0, 20), (1e4, None)], lorentz_fit=(3, 7)),
]


# ─── BAR PLOTS ────────────────────────────────────────────────────────────────

BAR_PLOT_DEFINITIONS = [
    BarPlot("Cumulative Metrics",
            (("dmInjector (kg/s)",  "integral"),
             ("PMGUK_Deploy (MJ)",  "integral"),
             ("PMGUK_Charge (MJ)",  "integral"))),
    BarPlot("Lap Time",
            (("tLap_Calc",     "max"),
             ("tGripLimited",  "integral"),
             ("tPowerLimited", "integral")),
            show_delta=True),
]


# ─── POWERPOINT EXPORT MAP ────────────────────────────────────────────────────
# Slide("main_plot",   "type/Plot Name")                       — full-width image
# Slide("double_plot", "type/Left Plot", "type/Right Plot")    — two side-by-side

POWERPOINT_EXPORT_MAP = [
    # 1. Driver / power unit
    Slide("main_plot",   "waveform/Driver Input"),
    Slide("main_plot",   "waveform/Power Unit"),
    Slide("double_plot", "scatter/Gear Ratios",         "scatter/Engine Power"),
    Slide("double_plot", "scatter/Engine Efficiency",   "bar/Cumulative Metrics"),

    # 2. Vehicle dynamics — g-forces
    Slide("double_plot", "scatter/Long Acceleration",   "scatter/Lat Acceleration"),
    Slide("double_plot", "scatter/GG Plot",             "bar/Lap Time"),

    # 3. Handling & steering
    Slide("main_plot",   "waveform/Handling Metrics"),
    Slide("main_plot",   "waveform/Yaw & Lateral Response"),
    Slide("double_plot", "scatter/Understeer Plot",     "scatter/Steering Moment"),
    Slide("double_plot", "scatter/Yaw Rate Response",   "scatter/Lateral Acceleration Response"),

    # 4. Brakes
    Slide("main_plot",   "scatter/Braking Efficiency"),

    # 5. Suspension loads & modes
    Slide("double_plot", "scatter/Front Heave",         "scatter/Rear Heave"),
    Slide("double_plot", "scatter/Front Roll",          "scatter/Rear Roll"),
    Slide("double_plot", "scatter/Front Pushrod vCar",  "scatter/Rear Pushrod vCar"),

    # 6. Ride height & chassis attitude
    Slide("main_plot",   "waveform/Ride Heights Waveform"),
    Slide("double_plot", "scatter/Front Ride vCar",     "scatter/Rear Ride vCar"),
    Slide("double_plot", "scatter/Ride Height Compare", "scatter/Roll angle gLat"),

    # 7. Plank wear
    Slide("main_plot",   "waveform/Plank Wear"),

    # 8. Ride-mode PSDs
    Slide("double_plot", "psd/Heave Mode PSD - abs",    "psd/Pitch Mode PSD - abs"),
    Slide("double_plot", "psd/Roll Mode PSD - abs",     "psd/Warp Mode PSD - abs"),

    # 9. Chassis vertical PSDs
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Rear Vertical Acceleration PSD"),

    # 10. Ride-height PSDs
    Slide("double_plot", "psd/hRideF PSD",              "psd/hRideR PSD"),
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
        bars=BAR_PLOT_DEFINITIONS,
        powerpoint_output=POWERPOINT_OUTPUT,
        export_map=POWERPOINT_EXPORT_MAP,
    )


if __name__ == "__main__":
    main()
