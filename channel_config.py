"""Project-wide configuration shared across all workflows.

Edit here to configure:
  - Data folder paths
  - Channel mappings, unit conversions, transforms
  - Calculated channels and filter settings
"""

from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid


# ─── PATHS ────────────────────────────────────────────────────────────────────
# All paths are resolved relative to this file's location (the project root).
# Only edit these if you move the Data/ folder.

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "Data"

CORRELATION_INPUT_DIR  = _DATA / "inputs"  / "correlation"
BOXPLOT_INPUT_DIR      = _DATA / "inputs"  / "boxplots"
DAMPER_INPUT_DIR       = _DATA / "inputs"  / "dampers"
TEMPLATES_DIR          = _DATA / "templates"

CORRELATION_OUTPUT_DIR = _DATA / "outputs" / "correlation"
BOXPLOT_OUTPUT_DIR     = _DATA / "outputs" / "boxplots"
DAMPER_PLOTS_DIR       = _DATA / "outputs" / "dampers"

# Create all directories on import so files can be dropped in immediately.
for _p in (
    CORRELATION_INPUT_DIR, BOXPLOT_INPUT_DIR, DAMPER_INPUT_DIR, TEMPLATES_DIR,
    CORRELATION_OUTPUT_DIR, BOXPLOT_OUTPUT_DIR, DAMPER_PLOTS_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)


def resolve_template_path(filename: str = "template.pptx") -> Path:
    return TEMPLATES_DIR / filename


def get_workflow_dirs(workflow: str, event: str = None) -> tuple:
    """Return (input_dir, output_dir) for any workflow name, creating folders if needed.

    If event is provided, directories are nested: Data/inputs/<workflow>/<event>/
    """
    if event:
        input_dir = _DATA / "inputs" / workflow / event
        output_dir = _DATA / "outputs" / workflow / event
    else:
        input_dir = _DATA / "inputs" / workflow
        output_dir = _DATA / "outputs" / workflow
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


# ─── CHANNEL MAPPINGS ────────────────────────────────────────────────────────
# Maps raw source column names to canonical names used in plot definitions.
# Add entries here when a data source uses non-standard column names.

CHANNEL_MAPPINGS = {
    "OC": {
        "rSLMActive": "SM",
        "aUndersteer_aSlip": "aUndersteerFromSlip",
        "dtLap_drGripFactorTotal": "Grip Sens.",
        "sRun": "sLap",
    },
    "DIL": {
        "BSLMActiveCan": "SM",
        # "FPushrodFL": "FPRodFL",
        # "FPushrodFR": "FPRodFR",
        # "FPushrodRL": "FPRodRL",
        # "FPushrodRR": "FPRodRR",
        "FPlankVertF": "FzPlankF",
        "EPlankWearLapF": "EPlankF",
        "PPlankWearF": "PPlankF",
        "pBrakeF1": "pBrakeF",
        "CAN_6_632_aSteerWheel_Can": "aSteerWheel",
    },
    "DLS": {
        "aRollCarTrack": "aRoll",
        # "FPushrodFL": "FPRodFL",
        # "FPushrodFR": "FPRodFR",
        # "FPushrodRL": "FPRodRL",
        # "FPushrodRR": "FPRodRR",
        "aUndersteer_aSlip": "aUndersteerFromSlip",
        "BAeroModeXDriver": "SM",
        "rThrottlePedal": "rThrottle",
        "EPlankLTS_Lap": "EPlankF",
        "PPlankWearF": "PPlankF",
        "zWheelCentreChassisFL": "xHubVertFL",
        "zWheelCentreChassisFR": "xHubVertFR",
        "zWheelCentreChassisRL": "xHubVertRL",
        "zWheelCentreChassisRR": "xHubVertRR",
    },
    "CAR": {
        "BNSLMEnablingStatusEnabled": "SM",
        "PMGUKActual": "PMGUK",
        "rThrottlePedal": "rThrottle",
        "xDamperPotFL": "xDamperFL",
        "xDamperPotFR": "xDamperFR",
        "xDamperPotRL": "xDamperRL",
        "xDamperPotRR": "xDamperRR",
        "nYawSlipSensor": "nYaw",
        "EPlankWearLapF": "EPlankF",
        "PPlankWearF": "PPlankF",
    },
}


# ─── UNITS MAP ────────────────────────────────────────────────────────────────
# Channel name (case-insensitive) → unit label shown on axes and legends.

UNITS_MAP = {
    "glat": "g",
    "glong": "g",
    "gvertf": "g",
    "gvertr": "g",
    "glat_abs": "g",
    "gLong (raw)": "g",
    "gVert": "g",
    "vcar": "kph",
    "aroll": "deg",
    "asteer": "deg",
    "asteerwheel": "deg",
    "aundersteerfromslip": "deg",
    "xrh": "mm",
    "laser": "mm",
    "hrider": "mm",
    "hridef": "mm",
    "damper": "mm",
    "xdamper": "mm",
    "fprod": "N",
    "fpushrod": "N",
    "pushrod": "N",
    "fprodfl": "N",
    "fprodfr": "N",
    "fprodrl": "N",
    "fprodrr": "N",
    "fprodavgf": "N",
    "fprodavgr": "N",
    "fproddeltaf": "N",
    "fproddeltar": "N",
    "trackrod": "N",
    "xdamperavgf": "mm",
    "xdamperavgr": "mm",
    "xdamperdeltaf": "mm",
    "xdamperdeltar": "mm",
    "nengine": "rpm",
    "mengine": "Nm",
    "msteerwheel": "Nm",
    "brake": "bar",
    "throttle": "%",
    "pmguk": "kW",
    "pengine": "kW",
    "nwheelr_avg": "rpm",
    "nyaw": "deg/s",
    "pbrakef": "bar",
    "rthrottle": "%",
    "EPlank_F": "kJ",
    "PPlank_F": "kW",
    "FzPlankF": "N",
    "PMGUK_Deploy (MJ)": "kW",
    "PMGUK_Charge (MJ)": "kW",
    "dmInjector": "kg/hr",
    "dmInjector (kg/s)": "",
}


# ─── CHANNEL TRANSFORMS ──────────────────────────────────────────────────────
# Numeric corrections applied per source type (sign flips, unit conversions, offsets).

CHANNEL_TRANSFORMS = {
    "DLS": {
        # "FPRodFL": lambda x: -x,
        # "FPRodFR": lambda x: -x,
        # "FPRodRL": lambda x: -x,
        # "FPRodRR": lambda x: -x,
        "aRoll":   lambda x: -x,
        "gVert":   lambda x: x - 1,
    },
    "CAR": {
        "sLap":  lambda x: x - 15,     # GPS alignment shift
    },
}


# ─── SCATTER / BAR RENDER SETTINGS ───────────────────────────────────────────

SCATTER_MAX_POINTS        = 45000    # Down-sample scatter above this count
BAR_SECONDARY_AXIS_RATIO  = 20.0     # Scale factor that triggers a secondary y-axis on bar plots


# ─── BOX PLOT VISUAL SETTINGS ────────────────────────────────────────────────

BOX_PLOT_SETTINGS = {
    "show_points":           True,
    "jitter":                0.15,
    "point_alpha":           0.25,
    "point_size":            18,
    "show_fliers":           False,
    "box_width":             0.65,
    "box_linewidth":         1.8,
    "box_edge_color":        "#4A4A4A",
    "medianline_color":      "#1A1A1A",
    "medianline_width":      2.5,
    "aggregated_box_color":  "#2E7D99",
    "aggregated_box_alpha":  0.75,
    "per_run_box_alpha":     0.75,
    "figsize_single_channel": (10, 6),
    "figsize_multi_channel":  (14, 10),
}


# ─── CALCULATED CHANNELS ─────────────────────────────────────────────────────
# Each entry is a lambda(df) that computes a new channel from existing ones.

CORRELATION_CALCULATED = {
    # Pushrod load differentials and averages
    "FPRodDeltaF":        lambda df: df["FPushrodFL"] - df["FPushrodFR"],
    "FPRodDeltaR":        lambda df: df["FPushrodRL"] - df["FPushrodRR"],
    "FPRodAvgF":          lambda df: (df["FPushrodFL"] + df["FPushrodFR"]) / 2,
    "FPRodAvgR":          lambda df: (df["FPushrodRL"] + df["FPushrodRR"]) / 2,
    # Damper travel differentials and averages
    "xDamperDeltaF":      lambda df: df["xDamperFL"] - df["xDamperFR"],
    "xDamperDeltaR":      lambda df: df["xDamperRL"] - df["xDamperRR"],
    "xDamperAvgF":        lambda df: (df["xDamperFL"] + df["xDamperFR"]) / 2,
    "xDamperAvgR":        lambda df: (df["xDamperRL"] + df["xDamperRR"]) / 2,
    # Lateral acceleration
    "gLat_Abs":           lambda df: df["gLat"].abs(),
    "gLong (raw)":        lambda df: df["gLong"],
    # Ride height (unfiltered copies)
    "hRideF (raw)":       lambda df: df["hRideF"],
    "hRideR (raw)":       lambda df: df["hRideR"],
    # Ride height (copies for high-pass filtering)
    "hRideF (high)":      lambda df: df["hRideF"],
    "hRideR (high)":      lambda df: df["hRideR"],

    # Power unit
    "PPUTotal":           lambda df: df["PMGUK"] + df["PEngine"],
    "nWheelAvg_R":        lambda df: (df["nWheelRL"] + df["nWheelRR"]) / 2,
    "dmInjector (kg/s)":  lambda df: df["dmInjector"] / 3600,
    "PMGUK_Deploy (MJ)":  lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] > 0).astype(float)).abs(),
    "PMGUK_Charge (MJ)":  lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] < 0).astype(float)).abs(),
    # Plank wear
    "PPlank_F":           lambda df: 0.001 * np.maximum(0.1 * df["FzPlankF"] * (df["vCar"] / 3.6), 0),
    "EPlank_F":           lambda df: cumulative_trapezoid(df["PPlank_F"], dx=0.01, initial=0),
    "tLap_Calc":          lambda df: cumulative_trapezoid(np.ones_like(df["vCar"]), dx=0.01, initial=0),

    # OC Only:
    "FzTyreF_Avg":      lambda df: (df["FzTyreFL"] + df["FzTyreFR"]) / 2,
    "xHubVertF_Avg":    lambda df: (df["xHubVertFL"] + df["xHubVertFR"]) / 2,
    "FzTyreR_Avg":      lambda df: (df["FzTyreRL"] + df["FzTyreRR"]) / 2,
    "xHubVertR_Avg":    lambda df: (df["xHubVertRL"] + df["xHubVertRR"]) / 2,

    "FzTyreF_Delta":    lambda df: df["FzTyreFL"] - df["FzTyreFR"],
    "xHubVertF_Delta":  lambda df: df["xHubVertFL"] - df["xHubVertFR"],
    "FzTyreR_Delta":    lambda df: df["FzTyreRL"] - df["FzTyreRR"],
    "xHubVertR_Delta":  lambda df: df["xHubVertRL"] - df["xHubVertRR"],
}

BOXPLOT_CALCULATED = {
    "dmInjector (kg/s)":    lambda df: df["dmInjector"] / 3600,
    "rLambda_avg (%)":      lambda df: 100 * (df["rLambdaL"] + df["rLambdaR"]) / 2,
    "dmExhaust":            lambda df: df["dmInjector"] * (1 + 13.23 * df["rLambda_avg (%)"] / 100) / 3600,
    "dmExhaust_Estimated":  lambda df: df["dmInjector"] * (1 + 13.23 * (0.155 * df["vCar"] + 109.329) / 100) / 3600,
    "Error_dmExhaust":      lambda df: df["dmExhaust"] - df["dmExhaust_Estimated"],
    "CosPhi_Calc":          lambda df: df["gLong"] / np.sqrt(df["gLat"] ** 2 + df["gLong"] ** 2),
}

DAMPER_CALCULATED = {
    "gLat_Abs": lambda df: np.abs(df["gLat"]),
}


# ─── FILTERS ──────────────────────────────────────────────────────────────────
# cutoff=0 disables filtering for that channel.
# "all" is a fallback applied to any channel not explicitly listed.
# Optional "type" key: "low" (default), "high", or "bandpass".
# For bandpass, cutoff is a two-element list [low_hz, high_hz].

# Base filter settings shared across workflows. Override per-workflow below.
_BASE_FILTERS = {
    "gVertF":        {"cutoff": 0,  "order": 2},
    "gVertR":        {"cutoff": 0,  "order": 2},
    "gVert":         {"cutoff": 0,  "order": 2},
    "PMGUK":         {"cutoff": 0,  "order": 2},
    "SM":            {"cutoff": 0,  "order": 2},
    "NGear":         {"cutoff": 0,  "order": 2},
    "nEngine":       {"cutoff": 0,  "order": 2},
    "rThrottle":     {"cutoff": 0,  "order": 2},
    "vCar":          {"cutoff": 0,  "order": 2},
    "dmInjector":    {"cutoff": 0,  "order": 2},
}

CORRELATION_FILTERS = {
    **_BASE_FILTERS,
    "FzPlankF":      {"cutoff": 0,  "order": 2},
    "nWheelAvg_R":   {"cutoff": 0,  "order": 2},
    "EPlank_F":      {"cutoff": 0,  "order": 2},
    "PPlank_F":      {"cutoff": 0,  "order": 2},
    "gLong (raw)":   {"cutoff": 0,  "order": 2},
    "hRideF (raw)":  {"cutoff": 0,  "order": 2},
    "hRideR (raw)":  {"cutoff": 0,  "order": 2},
    "hRideF (high)": {"cutoff": 0.5,  "order": 4, "type": "high"},
    "hRideR (high)": {"cutoff": 0.5,  "order": 4, "type": "high"},
    "PPUTotal":      {"cutoff": 0,  "order": 2},
    "all":           {"cutoff": 5,  "order": 2},
}

BOXPLOT_FILTERS = {
    **_BASE_FILTERS,
    "rLambdaL":    {"cutoff": 5, "order": 2},
    "rLambdaR":    {"cutoff": 5, "order": 2},
    "rLambda_avg": {"cutoff": 3, "order": 2},
    "dmInjector":  {"cutoff": 5, "order": 2},
    "dmFFMFuel":   {"cutoff": 5, "order": 2},
    "CosPhi":      {"cutoff": 5, "order": 2},
    "CosPhi_Calc": {"cutoff": 5, "order": 2},
    "gLong":       {"cutoff": 3, "order": 2},
    "gLat":        {"cutoff": 5, "order": 2},
}

DAMPER_FILTERS = {
    **_BASE_FILTERS,
    "CosPhi": {"cutoff": 3,  "order": 3},
    "rLLTD":  {"cutoff": 10, "order": 2},
    "gVert":  {"cutoff": 30, "order": 2},
    "gVertF": {"cutoff": 30, "order": 2},
    "gVertR": {"cutoff": 30, "order": 2},
    "all":    {"cutoff": 10, "order": 3},
}
