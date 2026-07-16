"""Project-wide configuration shared across all workflows.

Edit here to configure:
  - Data folder paths
  - Channel mappings, unit conversions, transforms
  - Calculated channels and filter settings
"""

from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from engine.datafunctions import calc_channel, calculate_cplv

# ─── PATHS ────────────────────────────────────────────────────────────────────
# All paths are resolved relative to this file's location (the project root).
# Only edit these if you move the Data/ folder.

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "Data"

CORRELATION_INPUT_DIR  = _DATA / "inputs"  / "correlation"
BOXPLOT_INPUT_DIR      = _DATA / "inputs"  / "boxplots"
DAMPER_INPUT_DIR       = _DATA / "inputs"  / "dampers"
RIDE_DIL_INPUT_DIR     = _DATA / "inputs"  / "ride_dil"
TEMPLATES_DIR          = _DATA / "templates"

CORRELATION_OUTPUT_DIR = _DATA / "outputs" / "correlation"
BOXPLOT_OUTPUT_DIR     = _DATA / "outputs" / "boxplots"
DAMPER_OUTPUT_DIR      = _DATA / "outputs" / "dampers"
RIDE_DIL_OUTPUT_DIR    = _DATA / "outputs" / "ride_dil"

# Create all directories on import so files can be dropped in immediately.
for _p in (
    CORRELATION_INPUT_DIR, BOXPLOT_INPUT_DIR, DAMPER_INPUT_DIR, RIDE_DIL_INPUT_DIR, TEMPLATES_DIR,
    CORRELATION_OUTPUT_DIR, BOXPLOT_OUTPUT_DIR, DAMPER_OUTPUT_DIR, RIDE_DIL_OUTPUT_DIR,
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
        "aUndersteer_gLat": "aUndersteerFromgLat",
        "aUndersteer_nYaw": "aUndersteerFromnYaw",
        "dtLap_drGripFactorTotal": "Grip Sens.",
        "sRun": "sLap",
        "aCamberKinematicFL": "aCamberFLKinematic",
        "aCamberKinematicFR": "aCamberFRKinematic",
        "aCamberKinematicRL": "aCamberRLKinematic",
        "aCamberKinematicRR": "aCamberRRKinematic",
    },
    "DIL": {
        "BSLMActiveCan": "SM",
        # "xDamperPotFL": "xDamperFL",
        # "xDamperPotFR": "xDamperFR",
        # "xDamperPotRL": "xDamperRL",
        # "xDamperPotRR": "xDamperRR",
        "FPlankVertF": "FzPlankF",
        "EPlankWearLapF": "EPlankF",
        "PPlankWearF": "PPlankF",
        "pBrakeF1": "pBrakeF",
        "CAN_6_632_aSteerWheel_Can": "aSteerWheel",
        "CAN_6_637_gVert_Can": "gVert",
        "sLapCan": "sLap",
    },
    "DLS": {
        "aRollCarTrack": "aRoll",
        "aUndersteer_aSlip": "aUndersteerFromSlip",
        "BAeroModeXDriver": "SM",
        "rThrottlePedal": "rThrottle",
        "EPlankLTS_Lap": "EPlankF",
        "PPlankWearF": "PPlankF",
        "zWheelCentreChassisFL": "xHubVertFL",
        "zWheelCentreChassisFR": "xHubVertFR",
        "zWheelCentreChassisRL": "xHubVertRL",
        "zWheelCentreChassisRR": "xHubVertRR",
        "vAero": "vAir",
    },
    "FMIOpt": {
        # Lap-sim (FMIOpt) parquet — behaves like DLS with a smaller channel
        # set. Any target that is not present in the source parquet is silently
        # skipped by apply_channel_mappings.
        "aRollCarTrack": "aRoll",
        "aUndersteer_aSlip": "aUndersteerFromSlip",
        "BAeroModeXDriver": "SM",
        "rThrottlePedal": "rThrottle",
        "EPlankLTS_Lap": "EPlankF",
        "PPlankWearF": "PPlankF",
        "zWheelCentreChassisFL": "xHubVertFL",
        "zWheelCentreChassisFR": "xHubVertFR",
        "zWheelCentreChassisRL": "xHubVertRL",
        "zWheelCentreChassisRR": "xHubVertRR",
        "vAero": "vAir",
    },
    "CAR": {
        "BNSLMEnablingStatusEnabled": "SM",
        "PMGUKActual": "PMGUK",
        "rThrottlePedal": "rThrottle",
        "xDamperPotFL": "xDamperFL",
        "xDamperPotFR": "xDamperFR",
        "xDamperPotRL": "xDamperRL",
        "xDamperPotRR": "xDamperRR",
        "nGyroYaw": "nYaw",
        "EPlankWearLapF": "EPlankF",
        "PPlankWearF": "PPlankF",
        "CLiftTotalF_Cp2CL": "CLiftTotalF",
        "CLiftTotalR_Cp2CL": "CLiftTotalR",
        "CLiftTotal_Cp2CL": "CLiftTotal",
        "rAerobalTotal_Cp2CL": "rAeroBal",
        "rBrakeBiasControl" : "rBrakeBias",
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
    "fprodheave":  "N",
    "fprodpitch":  "N",
    "fprodroll":   "N",
    "fprodwarp":   "N",
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
    "cplv_front": "N",
    "cplv_rear": "N",
    "EPlank_F": "kJ",
    "PPlank_F": "kW",
    "FzPlankF": "N",
    "PMGUK_Deploy (MJ)": "kW",
    "PMGUK_Charge (MJ)": "kW",
    "dmInjector": "kg/hr",
    "dmInjector (kg/s)": "",
    "tDiff": "s",
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
        "FPushrodFL": lambda x: -x,  # raw: tension-negative → flip to compression-positive
        "FPushrodFR": lambda x: -x,
        "FPushrodRL": lambda x: x,   # raw: already compression-positive
        "FPushrodRR": lambda x: x,
    },
    "CAR": {
        #"sLap":  lambda x: x - 10,     # GPS alignment shift
        "PBrakeFL": lambda x: -x,
        "PBrakeFR": lambda x: -x,
        "PBrakeRL": lambda x: -x,
        "PBrakeRR": lambda x: -x,
        "FPushrodFL": lambda x: -x,  # raw: tension-negative → flip to compression-positive
        "FPushrodFR": lambda x: -x,
        "FPushrodRL": lambda x: x,   # raw: already compression-positive
        "FPushrodRR": lambda x: x,
    },
    "DIL": {
        "PBrakeFL": lambda x: -x,
        "PBrakeFR": lambda x: -x,
        "PBrakeRL": lambda x: -x,
        "PBrakeRR": lambda x: -x,
        "FPushrodFL": lambda x: -x,  # raw: all tension-negative → flip to compression-positive
        "FPushrodFR": lambda x: -x,
        "FPushrodRL": lambda x: -x,
        "FPushrodRR": lambda x: -x,
    },
    "OC": {
        # "rBrakeBias": lambda x: x/100 ,  # Convert from % to 0-1
    },
    "FMIOpt": {
        # Lap-sim (FMIOpt): mirror DLS sign conventions. Missing channels
        # are silently skipped by apply_transformations.
        "aRoll":   lambda x: -x,
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
# Single comprehensive set of all derived channels. Each entry is a lambda(df)
# that computes a new column from existing ones. Channels whose dependencies
# are missing in a given run are silently skipped — no need to curate per-workflow.




CALCULATED_CHANNELS = {
    # ── Pushrod loads ────────────────────────────────────────────────────────
    "FPRodDeltaF":        lambda df: df["FPushrodFL"] - df["FPushrodFR"],
    "FPRodDeltaR":        lambda df: df["FPushrodRL"] - df["FPushrodRR"],
    "FPRodAvgF":          lambda df: (df["FPushrodFL"] + df["FPushrodFR"]) / 2,
    "FPRodAvgR":          lambda df: (df["FPushrodRL"] + df["FPushrodRR"]) / 2,
    # ── Ride modes (from corner pushrod forces) ──────────────────────────────
    "FPRodHeave":         lambda df: df["FPushrodFL"] + df["FPushrodFR"] + df["FPushrodRL"] + df["FPushrodRR"],
    "FPRodHeave_Shape":   lambda df: (1/5)*(df["FPushrodFL"] + df["FPushrodFR"]) + (4/5)*(df["FPushrodRL"] + df["FPushrodRR"]),
    "FPRodPitch":         lambda df: (df["FPushrodFL"] + df["FPushrodFR"]) - (df["FPushrodRL"] + df["FPushrodRR"]),
    "FPRodPitch_Shape":   lambda df: (5/9)*(df["FPushrodFL"] + df["FPushrodFR"]) - (4/9)*(df["FPushrodRL"] + df["FPushrodRR"]),
    "FPRodRoll":          lambda df: (df["FPushrodFR"] + df["FPushrodRR"]) - (df["FPushrodFL"] + df["FPushrodRL"]),
    "FPRodWarp":          lambda df: (df["FPushrodFL"] + df["FPushrodRR"]) - (df["FPushrodFR"] + df["FPushrodRL"]),
    "FProdVarFL":         lambda df: abs(df["FPushrodFL"] - df["FPushrodFL"].rolling(window=32, min_periods=1, center=True).mean()),
    "FProdVarFR":         lambda df: abs(df["FPushrodFR"] - df["FPushrodFR"].rolling(window=32, min_periods=1, center=True).mean()),
    "FProdVarRL":         lambda df: abs(df["FPushrodRL"] - df["FPushrodRL"].rolling(window=32, min_periods=1, center=True).mean()),
    "FProdVarRR":         lambda df: abs(df["FPushrodRR"] - df["FPushrodRR"].rolling(window=32, min_periods=1, center=True).mean()),
    "FPushrodFL_High":   lambda df: df["FPushrodFL"],
    "FPushrodFR_High":   lambda df: df["FPushrodFR"],
    "FPushrodRL_High":   lambda df: df["FPushrodRL"],
    "FPushrodRR_High":   lambda df: df["FPushrodRR"],
    # ── Damper travel ────────────────────────────────────────────────────────
    "xDamperDeltaF":      lambda df: df["xDamperFL"] - df["xDamperFR"],
    "xDamperDeltaR":      lambda df: df["xDamperRL"] - df["xDamperRR"],
    "xDamperAvgF":        lambda df: (df["xDamperFL"] + df["xDamperFR"]) / 2,
    "xDamperAvgR":        lambda df: (df["xDamperRL"] + df["xDamperRR"]) / 2,
    "vDamperDeltaF":      lambda df: np.gradient(df["xDamperFL"] - df["xDamperFR"], 1.0 / RESAMPLE_RATE, edge_order=2),
    "vDamperDeltaR":      lambda df: np.gradient(df["xDamperRL"] - df["xDamperRR"], 1.0 / RESAMPLE_RATE, edge_order=2),
    "vDamperAvgF":        lambda df: np.gradient((df["xDamperFL"] + df["xDamperFR"]) / 2, 1.0 / RESAMPLE_RATE, edge_order=2),
    "vDamperAvgR":        lambda df: np.gradient((df["xDamperRL"] + df["xDamperRR"]) / 2, 1.0 / RESAMPLE_RATE, edge_order=2),
    "xDamperVarFL":       lambda df: abs(df["xDamperFL"] - df["xDamperFL"].rolling(window=32, min_periods=1, center=True).mean()),
    "xDamperVarFR":       lambda df: abs(df["xDamperFR"] - df["xDamperFR"].rolling(window=32, min_periods=1, center=True).mean()),
    "xDamperVarRL":       lambda df: abs(df["xDamperRL"] - df["xDamperRL"].rolling(window=32, min_periods=1, center=True).mean()),
    "xDamperVarRR":       lambda df: abs(df["xDamperRR"] - df["xDamperRR"].rolling(window=32, min_periods=1, center=True).mean()),
    "xDamperFL_High":    lambda df: df["xDamperFL"],
    "xDamperFR_High":    lambda df: df["xDamperFR"],
    "xDamperRL_High":    lambda df: df["xDamperRL"],
    "xDamperRR_High":    lambda df: df["xDamperRR"],
    # ── Lateral / longitudinal acceleration ──────────────────────────────────
    "gLat_Abs":           lambda df: df["gLat"].abs(),
    "gLatAbs":            lambda df: df["gLat"].abs(),
    "gLong (raw)":        lambda df: df["gLong"],
    "CosPhi_Calc":        lambda df: df["gLong"] / np.sqrt(df["gLat"] ** 2 + df["gLong"] ** 2),
    # ── Ride height (unfiltered / high-pass copies) ──────────────────────────
    # Corner-average fallbacks — used only when the source lacks a native
    # axle-level hRideF/hRideR (e.g. FMIOpt lap-sim, which reports the four
    # corners hRideFL/FR/RL/RR only). Sources that already provide hRideF /
    # hRideR keep their native calibration.
    "hRideF":             calc_channel("hRideF", "hRideFL", "hRideFR")(lambda df: df["hRideF"] if "hRideF" in df.columns else (df["hRideFL"] + df["hRideFR"]) / 2),
    "hRideR":             calc_channel("hRideR", "hRideRL", "hRideRR")(lambda df: df["hRideR"] if "hRideR" in df.columns else (df["hRideRL"] + df["hRideRR"]) / 2),
    "hRideF (raw)":       lambda df: df["hRideF"],
    "hRideR (raw)":       lambda df: df["hRideR"],
    "hRideF (high)":      lambda df: df["hRideF"],
    "hRideR (high)":      lambda df: df["hRideR"],
    # ── Kinematic camber fallbacks ───────────────────────────────────────────
    # FMIOpt lap-sim outputs plain aCamberFL/FR/RL/RR (rigid-suspension model,
    # so kinematic == total). Sources that already carry an explicit
    # ...Kinematic channel are preserved.
    "aCamberFLKinematic": calc_channel("aCamberFLKinematic", "aCamberFL")(lambda df: df["aCamberFLKinematic"] if "aCamberFLKinematic" in df.columns else df["aCamberFL"]),
    "aCamberFRKinematic": calc_channel("aCamberFRKinematic", "aCamberFR")(lambda df: df["aCamberFRKinematic"] if "aCamberFRKinematic" in df.columns else df["aCamberFR"]),
    "aCamberRLKinematic": calc_channel("aCamberRLKinematic", "aCamberRL")(lambda df: df["aCamberRLKinematic"] if "aCamberRLKinematic" in df.columns else df["aCamberRL"]),
    "aCamberRRKinematic": calc_channel("aCamberRRKinematic", "aCamberRR")(lambda df: df["aCamberRRKinematic"] if "aCamberRRKinematic" in df.columns else df["aCamberRR"]),
    # ── Power unit ───────────────────────────────────────────────────────────
    "PPUTotal":           lambda df: df["PMGUK"] + df["PEngine"],
    "nWheelAvg_R":        lambda df: (df["nWheelRL"] + df["nWheelRR"]) / 2,
    "dmInjector (kg/s)":  lambda df: df["dmInjector"] / 3600,
    "PMGUK_Deploy (MJ)":  lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] > 0).astype(float)).abs(),
    "PMGUK_Charge (MJ)":  lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] < 0).astype(float)).abs(),
    # ── Plank wear ───────────────────────────────────────────────────────────
    "PPlank_F":           lambda df: 0.001 * np.maximum(0.1 * df["FzPlankF"] * (df["vCar"] / 3.6), 0) * (df["FzPlankF"] > 500).astype(float),
    "EPlank_F":           lambda df: cumulative_trapezoid(df["PPlank_F"], dx=0.01, initial=0),
    "tLap_Calc":          lambda df: cumulative_trapezoid(np.ones_like(df["vCar"]), dx=0.01, initial=0),
    # ── Tyre / suspension (OC sources) ───────────────────────────────────────
    "FzTyreF_Avg":        lambda df: (df["FzTyreFL"] + df["FzTyreFR"]) / 2,
    "xHubVertF_Avg":      lambda df: (df["xHubVertFL"] + df["xHubVertFR"]) / 2,
    "FzTyreR_Avg":        lambda df: (df["FzTyreRL"] + df["FzTyreRR"]) / 2,
    "xHubVertR_Avg":      lambda df: (df["xHubVertRL"] + df["xHubVertRR"]) / 2,
    "FzTyreF_Delta":      lambda df: df["FzTyreFL"] - df["FzTyreFR"],
    "xHubVertF_Delta":    lambda df: df["xHubVertFL"] - df["xHubVertFR"],
    "FzTyreR_Delta":      lambda df: df["FzTyreRL"] - df["FzTyreRR"],
    "xHubVertR_Delta":    lambda df: df["xHubVertRL"] - df["xHubVertRR"],
    "CPLV_Front":         calc_channel("FzTyreFL", "FzTyreFR", "FzTyreRL", "FzTyreRR")(
                              lambda df: calculate_cplv(df, "front", sample_rate=RESAMPLE_RATE)),
    "CPLV_Rear":          calc_channel("FzTyreFL", "FzTyreFR", "FzTyreRL", "FzTyreRR")(
                              lambda df: calculate_cplv(df, "rear", sample_rate=RESAMPLE_RATE)),
    # ── Aero ─────────────────────────────────────────────────────────────────
    "vWindHead":          lambda df: df["vAir"] - df["vCar"],
    "SC_CLT":             lambda df: df["CLiftTotal"] * ((df["vCar"] + df["vWindHead"]) / (df["vCar"])) ** 2,
    # ── Brake Powers ─────────────────────────
    "PBrakeF_Avg":       lambda df: (df["PBrakeFL"] + df["PBrakeFR"]) / 2,
    "PBrakeR_Avg":       lambda df: (df["PBrakeRL"] + df["PBrakeRR"]) / 2,
    "EBrakeFL":        lambda df: cumulative_trapezoid(abs(df["PBrakeFL"] / 1000), dx=0.01, initial=0),
    "EBrakeFR":        lambda df: cumulative_trapezoid(abs(df["PBrakeFR"] / 1000), dx=0.01, initial=0),
    "EBrakeRL":        lambda df: cumulative_trapezoid(abs(df["PBrakeRL"] / 1000), dx=0.01, initial=0),
    "EBrakeRR":        lambda df: cumulative_trapezoid(abs(df["PBrakeRR"] / 1000), dx=0.01, initial=0),
    # ─── SM Metrics ─────────────────────────
    "time_in_SM_100":       lambda df: cumulative_trapezoid((df["SM"] >= 0.999).astype(float), dx=0.01, initial=0),
    "time_in_SM_90":        lambda df: cumulative_trapezoid((df["SM"] >= 0.9).astype(float), dx=0.01, initial=0),
    "time_in_SM_80":        lambda df: cumulative_trapezoid((df["SM"] >= 0.8).astype(float), dx=0.01, initial=0),
    "ratio_time_in_SM_100": lambda df: cumulative_trapezoid((df["SM"] >= 0.999).astype(float), dx=0.01, initial=0) / (cumulative_trapezoid(np.ones_like(df["SM"]), dx=0.01, initial=0) + 1e-6),
    "ratio_time_in_SM_90":  lambda df: cumulative_trapezoid((df["SM"] >= 0.9).astype(float), dx=0.01, initial=0) / (cumulative_trapezoid(np.ones_like(df["SM"]), dx=0.01, initial=0) + 1e-6),
    "ratio_time_in_SM_80":  lambda df: cumulative_trapezoid((df["SM"] >= 0.8).astype(float), dx=0.01, initial=0) / (cumulative_trapezoid(np.ones_like(df["SM"]), dx=0.01, initial=0) + 1e-6),
}

# ─── RESAMPLING ───────────────────────────────────────────────────────────────
# All input channels are resampled to this uniform rate (Hz) BEFORE any
# filtering is applied. This guarantees filter cutoffs are consistent
# channel-to-channel and run-to-run regardless of the source logging rate.
# Set to 0 (or None) to disable resampling and use the native sample rate.
RESAMPLE_RATE = None


# ─── FILTERS ──────────────────────────────────────────────────────────────────
# Single filter dict used by ALL workflows. Channels not present in a given run
# are silently ignored — safe to include everything here.
#
# Format per entry:  "ChannelName": {"cutoff": Hz, "order": N, "type": "low"|"high"|"bandpass"}
#   - cutoff = 0  → no filtering (preserves raw signal)
#   - type defaults to "low" if omitted
#   - For bandpass, cutoff is a two-element list [low_hz, high_hz]
#   - "all" is the fallback applied to any channel NOT explicitly listed
#
# To override for a specific workflow, pass `filters={...}` to run_workflow().

FILTERS = {
    # ── Unfiltered channels (discrete / categorical / integral signals) ───────
    "SM":            {"cutoff": 0, "order": 2},
    "NGear":         {"cutoff": 0, "order": 2},
    "vCar":          {"cutoff": 0, "order": 2},
    "nEngine":       {"cutoff": 0, "order": 2},
    "rThrottle":     {"cutoff": 0, "order": 2},
    "PMGUK":         {"cutoff": 0, "order": 2},
    "PPUTotal":      {"cutoff": 0, "order": 2},
    "dmInjector":    {"cutoff": 0, "order": 2},

    # ── Vertical accelerations (raw for PSD, unfiltered) ──────────────────────
    "gVert":         {"cutoff": 0, "order": 2},
    "gVertF":        {"cutoff": 0, "order": 2},
    "gVertR":        {"cutoff": 0, "order": 2},
    "gHubVertFL":    {"cutoff": 0, "order": 2},
    "gHubVertFR":    {"cutoff": 0, "order": 2},
    "gHubVertRL":    {"cutoff": 0, "order": 2},
    "gHubVertRR":    {"cutoff": 0, "order": 2},

    # ── Pushrod forces (raw for PSD) ──────────────────────────────────────────
    "FPushrodFL":    {"cutoff": 0, "order": 2},
    "FPushrodFR":    {"cutoff": 0, "order": 2},
    "FPushrodRL":    {"cutoff": 0, "order": 2},
    "FPushrodRR":    {"cutoff": 0, "order": 2},

    # ── Ride modes (high-pass to remove static offset) ────────────────────────
    "FPRodHeave":    {"cutoff": (1.5, 15), "order": 4, "type": "bandpass"},
    "FPRodPitch":    {"cutoff": (1.5, 15), "order": 4, "type": "bandpass"},
    "FPRodRoll":     {"cutoff": (1.5, 15), "order": 4, "type": "bandpass"},
    "FPRodWarp":     {"cutoff": (1.5, 15), "order": 4, "type": "bandpass"},

    # ── Ride heights ──────────────────────────────────────────────────────────
    "hRideF (raw)":  {"cutoff": 0, "order": 2},
    "hRideR (raw)":  {"cutoff": 0, "order": 2},
    "hRideF (high)": {"cutoff": 0.5, "order": 4, "type": "high"},
    "hRideR (high)": {"cutoff": 0.5, "order": 4, "type": "high"},

    # ── Brakes ────────────────────────────────────────────────────────────────
    "PBrakeFL":      {"cutoff": 0, "order": 2},
    "PBrakeFR":      {"cutoff": 0, "order": 2},
    "PBrakeRL":      {"cutoff": 0, "order": 2},
    "PBrakeRR":      {"cutoff": 0, "order": 2},

    # ── Plank / energy channels ───────────────────────────────────────────────
    "FzPlankF":      {"cutoff": 0, "order": 2},
    "EPlank_F":      {"cutoff": 0, "order": 2},
    "PPlank_F":      {"cutoff": 0, "order": 2},
    "nWheelAvg_R":   {"cutoff": 0, "order": 2},

    # ── Misc unfiltered ───────────────────────────────────────────────────────
    "gLong (raw)":   {"cutoff": 0, "order": 2},
    "CPLV_Front":    {"cutoff": 0, "order": 2},
    "CPLV_Rear":     {"cutoff": 0, "order": 2},
    "tDiff":         {"cutoff": 0, "order": 2},
    "dtLap_dhCoGStatic":          {"cutoff": 0, "order": 2},
    "dtLap_dxCoGStatic":          {"cutoff": 0, "order": 2},
    "dtLap_dCDragTotal":          {"cutoff": 0, "order": 2},
    "dtLap_dCLiftTotal":          {"cutoff": 0, "order": 2},
    "dtLap_dhCoGStatic_Integral": {"cutoff": 0, "order": 2},
    "dtLap_dxCoGStatic_Integral": {"cutoff": 0, "order": 2},
    "dtLap_dCDragTotal_Integral": {"cutoff": 0, "order": 2},
    "dtLap_dCLiftTotal_Integral": {"cutoff": 0, "order": 2},

    # ── Damper velocities (no filter — derivative already limits bandwidth) ───
    "vDamperDeltaF": {"cutoff": 0, "order": 2},
    "vDamperDeltaR": {"cutoff": 0, "order": 2},
    "vDamperAvgF":   {"cutoff": 0, "order": 2},
    "vDamperAvgR":   {"cutoff": 0, "order": 2},

    # ── Damper / force variation channels (low-pass envelope) ─────────────────
    "xDamperVarFL":  {"cutoff": 2, "order": 2},
    "xDamperVarFR":  {"cutoff": 2, "order": 2},
    "xDamperVarRL":  {"cutoff": 2, "order": 2},
    "xDamperVarRR":  {"cutoff": 2, "order": 2},
    "FProdVarFL":    {"cutoff": 2, "order": 2},
    "FProdVarFR":    {"cutoff": 2, "order": 2},
    "FProdVarRL":    {"cutoff": 2, "order": 2},
    "FProdVarRR":    {"cutoff": 2, "order": 2},

    # ── High-pass copies (for PSD of AC content only) ─────────────────────────
    "FPushrodFL_High":      {"cutoff": 2, "order": 2, "type": "high"},
    "FPushrodFR_High":      {"cutoff": 2, "order": 2, "type": "high"},
    "FPushrodRL_High":      {"cutoff": 2, "order": 2, "type": "high"},
    "FPushrodRR_High":      {"cutoff": 2, "order": 2, "type": "high"},
    "xDamperFL_High":    {"cutoff": 2, "order": 2, "type": "high"},
    "xDamperFR_High":    {"cutoff": 2, "order": 2, "type": "high"},
    "xDamperRL_High":    {"cutoff": 2, "order": 2, "type": "high"},
    "xDamperRR_High":    {"cutoff": 2, "order": 2, "type": "high"},
    "FPRodHeave_Shape":  {"cutoff": 0, "order": 4},
    "FPRodPitch_Shape":  {"cutoff": 0, "order": 4},

    # ── Suspension / dynamics ─────────────────────────────────────────────────
    "rLLTD":         {"cutoff": 1, "order": 4},
    "CosPhi":        {"cutoff": 3, "order": 3},

    # ── Fallback for any channel not listed above ─────────────────────────────
    "all":           {"cutoff": 5, "order": 2},
}

# Backward-compatible aliases (referenced by legacy workflow names in plot_runtime).
DEFAULT_FILTERS      = FILTERS
CORRELATION_FILTERS  = FILTERS
BOXPLOT_FILTERS      = FILTERS
DAMPER_FILTERS       = FILTERS
RIDE_DIL_FILTERS     = FILTERS


TRACK_LENGTHS = {
    "BAH" : 5410.6,
    "MEL" : 5274.7,
    "SHA" : 5450.0,
    "SUZ" : 5806.1,
    "MIA" : 5409.2,
    "MTL" : 4364.4,
    "MCO" : 3335.8,
    "BCN" : 4657.2,
    "SPB" : 4309.6,
    "SIL" : 5888.6,
    "SPA" : 7000.2,
    "HUN" : 4377.0,
    "ZVT" : 4255.6,
    "MZA" : 5793.6,
    "MAD" : 5415.4,
    "BAK" : 5997.5,
    "SIN" : 4924.8,
    "COT" : 5510.3,
    "MEX" : 4301.8,
    "SAO" : 4299.9,
    "LAS" : 6200.2,
    "DOH" : 5417.3,
    "YAS" : 5281.4,
    "JED" : 6175.2,
}
