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
# Paths are anchored to this file's location; edit only if you move Data/.
_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "Data"

CORRELATION_INPUT_DIR = _DATA / "inputs" / "correlation"
BOXPLOT_INPUT_DIR     = _DATA / "inputs" / "boxplots"
DAMPER_INPUT_DIR      = _DATA / "inputs" / "dampers"
RIDE_DIL_INPUT_DIR    = _DATA / "inputs" / "ride_dil"
TEMPLATES_DIR         = _DATA / "templates"

CORRELATION_OUTPUT_DIR = _DATA / "outputs" / "correlation"
BOXPLOT_OUTPUT_DIR     = _DATA / "outputs" / "boxplots"
DAMPER_OUTPUT_DIR      = _DATA / "outputs" / "dampers"
RIDE_DIL_OUTPUT_DIR    = _DATA / "outputs" / "ride_dil"

for _p in (
    CORRELATION_INPUT_DIR, BOXPLOT_INPUT_DIR, DAMPER_INPUT_DIR, RIDE_DIL_INPUT_DIR,
    TEMPLATES_DIR, CORRELATION_OUTPUT_DIR, BOXPLOT_OUTPUT_DIR, DAMPER_OUTPUT_DIR,
    RIDE_DIL_OUTPUT_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)


def resolve_template_path(filename: str = "template.pptx") -> Path:
    return TEMPLATES_DIR / filename


def get_workflow_dirs(workflow: str, event: str = None) -> tuple:
    """Return (input_dir, output_dir) for any workflow, nested by event if given."""
    parts = (workflow, event) if event else (workflow,)
    input_dir = _DATA.joinpath("inputs", *parts)
    output_dir = _DATA.joinpath("outputs", *parts)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


# ─── RESAMPLING ───────────────────────────────────────────────────────────────
# All channels resampled to this rate (Hz) BEFORE filtering, so filter cutoffs
# stay consistent regardless of the source logging rate. 0/None disables it.
RESAMPLE_RATE = 50


# ─── CHANNEL MAPPINGS ────────────────────────────────────────────────────────
# Raw source column → canonical name used in plot definitions. Add entries
# when a data source uses non-standard column names.
CHANNEL_MAPPINGS = {
    "OC": {
        "rSLMActive": "SM",
        "aUndersteer_aSlip": "aUndersteerFromSlip",
        "aUndersteer_gLat":  "aUndersteerFromgLat",
        "aUndersteer_nYaw":  "aUndersteerFromnYaw",
        "dtLap_drGripFactorTotal": "Grip Sens.",
        "sRun": "sLap",
        # aCamber* is intentionally NOT mapped: CAR's `aCamber*Kinematic` is
        # the TOTAL quantity, and the calc-channel fallback below rebuilds
        # OC's kinematic value as (aCamber − aCamberComplianceDelta).
    },
    "DIL": {
        "BSLMActiveCan": "SM",
        "FPlankVertF": "FzPlankF",
        "FPlankVertR": "FzPlankR",
        "EPlankWearLapF": "EPlankF",
        "PPlankWearF": "PPlankF",
        "CAN_6_622_pBrakeF_Can": "pBrakeF",
        "CAN_6_632_aSteerWheel_Can": "aSteerWheel",
        "CAN_6_637_gVert_Can": "gVert",
        "sLapCan": "sLap",
        "PMGUKActual": "PMGUK",
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
        "CLiftTotal_Cp2CL":  "CLiftTotal",
        "rAerobalTotal_Cp2CL": "rAeroBal",
        "rBrakeBiasControl": "rBrakeBias",
        "EESSOCDelta": "rSOCDelta",
    },
}
# FMIOpt (lap-sim parquet) mirrors DLS; unmapped channels are silently skipped.
CHANNEL_MAPPINGS["FMIOpt"] = dict(CHANNEL_MAPPINGS["DLS"])


# ─── UNITS MAP ────────────────────────────────────────────────────────────────
# Channel name (case-insensitive) → unit label shown on axes and legends.
UNITS_MAP = {
    # accelerations (g)
    "glat": "g", "glong": "g", "gvertf": "g", "gvertr": "g", "glat_abs": "g",
    "gLong (raw)": "g", "gCombined": "g", "gVert": "g",
    # speeds / rates
    "vcar": "kph", "nengine": "rpm", "nwheelr_avg": "rpm", "nyaw": "deg/s",
    # angles (deg)
    "aroll": "deg", "asteer": "deg", "asteerwheel": "deg",
    "aundersteerfromslip": "deg",
    # displacements (mm)
    "xrh": "mm", "laser": "mm", "hrider": "mm", "hridef": "mm",
    "damper": "mm", "xdamper": "mm",
    "xdamperavgf": "mm", "xdamperavgr": "mm",
    "xdamperdeltaf": "mm", "xdamperdeltar": "mm",
    # forces (N)
    "fprod": "N", "fpushrod": "N", "pushrod": "N", "trackrod": "N",
    "fprodfl": "N", "fprodfr": "N", "fprodrl": "N", "fprodrr": "N",
    "fprodavgf": "N", "fprodavgr": "N", "fproddeltaf": "N", "fproddeltar": "N",
    "fprodheave": "N", "fprodpitch": "N", "fprodroll": "N", "fprodwarp": "N",
    "cplv_front": "N", "cplv_rear": "N",
    "FzPlankF": "N", "FzPlankF (smooth)": "N",
    "FzPlankR": "N", "FzPlankR (smooth)": "N",
    # torque
    "mengine": "Nm", "msteerwheel": "Nm",
    # pressures / percent
    "brake": "bar", "pbrakef": "bar",
    "throttle": "%", "rthrottle": "%", "rmdeceltot": "%",
    # power / energy
    "pmguk": "kW", "pengine": "kW",
    "PPlank_F": "kW", "PPlank_R": "kW",
    "PMGUK_Deploy (MJ)": "kW", "PMGUK_Charge (MJ)": "kW",
    "EPlank_F": "kJ", "EPlank_R": "kJ",
    # misc
    "dmInjector": "kg/hr", "dmInjector (kg/s)": "",
    "tDiff": "s", "rsoc": "MJ", "rsocdelta": "%",
}


# ─── CHANNEL TRANSFORMS ──────────────────────────────────────────────────────
# Per-source sign flips / unit conversions applied after mapping. Missing
# channels are silently skipped.
CHANNEL_TRANSFORMS = {
    "DLS": {
        "aRoll": lambda x: -x,
        "gVert": lambda x: x - 1,
        # Front pushrods are tension-negative → flip to compression-positive.
        "FPushrodFL": lambda x: -x,
        "FPushrodFR": lambda x: -x,
        "FPushrodRL": lambda x: x,
        "FPushrodRR": lambda x: x,
    },
    "CAR": {
        "PBrakeFL": lambda x: -x,
        "PBrakeFR": lambda x: -x,
        "PBrakeRL": lambda x: -x,
        "PBrakeRR": lambda x: -x,
        # FPushrodFR 1.16x is a temporary calibration to match DLS FPRodFL.
        "FPushrodFL": lambda x: -x,
        "FPushrodFR": lambda x: -1.16 * x,
        "FPushrodRL": lambda x: x,
        "FPushrodRR": lambda x: x,
        "rSOCDelta": lambda x: x * (100.0 / 4.0),  # MJ → % of ES capacity
    },
    "DIL": {
        "aRoll": lambda x: -x,
        "PBrakeFL": lambda x: -x,
        "PBrakeFR": lambda x: -x,
        "PBrakeRL": lambda x: -x,
        "PBrakeRR": lambda x: -x,
        # All pushrods are tension-negative on DIL.
        "FPushrodFL": lambda x: -x,
        "FPushrodFR": lambda x: -x,
        "FPushrodRL": lambda x: -x,
        "FPushrodRR": lambda x: -x,
    },
    "OC": {},
    "FMIOpt": {
        "aRoll": lambda x: -x,
    },
}


# ─── PLOT RENDER SETTINGS ────────────────────────────────────────────────────
SCATTER_MAX_POINTS = 45000        # Down-sample scatter above this count.
BAR_SECONDARY_AXIS_RATIO = 20.0   # Auto-trigger secondary y-axis on bar plots.

BOX_PLOT_SETTINGS = {
    "show_points": True,
    "jitter": 0.15,
    "point_alpha": 0.25,
    "point_size": 18,
    "show_fliers": False,
    "box_width": 0.65,
    "box_linewidth": 1.8,
    "box_edge_color": "#4A4A4A",
    "medianline_color": "#1A1A1A",
    "medianline_width": 2.5,
    "aggregated_box_color": "#2E7D99",
    "aggregated_box_alpha": 0.75,
    "per_run_box_alpha": 0.75,
    "figsize_single_channel": (10, 6),
    "figsize_multi_channel": (14, 10),
}


# ─── CALCULATED CHANNELS ─────────────────────────────────────────────────────
# Each entry is a lambda(df) that produces a new column from existing ones.
# Channels whose dependencies are missing in a run are silently skipped.
#
# Helpers below wrap common patterns; every helper decorates the returned
# function with `calc_channel(...)` so the parquet/CSV column projection
# always sees the true source dependencies.

_CORNERS = ("FL", "FR", "RL", "RR")
_DT = lambda: 1.0 / RESAMPLE_RATE  # evaluated at lambda call time, not at import


def _rolling_var(ch, window=32):
    """Envelope of AC content: |value − centred rolling mean|."""
    return calc_channel(ch)(
        lambda df: abs(df[ch] - df[ch].rolling(window=window, min_periods=1, center=True).mean())
    )


def _passthrough(ch):
    """Alias a channel under a new name so a different filter can be applied."""
    return calc_channel(ch)(lambda df: df[ch])


def _cum_energy_kj(ch):
    """Cumulative energy (kJ) from an absolute power in W."""
    return calc_channel(ch)(
        lambda df: cumulative_trapezoid(abs(df[ch] / 1000), dx=_DT(), initial=0)
    )


def _gate_cumtrap(cond_ch, cmp_op, threshold):
    """Cumulative time (s) during which cmp_op(df[cond_ch], threshold) holds."""
    return calc_channel(cond_ch)(
        lambda df: cumulative_trapezoid(cmp_op(df[cond_ch], threshold).astype(float), dx=_DT(), initial=0)
    )


def _gate_ratio(cond_ch, cmp_op, threshold):
    """Rolling ratio: gated time / total elapsed time."""
    def _calc(df):
        mask = cmp_op(df[cond_ch], threshold).astype(float)
        num = cumulative_trapezoid(mask, dx=_DT(), initial=0)
        den = cumulative_trapezoid(np.ones_like(mask), dx=_DT(), initial=0) + 1e-6
        return num / den
    return calc_channel(cond_ch)(_calc)


def _corner_avg_or_native(target, cornerL, cornerR):
    """Prefer the native `target` column; otherwise average the two corners."""
    return calc_channel(target, cornerL, cornerR)(
        lambda df: df[target] if target in df.columns else (df[cornerL] + df[cornerR]) / 2
    )


def _kinematic_camber(corner):
    """OC: aCamberXXKinematic = aCamberXX − aCamberXXComplianceDelta.
    CAR: native aCamberXXKinematic used as-is. FMIOpt: falls back to aCamberXX."""
    target = f"aCamber{corner}Kinematic"
    total  = f"aCamber{corner}"
    delta  = f"aCamber{corner}ComplianceDelta"
    return calc_channel(target, total, delta)(
        lambda df: df[target]
        if target in df.columns
        else (df[total] - df[delta]) if delta in df.columns else df[total]
    )


CALCULATED_CHANNELS = {
    # ── Pushrod loads: averages / deltas ─────────────────────────────────────
    "FPRodAvgF":   lambda df: (df["FPushrodFL"] + df["FPushrodFR"]) / 2,
    "FPRodAvgR":   lambda df: (df["FPushrodRL"] + df["FPushrodRR"]) / 2,
    "FPRodDeltaF": lambda df: df["FPushrodFL"] - df["FPushrodFR"],
    "FPRodDeltaR": lambda df: df["FPushrodRL"] - df["FPushrodRR"],
    # ── Ride modes from corner pushrod forces ────────────────────────────────
    "FPRodHeave": lambda df: df["FPushrodFL"] + df["FPushrodFR"] + df["FPushrodRL"] + df["FPushrodRR"],
    "FPRodPitch": lambda df: (df["FPushrodFL"] + df["FPushrodFR"]) - (df["FPushrodRL"] + df["FPushrodRR"]),
    "FPRodRoll":  lambda df: (df["FPushrodFR"] + df["FPushrodRR"]) - (df["FPushrodFL"] + df["FPushrodRL"]),
    "FPRodWarp":  lambda df: (df["FPushrodFL"] + df["FPushrodRR"]) - (df["FPushrodFR"] + df["FPushrodRL"]),
    # Rolling force variation (envelope of AC content around a 32-sample mean).
    **{f"FProdVar{c}": _rolling_var(f"FPushrod{c}") for c in _CORNERS},
    # Aliased copies so a high-pass filter can be applied via FILTERS.
    **{f"FPushrod{c}_High": _passthrough(f"FPushrod{c}") for c in _CORNERS},
    "FPRodAvgF_High": lambda df: (df["FPushrodFL"] + df["FPushrodFR"]) / 2,
    "FPRodAvgR_High": lambda df: (df["FPushrodRL"] + df["FPushrodRR"]) / 2,

    # ── Damper travel and velocity ───────────────────────────────────────────
    "xDamperAvgF":   lambda df: (df["xDamperFL"] + df["xDamperFR"]) / 2,
    "xDamperAvgR":   lambda df: (df["xDamperRL"] + df["xDamperRR"]) / 2,
    "xDamperDeltaF": lambda df: df["xDamperFL"] - df["xDamperFR"],
    "xDamperDeltaR": lambda df: df["xDamperRL"] - df["xDamperRR"],
    "vDamperAvgF":   lambda df: np.gradient((df["xDamperFL"] + df["xDamperFR"]) / 2, 1.0 / RESAMPLE_RATE, edge_order=2),
    "vDamperAvgR":   lambda df: np.gradient((df["xDamperRL"] + df["xDamperRR"]) / 2, 1.0 / RESAMPLE_RATE, edge_order=2),
    "vDamperDeltaF": lambda df: np.gradient(df["xDamperFL"] - df["xDamperFR"], 1.0 / RESAMPLE_RATE, edge_order=2),
    "vDamperDeltaR": lambda df: np.gradient(df["xDamperRL"] - df["xDamperRR"], 1.0 / RESAMPLE_RATE, edge_order=2),
    **{f"xDamperVar{c}": _rolling_var(f"xDamper{c}") for c in _CORNERS},
    **{f"xDamper{c}_High": _passthrough(f"xDamper{c}") for c in _CORNERS},

    # ── Lateral / longitudinal acceleration ──────────────────────────────────
    "gLat_Abs":    lambda df: df["gLat"].abs(),
    "gLatAbs":     lambda df: df["gLat"].abs(),
    "gLong (raw)": lambda df: df["gLong"],
    "gCombined":   lambda df: np.sqrt(df["gLat"] ** 2 + df["gLong"] ** 2),
    "CosPhi_Calc": lambda df: df["gLong"] / np.sqrt(df["gLat"] ** 2 + df["gLong"] ** 2),

    # ── Ride heights (corner-average fallback for sources without axle native) ─
    "hRideF": _corner_avg_or_native("hRideF", "hRideFL", "hRideFR"),
    "hRideR": _corner_avg_or_native("hRideR", "hRideRL", "hRideRR"),
    "hRideF (raw)":  lambda df: df["hRideF"],
    "hRideR (raw)":  lambda df: df["hRideR"],
    "hRideF (high)": lambda df: df["hRideF"],
    "hRideR (high)": lambda df: df["hRideR"],

    # ── Kinematic camber (per-source fallback strategy) ──────────────────────
    **{f"aCamber{c}Kinematic": _kinematic_camber(c) for c in _CORNERS},

    # ── Power unit ───────────────────────────────────────────────────────────
    "PPUTotal":           lambda df: df["PMGUK"] + df["PEngine"],
    "nWheelAvg_R":        lambda df: (df["nWheelRL"] + df["nWheelRR"]) / 2,
    "dmInjector (kg/s)":  lambda df: df["dmInjector"] / 3600,
    "PMGUK_Deploy (MJ)":  lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] > 0).astype(float)).abs(),
    "PMGUK_Charge (MJ)":  lambda df: (df["PMGUK"] / 1000 * (df["PMGUK"] < 0).astype(float)).abs(),
    "rSOCDelta": calc_channel("rSOCDelta", "rSOC")(
        lambda df: df["rSOCDelta"] if "rSOCDelta" in df.columns else df["rSOC"] - df["rSOC"].dropna().iloc[0]
    ),

    # ── Plank wear (gated on load > 500 N; NaN-safe integration) ─────────────
    "PPlank_F": lambda df: 0.001 * np.maximum(0.1 * df["FzPlankF"] * (df["vCar"] / 3.6), 0) * (df["FzPlankF"] > 500).astype(float),
    "PPlank_R": lambda df: 0.001 * np.maximum(0.1 * df["FzPlankR"] * (df["vCar"] / 3.6), 0) * (df["FzPlankR"] > 500).astype(float),
    # Plot-only smoothed copies (low-pass applied via FILTERS).
    "FzPlankF (smooth)": lambda df: df["FzPlankF"],
    "FzPlankR (smooth)": lambda df: df["FzPlankR"],
    # nan_to_num prevents the cumulative integral from flatlining across
    # source-file dropouts that survive the interpolate limit.
    "EPlank_F": lambda df: cumulative_trapezoid(np.nan_to_num(df["PPlank_F"].to_numpy(), nan=0.0), dx=1.0 / RESAMPLE_RATE, initial=0),
    "EPlank_R": lambda df: cumulative_trapezoid(np.nan_to_num(df["PPlank_R"].to_numpy(), nan=0.0), dx=1.0 / RESAMPLE_RATE, initial=0),
    "tLap_Calc": lambda df: cumulative_trapezoid(np.ones_like(df["vCar"]), dx=1.0 / RESAMPLE_RATE, initial=0),

    # ── Tyre / suspension (OC sources) ───────────────────────────────────────
    "FzTyreF_Avg":     lambda df: (df["FzTyreFL"] + df["FzTyreFR"]) / 2,
    "FzTyreR_Avg":     lambda df: (df["FzTyreRL"] + df["FzTyreRR"]) / 2,
    "FzTyreF_Delta":   lambda df: df["FzTyreFL"] - df["FzTyreFR"],
    "FzTyreR_Delta":   lambda df: df["FzTyreRL"] - df["FzTyreRR"],
    "xHubVertF_Avg":   lambda df: (df["xHubVertFL"] + df["xHubVertFR"]) / 2,
    "xHubVertR_Avg":   lambda df: (df["xHubVertRL"] + df["xHubVertRR"]) / 2,
    "xHubVertF_Delta": lambda df: df["xHubVertFL"] - df["xHubVertFR"],
    "xHubVertR_Delta": lambda df: df["xHubVertRL"] - df["xHubVertRR"],
    "CPLV_Front": calc_channel("FzTyreFL", "FzTyreFR", "FzTyreRL", "FzTyreRR")(
        lambda df: calculate_cplv(df, "front", sample_rate=RESAMPLE_RATE)
    ),
    "CPLV_Rear": calc_channel("FzTyreFL", "FzTyreFR", "FzTyreRL", "FzTyreRR")(
        lambda df: calculate_cplv(df, "rear", sample_rate=RESAMPLE_RATE)
    ),

    # ── Aero (CL proxies from pushrod loads at high speed) ───────────────────
    "vWindHead": lambda df: df["vAir"] - df["vCar"],
    "SC_CLT":    lambda df: df["CLiftTotal"] * ((df["vCar"] + df["vWindHead"]) / df["vCar"]) ** 2,
    # Per-side static mass (780 kg × F/R split × 9.81 / 2) subtracted so the
    # plateau reflects aero downforce only. vCar ≤ 10 kph is masked to NaN.
    "CLF_Proxy": calc_channel("FPRodAvgF", "vCar")(
        lambda df: (df["FPRodAvgF"] - (780 * 0.45 * 9.81 / 2)) / (df["vCar"] ** 2).where(df["vCar"] > 10)
    ),
    "CLR_Proxy": calc_channel("FPRodAvgR", "vCar")(
        lambda df: (df["FPRodAvgR"] - (780 * 0.55 * 9.81 / 2)) / (df["vCar"] ** 2).where(df["vCar"] > 10)
    ),

    # ── Brake power / energy ─────────────────────────────────────────────────
    "PBrakeF_Avg": lambda df: (df["PBrakeFL"] + df["PBrakeFR"]) / 2,
    "PBrakeR_Avg": lambda df: (df["PBrakeRL"] + df["PBrakeRR"]) / 2,
    "PBrakeF":     lambda df: df["PBrakeFL"] + df["PBrakeFR"],
    "PBrakeR":     lambda df: df["PBrakeRL"] + df["PBrakeRR"],
    **{f"EBrake{c}": _cum_energy_kj(f"PBrake{c}") for c in _CORNERS},
    # rMDecelTot: native on CAR .txt; derived from wheel torques on OC parquet.
    "rMDecelTot": calc_channel(
        "rMDecelTot", "MWheelFL", "MWheelFR",
        "MWheelRLInertiaCompensated", "MWheelRRInertiaCompensated", "pBrakeF",
    )(
        lambda df: df["rMDecelTot"]
        if "rMDecelTot" in df.columns
        else (
            100.0
            * (df["MWheelFL"] + df["MWheelFR"])
            / (df["MWheelFL"] + df["MWheelFR"]
               + df["MWheelRLInertiaCompensated"] + df["MWheelRRInertiaCompensated"])
        ).where(df["pBrakeF"] > 5.0)
    ),

    # ── SM (safety-margin) usage metrics ─────────────────────────────────────
    "time_in_SM_100":       _gate_cumtrap("SM", lambda a, b: a >= b, 0.999),
    "time_in_SM_90":        _gate_cumtrap("SM", lambda a, b: a >= b, 0.9),
    "time_in_SM_80":        _gate_cumtrap("SM", lambda a, b: a >= b, 0.8),
    "ratio_time_in_SM_100": _gate_ratio("SM", lambda a, b: a >= b, 0.999),
    "ratio_time_in_SM_90":  _gate_ratio("SM", lambda a, b: a >= b, 0.9),
    "ratio_time_in_SM_80":  _gate_ratio("SM", lambda a, b: a >= b, 0.8),
    "time_grip_limited":        _gate_cumtrap("BGripLimited", lambda a, b: a > b, 0.5),
    "ratio_time_grip_limited":  _gate_ratio("BGripLimited", lambda a, b: a > b, 0.5),
    # 1/0 masks integrated against real tLap in BarPlot (integral aggregation).
    "tGripLimited":  calc_channel("rThrottle")(lambda df: (df["rThrottle"] < 98).astype(float)),
    "tPowerLimited": calc_channel("rThrottle")(lambda df: (df["rThrottle"] >= 98).astype(float)),
}


# ─── FILTERS ──────────────────────────────────────────────────────────────────
# One filter dict used by every workflow. Missing channels are ignored.
#
# Format: {"cutoff": Hz, "order": N, "type": "low"|"high"|"bandpass"}
#   - cutoff = 0 → no filtering
#   - type defaults to "low"
#   - For bandpass, cutoff is a (low_hz, high_hz) tuple
#   - "all" is the fallback for any channel not explicitly listed
#
# To override for a specific workflow, pass `filters={...}` to run_workflow().

def _spec(cutoff, order=2, type_=None):
    d = {"cutoff": cutoff, "order": order}
    if type_ is not None:
        d["type"] = type_
    return d


def _bulk(cutoff, order, chs, type_=None):
    return {ch: _spec(cutoff, order, type_) for ch in chs}


# Channels that MUST stay unfiltered: monotonic ramps (filter causes Gibbs
# ringing at lap resets → breaks sLap alignment), discrete/categorical signals,
# already-integrated channels, and raw traces the PSD path needs untouched.
_UNFILTERED = (
    # distance / time / lap indices
    "sLap", "tLap", "tLap_Calc", "TimeIntoExport", "nLap", "nRun",
    # discrete / integral signals
    "SM", "NGear", "vCar", "nEngine", "rThrottle", "PMGUK", "rMDecelTot",
    "PPUTotal", "dmInjector",
    # vertical accels — raw for PSD
    "gVert", "gVertF", "gVertR",
    "gHubVertFL", "gHubVertFR", "gHubVertRL", "gHubVertRR",
    # pushrod forces — raw for PSD
    "FPushrodFL", "FPushrodFR", "FPushrodRL", "FPushrodRR",
    # ride heights (raw copies)
    "hRideF (raw)", "hRideR (raw)",
    # brakes
    "PBrakeFL", "PBrakeFR", "PBrakeRL", "PBrakeRR", "PBrakeF", "PBrakeR",
    # plank / energy
    "FzPlankF", "FzPlankR", "EPlank_F", "EPlank_R", "PPlank_F", "PPlank_R",
    "nWheelAvg_R",
    # misc
    "gLong (raw)", "CPLV_Front", "CPLV_Rear", "tDiff",
    "dtLap_dhCoGStatic", "dtLap_dxCoGStatic", "dtLap_dCDragTotal", "dtLap_dCLiftTotal",
    "dtLap_dhCoGStatic_Integral", "dtLap_dxCoGStatic_Integral",
    "dtLap_dCDragTotal_Integral", "dtLap_dCLiftTotal_Integral",
    # damper velocities (derivative already limits bandwidth)
    "vDamperDeltaF", "vDamperDeltaR", "vDamperAvgF", "vDamperAvgR",
    # grip-limited flags
    "BGripLimited", "time_grip_limited", "ratio_time_grip_limited",
)

_RIDE_BAND = {"cutoff": (1.5, 15), "order": 4, "type": "bandpass"}

FILTERS = {
    # Unfiltered channels.
    **_bulk(0, 2, _UNFILTERED),
    # Ride modes — bandpass to isolate 1.5–15 Hz dynamic content.
    "FPRodHeave": dict(_RIDE_BAND), "FPRodPitch": dict(_RIDE_BAND),
    "FPRodRoll":  dict(_RIDE_BAND), "FPRodWarp":  dict(_RIDE_BAND),
    "FPRodAvgF_High": dict(_RIDE_BAND), "FPRodAvgR_High": dict(_RIDE_BAND),
    # Ride-height copies: high-pass isolates AC content; native gets a light LP.
    "hRideF (high)": _spec(0.5, 4, "high"),
    "hRideR (high)": _spec(0.5, 4, "high"),
    "hRideF": _spec(3, 2),
    "hRideR": _spec(3, 2),
    # Plank load smoothing.
    "FzPlankF (smooth)": _spec(3, 2),
    "FzPlankR (smooth)": _spec(3, 2),
    # CL proxies smoothing.
    "CLF_Proxy": _spec(3, 2),
    "CLR_Proxy": _spec(3, 2),
    # Damper / force variation envelopes.
    **_bulk(2, 2, [f"xDamperVar{c}" for c in _CORNERS] + [f"FProdVar{c}" for c in _CORNERS]),
    # High-pass PSD copies (AC content only).
    **_bulk(2, 2,
            [f"FPushrod{c}_High" for c in _CORNERS] + [f"xDamper{c}_High" for c in _CORNERS],
            type_="high"),
    # Shape channels: unfiltered (already band-limited by definition).
    "FPRodHeave_Shape": _spec(0, 4),
    "FPRodPitch_Shape": _spec(0, 4),
    # Suspension / dynamics low-pass.
    "rLLTD":  _spec(1, 4),
    "CosPhi": _spec(3, 3),
    # Fallback: any channel not listed above.
    "all": _spec(5, 2),
}

# Legacy aliases (referenced by older configs / docs).
DEFAULT_FILTERS = CORRELATION_FILTERS = BOXPLOT_FILTERS = DAMPER_FILTERS = RIDE_DIL_FILTERS = FILTERS


# ─── TRACK LENGTHS (metres) ──────────────────────────────────────────────────
TRACK_LENGTHS = {
    "BAH": 5410.6, "MEL": 5274.7, "SHA": 5450.0, "SUZ": 5806.1, "MIA": 5409.2,
    "MTL": 4364.4, "MCO": 3335.8, "BCN": 4657.2, "SPB": 4309.6, "SIL": 5888.6,
    "SPA": 7000.2, "BUD": 4377.0, "ZVT": 4255.6, "MZA": 5793.6, "MAD": 5415.4,
    "BAK": 5997.5, "SIN": 4924.8, "COT": 5510.3, "MEX": 4301.8, "SAO": 4299.9,
    "LAS": 6200.2, "DOH": 5417.3, "YAS": 5281.4, "JED": 6175.2, "SEP": 5540.4,
}
