"""Shared data-cleaning, filtering, PSD, and scatter-fit utilities."""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import butter, filtfilt, welch


# ================================================================
# BASIC CLEANING UTILITIES
# ================================================================

def convert_yes_no_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert YES/NO strings to 1/0 in all DataFrame columns.
    Columns containing mixed string types but INCLUDING YES/NO are converted.
    Other string columns remain unchanged.
    """
    columns_converted = []

    for col in df.columns:
        dtype = df[col].dtype

        # Only operate on object/string columns
        if dtype == "object" or dtype.name in ["string", "str"]:
            non_nan = df[col].dropna()
            if len(non_nan) == 0:
                continue

            # upper-case all values for consistency
            str_values = [str(x).upper() for x in non_nan if isinstance(x, str)]

            if any(v in ["YES", "NO"] for v in str_values):
                df[col] = df[col].astype(str).str.upper().replace({"YES": 1, "NO": 0})
                df[col] = pd.to_numeric(df[col], errors="coerce")
                columns_converted.append(col)

    if columns_converted:
        print(f" Converted YES/NO to 1/0 in: {', '.join(columns_converted)}")

    return df


def sanitize_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert non‑numeric values to NaN and replace sentinel int64 min/max & infinities with NaN.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max

    numeric = numeric.replace([int64_min, int64_max, -np.inf, np.inf], np.nan)

    return numeric


# ================================================================
# CHANNEL MAPPINGS & TRANSFORMATIONS
# ================================================================

def apply_channel_mappings(df: pd.DataFrame, channel_mappings, source_type: str):
    """
    Rename channels according to channel_mappings[source_type].

    mapping format:
        {
            "src_name": "target_name",
            ...
        }

    Only rename when:
        - src exists in df
        - target does NOT already exist (avoid overwrites)
    """
    try:
        mapping = channel_mappings.get(source_type, {})
    except Exception:
        print(f" No channel mappings found for {source_type.upper()} — skipping.")
        return df

    if not mapping:
        return df

    rename_dict = {
        src: tgt
        for src, tgt in mapping.items()
        if src in df.columns and tgt not in df.columns
    }

    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f" Renamed {len(rename_dict)} channels for {source_type.upper()}")

    return df


def apply_transformations(df: pd.DataFrame, source_type: str, channel_transforms):
    """
    Apply per-channel transformation functions.

    channel_transforms format:
    {
        "dls": { "FProdFL": lambda x: -x, ... },
        "car": { ... },
        "all": { ... }
    }

    - 'all': applies to ALL channels of this source.
    - other channel keys apply only to those channels.
    """
    try:
        transforms = channel_transforms.get(source_type, {})
    except Exception:
        print(f" No channel transformations found for {source_type.upper()} — skipping.")
        return df

    if not transforms:
        return df

    transformed_channels = []

    for channel, func in transforms.items():

        # apply to all columns
        if channel.lower() == "all":
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = func(df[col])
            print(f" Applied 'all' transformations to {source_type.upper()} data")
            continue

        # normal single-column transformation
        if channel in df.columns:
            df[channel] = pd.to_numeric(df[channel], errors="coerce")
            df[channel] = func(df[channel])
            transformed_channels.append(channel)
        else:
            print(f"[WARNING][datafunctions] Cannot transform missing channel '{channel}' for source '{source_type.upper()}'.")

    print(f" Applied transformations to {len(transformed_channels)} channels for {source_type.upper()}")
    #print(f" Transformed channels for {source_type.upper()}: {', '.join(transformed_channels)}")
    return df


# ================================================================
# CALCULATED CHANNELS
# ================================================================

def apply_calculated_channels(df: pd.DataFrame, source_type: str, calculated_channels):
    """
    Apply lambda-based computed channels.

    calculated_channels may be:
        - a shared dict: { "NewCol": lambda df: ... }
        - a per-source dict: { "dls": {...}, "car": {...} }
    """
    if calculated_channels is None:
        return df

    if isinstance(calculated_channels, dict) and source_type in calculated_channels:
        calc_set = calculated_channels[source_type]
    else:
        calc_set = calculated_channels

    if not isinstance(calc_set, dict):
        return df

    calculated_channels = []

    for channel_name, func in calc_set.items():
        try:
            df[channel_name] = pd.to_numeric(func(df), errors="coerce")
            calculated_channels.append(channel_name)
        except KeyError as e:
            print(f"[WARNING][datafunctions] Missing dependency {e} for calculated channel '{channel_name}'.")
        except Exception as e:
            print(f"[WARNING][datafunctions] Could not compute calculated channel '{channel_name}': {e}")
    print(f" Added {len(calculated_channels)} calculated channels for {source_type.upper()}")
    #print(f" Added calculated channels for {source_type.upper()}: {', '.join(calculated_channels)}")
    return df


# ================================================================
# LOW-PASS FILTERING
# ================================================================

def apply_lowpass_filters(df: pd.DataFrame, low_pass_filters, sample_rate: float, source_type: str):
    """
    Apply Butterworth low-pass filters to channels.

    low_pass_filters format:
        {
            "FzPlankF": {"cutoff": 4, "order": 2},
            "all": {"cutoff": 5, "order": 2}
        }

    A channel may have source-specific settings:
        {"cutoff": 5, "order": 2}
        OR
        {"dls": {"cutoff": ...}, "track": {"cutoff": ...}}

    Any non-numeric or too-short series is skipped safely.
    """
    if not low_pass_filters:
        return df

    filtered = []
    channels_to_skip = []
    filter_all = "all" in low_pass_filters
    all_cfg = low_pass_filters.get("all", None)

    # ------------------------------
    # First: specific channels
    # ------------------------------
    for channel, cfg in low_pass_filters.items():
        if channel == "all":
            continue

        if channel not in df.columns:
            print(f"[WARNING][datafunctions] Cannot filter missing channel '{channel}'.")
            continue

        channels_to_skip.append(channel)
        df[channel] = pd.to_numeric(df[channel], errors="coerce")

        # choose correct config
        if isinstance(cfg, dict) and source_type in cfg:
            config = cfg[source_type]
        else:
            config = cfg

        if "cutoff" not in config:
            print(f"[WARNING][datafunctions] Invalid filter config for channel '{channel}'.")
            continue

        cutoff = config["cutoff"]
        order = config.get("order", 2)

        if cutoff <= 0:
            continue

        # Prepare the signal
        data = df[channel].values
        if len(data) <= order * 3 or np.all(np.isnan(data)):
            print(f"[WARNING][datafunctions] Not enough data to filter channel '{channel}'.")
            continue

        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist

        if normal_cutoff >= 1.0:
            print(f"[WARNING][datafunctions] Cutoff is too high for channel '{channel}'. Skipping filter.")
            continue

        b, a = butter(order, normal_cutoff, btype="low", analog=False)

        # interpolate missing data
        mask_nan = np.isnan(data)
        if mask_nan.any():
            interp = pd.Series(data).interpolate("linear", limit_direction="both").values
            filtered_data = filtfilt(b, a, interp)
            filtered_data[mask_nan] = np.nan
        else:
            filtered_data = filtfilt(b, a, data)

        df[channel] = filtered_data
        filtered.append(f"{channel}@{cutoff}Hz")

    # ------------------------------
    # Second: generic "all" channels
    # ------------------------------
    if filter_all and all_cfg:
        cutoff = all_cfg.get("cutoff", 0)
        order = all_cfg.get("order", 2)

        if cutoff > 0:
            for col in df.columns:
                if col in channels_to_skip:
                    continue

                df[col] = pd.to_numeric(df[col], errors="coerce")
                data = df[col].values

                if len(data) <= order * 3 or np.all(np.isnan(data)):
                    continue

                nyquist = 0.5 * sample_rate
                normal_cutoff = cutoff / nyquist

                if normal_cutoff >= 1.0:
                    continue

                b, a = butter(order, normal_cutoff, btype="low", analog=False)

                mask_nan = np.isnan(data)
                if mask_nan.any():
                    interp = pd.Series(data).interpolate("linear", limit_direction="both").values
                    filtered_data = filtfilt(b, a, interp)
                    filtered_data[mask_nan] = np.nan
                else:
                    filtered_data = filtfilt(b, a, data)

                df[col] = filtered_data
                filtered.append(f"{col}@{cutoff}Hz")

    if filtered:
        print(f" Applied {len(filtered)} low-pass filters for {source_type.upper()}")
        #print(f" Applied low-pass filters: {', '.join(filtered)}")

    return df


# ================================================================
# PSD CALCULATION
# ================================================================

def calculate_psd(signal, sample_rate, nperseg=512):
    """
    Calculate PSD using Welch’s method.
    Returns (frequencies, power).

    Returns (None, None) when signal is too short.
    """
    series = pd.to_numeric(pd.Series(signal), errors="coerce").dropna()
    series = np.asarray(series, dtype=float)

    if len(series) < 8:
        return None, None

    nperseg = min(nperseg, len(series))
    if nperseg < 8:
        return None, None

    freq, power = welch(series, fs=sample_rate, nperseg=nperseg)
    return freq, power


# ================================================================
# SCATTER PLOTTING HELPERS
# ================================================================

def find_best_text_position(ax):
    """
    Detect which corner of a scatter plot has the fewest data points.
    Returns:
        (x_pos, y_pos, horizontal_alignment, vertical_alignment)
    """
    xs = []
    ys = []

    for coll in ax.collections:
        offs = coll.get_offsets()
        if len(offs) > 0:
            xs.extend(offs[:, 0])
            ys.extend(offs[:, 1])

    if not xs:
        return 0.95, 0.95, "right", "top"

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    (xmin, xmax) = ax.get_xlim()
    (ymin, ymax) = ax.get_ylim()

    # normalise 0–1
    x_norm = (xs - xmin) / (xmax - xmin)
    y_norm = (ys - ymin) / (ymax - ymin)

    corners = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
    threshold = 0.4

    for x, y in zip(x_norm, y_norm):
        if y > (1 - threshold):  # top
            if x < threshold:
                corners["tl"] += 1
            elif x > (1 - threshold):
                corners["tr"] += 1
        elif y < threshold:  # bottom
            if x < threshold:
                corners["bl"] += 1
            elif x > (1 - threshold):
                corners["br"] += 1

    best = min(corners, key=corners.get)

    pos = {
        "tl": (0.05, 0.95, "left", "top"),
        "tr": (0.95, 0.95, "right", "top"),
        "bl": (0.05, 0.05, "left", "bottom"),
        "br": (0.95, 0.05, "right", "bottom"),
    }

    return pos[best]


def plot_scatter(ax, x_data, y_data, label, color, alpha, size, x_var="", y_var=""):
    """Simple scatter plot."""
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for scatter: {label} ({x_var} vs {y_var}).")
        return False, None, None

    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color, label=label, edgecolors="none")
    return True, None, None


def _plot_scatter_fit_line(ax, x_values, y_values, color, linestyle="-", linewidth=1.8):
    """Draw a high-contrast fit line with a distinct linestyle."""
    ax.plot(
        x_values,
        y_values,
        color="#000000",
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.98,
        zorder=4,
    )


def plot_scatter_with_1fit(
    ax, x_data, y_data, label, color, alpha, size, x_var="", y_var="", FIT_LINE_X_LIMITS=None
):
    """Scatter + single linear trendline."""
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for single fit: {label} ({x_var} vs {y_var}).")
        return False, None, None, None, None

    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color, label=label, edgecolors="none")

    if FIT_LINE_X_LIMITS:
        xmin, xmax = FIT_LINE_X_LIMITS
    else:
        xmin, xmax = np.min(x_data), np.max(x_data)

    try:
        slope, interc, rval, _, _ = linregress(x_data, y_data)
    except ValueError:
        print(f"[WARNING][datafunctions] Not enough data for fit: {label} ({x_var} vs {y_var}).")
        return False, None, None, None, None

    xr = np.linspace(xmin, xmax, 100)
    yr = slope * xr + interc
    _plot_scatter_fit_line(ax, xr, yr, color=color, linestyle="--", linewidth=1.9)

    equation = f"y = {slope:.3f}x + {interc:.3f}"
    return True, slope, interc, equation, color


def plot_scatter_with_double_fit(
    ax, x_data, y_data, label, color, alpha, size, x_var="", y_var="", fit_split=None
):
    """Scatter + piecewise two-segment fit."""
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for double fit: {label} ({x_var} vs {y_var}).")
        return False, None, None, None, None

    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color, label=label, edgecolors="none")

    if fit_split is None:
        return plot_scatter_with_1fit(
            ax, x_data, y_data, label, color, alpha, size, x_var, y_var
        )

    axis, split_val = fit_split
    if axis == "x":
        mask_before = x_data < split_val
        mask_after = x_data >= split_val
        axis_name = x_var or "x"
    elif axis == "y":
        mask_before = y_data < split_val
        mask_after = y_data >= split_val
        axis_name = y_var or "y"
    else:
        return plot_scatter_with_1fit(
            ax, x_data, y_data, label, color, alpha, size, x_var, y_var
        )

    eq_text = ""
    slope_before = interc_before = None
    slope_after = interc_after = None

    # Before split
    if mask_before.sum() > 1:
        xb = x_data[mask_before]
        yb = y_data[mask_before]
        try:
            slope_before, interc_before, _, _, _ = linregress(xb, yb)
        except ValueError:
            print(f"[WARNING][datafunctions] Not enough data for fit: {label} ({x_var} vs {y_var}).")
            return False, None, None, None, None
        xr = np.linspace(np.min(xb), np.max(xb), 50)
        yr = slope_before * xr + interc_before
        _plot_scatter_fit_line(ax, xr, yr, color=color, linestyle="--", linewidth=1.9)
        eq_text += f"{label} ({axis_name} < {split_val}): y = {slope_before:.3f}x + {interc_before:.3f}\n"

    # After split
    if mask_after.sum() > 1:
        xa = x_data[mask_after]
        ya = y_data[mask_after]
        slope_after, interc_after, _, _, _ = linregress(xa, ya)
        xr = np.linspace(np.min(xa), np.max(xa), 50)
        yr = slope_after * xr + interc_after
        _plot_scatter_fit_line(ax, xr, yr, color=color, linestyle="-.", linewidth=1.9)

        eq_text += f"({axis_name} ≥ {split_val}): y = {slope_after:.3f}x + {interc_after:.3f}"

    return True, (slope_before, slope_after), (interc_before, interc_after), eq_text.strip(), color

def plot_scatter_with_multi_fit(
    ax, x_data, y_data, label, color, alpha, size, x_var="", y_var="", fit_defs=None
):
    """Scatter + as many linear fit segments as provided."""
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for multi-fit: {label} ({x_var} vs {y_var}).")
        return False, None, None, None, None

    if not fit_defs:
        return plot_scatter_with_1fit(
            ax, x_data, y_data, label, color, alpha, size, x_var, y_var
        )

    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color, label=label, edgecolors="none")

    slopes_list = []
    intercepts_list = []
    eq_lines = []
    line_styles = ["--", "-.", ":"]

    def _format_bound(value):
        return f"{float(value):.4g}"

    for idx, fit_def in enumerate(fit_defs):
        try:
            axis, min_val, max_val = fit_def
        except (TypeError, ValueError):
            return plot_scatter_with_1fit(
                ax, x_data, y_data, label, color, alpha, size, x_var, y_var
            )

        if axis == "x":
            min_bound = np.min(x_data) if min_val is None else min_val
            max_bound = np.max(x_data) if max_val is None else max_val
            lower_mask = x_data >= min_bound if min_val is None else x_data > min_bound
            upper_mask = x_data <= max_bound if max_val is None else x_data < max_bound
            mask = lower_mask & upper_mask
            axis_name = x_var or "x"
        elif axis == "y":
            min_bound = np.min(y_data) if min_val is None else min_val
            max_bound = np.max(y_data) if max_val is None else max_val
            lower_mask = y_data >= min_bound if min_val is None else y_data > min_bound
            upper_mask = y_data <= max_bound if max_val is None else y_data < max_bound
            mask = lower_mask & upper_mask
            axis_name = y_var or "y"
        else:
            return plot_scatter_with_1fit(
                ax, x_data, y_data, label, color, alpha, size, x_var, y_var
            )

        if mask.sum() <= 1:
            slopes_list.append(None)
            intercepts_list.append(None)
            continue

        xb = x_data[mask]
        yb = y_data[mask]
        try:
            slope, interc, _, _, _ = linregress(xb, yb)
        except ValueError:
            print(f"[WARNING][datafunctions] Not enough data for fit: {label} ({x_var} vs {y_var}).")
            return False, None, None, None, None

        xr = np.linspace(np.min(xb), np.max(xb), 50)
        yr = slope * xr + interc
        _plot_scatter_fit_line(
            ax,
            xr,
            yr,
            color=color,
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=1.9,
        )
        eq_lines.append(
            f"{label} ({_format_bound(min_bound)} < {axis_name} < {_format_bound(max_bound)}): y = {slope:.3f}x + {interc:.3f}"
        )
        slopes_list.append(slope)
        intercepts_list.append(interc)

    return True, tuple(slopes_list), tuple(intercepts_list), "\n".join(eq_lines), color

# ================================================================
# LABEL HELPERS
# ================================================================

def add_units_to_label(var_name: str, units_map: dict):
    """
    Attach units from units_map to a variable label if present.
    """
    key = var_name.lower()
    for k, v in units_map.items():
        if k.lower() == key:
            return f"{var_name} ({v})"
    return var_name
