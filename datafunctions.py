"""Shared data cleaning, filtering, and plotting helpers."""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import butter, filtfilt, welch
from matplotlib import patheffects as pe


def _fmt_g(v, sig=3):
    """Format v to `sig` significant figures, using compact fixed-point when possible."""
    if v == 0:
        return "0"
    raw = f"{v:.{sig}g}"
    # If Python chose scientific notation but the value is in a readable range,
    # switch to fixed-point with thousands separators.
    if "e" in raw or "E" in raw:
        abs_v = abs(v)
        if 1 <= abs_v < 1_000_000:
            decimals = max(0, sig - len(str(int(abs_v))))
            formatted = f"{v:,.{decimals}f}"
            if decimals > 0:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
    return raw


# ================================================================
# BASIC CLEANING UTILITIES
# ================================================================

def _safe_get_config(config_dict, source_type: str, config_name: str) -> dict:
    """
    Safely retrieve configuration for a source type.
    Returns config dict if found, otherwise prints warning and returns empty dict.
    """
    if config_dict is None:
        return {}
    try:
        return config_dict.get(source_type, {})
    except Exception:
        print(f" No {config_name} found for {source_type.upper()} - skipping.")
        return {}


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert series to numeric, coercing non-numeric values to NaN."""
    return pd.to_numeric(series, errors="coerce")


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
                df[col] = df[col].astype(str).str.upper().replace({"YES": 1, "NO": 0}).infer_objects(copy=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                columns_converted.append(col)

    if columns_converted:
        print(f" Converted YES/NO to 1/0 in: {', '.join(columns_converted)}")

    return df


def sanitize_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert non-numeric values to NaN and replace sentinel int64 min/max and infinities with NaN.
    """
    numeric = _to_numeric_safe(series)

    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max

    numeric = numeric.replace([int64_min, int64_max, -np.inf, np.inf], np.nan)

    return numeric


# ================================================================
# CHANNEL MAPPINGS & TRANSFORMATIONS
# ================================================================

def apply_channel_mappings(df: pd.DataFrame, channel_mappings, source_type: str):
    """Rename channels per channel_mappings[source_type]. Skip if target already exists."""
    mapping = _safe_get_config(channel_mappings, source_type, "channel mappings")

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
    """Apply per-channel numeric transforms (sign flips, unit conversions). 'all' applies globally."""
    transforms = _safe_get_config(channel_transforms, source_type, "channel transformations")

    if not transforms:
        return df

    transformed_channels = []

    for channel, func in transforms.items():

        # apply to all columns
        if channel.lower() == "all":
            for col in df.columns:
                df[col] = _to_numeric_safe(df[col])
                df[col] = func(df[col])
            print(f" Applied 'all' transformations to {source_type.upper()} data")
            continue

        # normal single-column transformation
        if channel in df.columns:
            df[channel] = _to_numeric_safe(df[channel])
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
    """Compute derived channels from lambda(df) definitions."""
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
            df[channel_name] = _to_numeric_safe(func(df))
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

def _apply_butterworth_filter_to_data(data, cutoff: float, order: int, sample_rate: float) -> tuple:
    """Apply Butterworth low-pass filter. Returns (filtered_data, success_flag)."""
    if len(data) <= order * 3 or np.all(np.isnan(data)):
        return None, False

    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist

    if normal_cutoff >= 1.0:
        return None, False

    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Interpolate missing data
    mask_nan = np.isnan(data)
    if mask_nan.any():
        interp = pd.Series(data).interpolate("linear", limit_direction="both").values
        filtered_data = filtfilt(b, a, interp)
        filtered_data[mask_nan] = np.nan
    else:
        filtered_data = filtfilt(b, a, data)

    return filtered_data, True


def apply_lowpass_filters(df: pd.DataFrame, low_pass_filters, sample_rate: float, source_type: str):
    """Apply Butterworth low-pass filters. Per-channel configs override the 'all' fallback."""
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
        df[channel] = _to_numeric_safe(df[channel])

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

        filtered_data, success = _apply_butterworth_filter_to_data(
            df[channel].values, cutoff, order, sample_rate
        )

        if not success:
            if cutoff / (0.5 * sample_rate) >= 1.0:
                print(f"[WARNING][datafunctions] Cutoff is too high for channel '{channel}'. Skipping filter.")
            else:
                print(f"[WARNING][datafunctions] Not enough data to filter channel '{channel}'.")
            continue

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

                df[col] = _to_numeric_safe(df[col])
                
                filtered_data, success = _apply_butterworth_filter_to_data(
                    df[col].values, cutoff, order, sample_rate
                )

                if success:
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
    Calculate PSD using Welch's method.
    Returns (frequencies, power).

    Returns (None, None) when signal is too short.
    """
    series = _to_numeric_safe(pd.Series(signal)).dropna()
    series = np.asarray(series, dtype=float)

    if len(series) < 8:
        return None, None

    nperseg = min(nperseg, len(series))
    if nperseg < 8:
        return None, None

    freq, power = welch(series, fs=sample_rate, nperseg=nperseg)
    return freq, power


# ================================================================
# GENERAL PLOT HELPERS
# ================================================================

def mask_waveform_discontinuities(x_values, y_values):
    """Mask invalid lap-distance regions so line plots break at discontinuities."""
    xs = pd.Series(x_values).reset_index(drop=True)
    ys = pd.Series(y_values).reset_index(drop=True).copy()

    neg_mask = xs < 0
    xs.loc[neg_mask] = np.nan
    ys.loc[neg_mask] = np.nan

    if xs.notna().sum() > 1:
        reset_mask = xs.diff() < 0
        ys.loc[reset_mask] = np.nan

    return xs, ys


def format_psd_ylabel(channel, units_map):
    """Format PSD y-axis label with units when available."""
    units = ""
    if units_map:
        for key, value in units_map.items():
            if key.lower() == channel.lower():
                units = value
                break

    if units:
        return f"{channel} PSD ({units}^2/Hz)"
    return f"{channel} PSD"


def compute_nice_histogram_bins(data, num_bins=30):
    """Compute round-number histogram bins with integer-preferred widths."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0])

    data_min = float(np.min(values))
    data_max = float(np.max(values))

    if np.isclose(data_min, data_max):
        start = np.floor(data_min)
        return np.array([start, start + 1.0])

    raw_step = (data_max - data_min) / max(num_bins, 1)
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    step = nice_fraction * (10 ** exponent)
    if step >= 1:
        step = max(1.0, float(np.round(step)))

    start = np.floor(data_min / step) * step
    end = np.ceil(data_max / step) * step
    bins = np.arange(start, end + step * 0.5, step)

    if bins.size < 2:
        bins = np.array([start, start + step])

    return bins


def compute_equal_width_bins_in_limits(xmin, xmax, reference_bins):
    """Compute equal-width bins in [xmin, xmax] with count derived from a near-nice step."""
    xmin = float(xmin)
    xmax = float(xmax)
    if xmax <= xmin:
        return np.array([xmin, xmin + 1.0])

    if reference_bins is not None and len(reference_bins) > 1:
        target_step = float(reference_bins[1] - reference_bins[0])
    else:
        target_step = (xmax - xmin) / 30.0

    if target_step <= 0:
        target_step = (xmax - xmin) / 30.0

    bin_count = max(1, int(np.round((xmax - xmin) / target_step)))
    return np.linspace(xmin, xmax, bin_count + 1)


# ================================================================
# BAR-PLOT HELPERS
# ================================================================

def normalize_bar_metric_specs(metric_specs, default_aggregation="last"):
    """Normalize bar metric specs into [(channel, aggregation), ...]."""
    if isinstance(metric_specs, str):
        metric_specs = (metric_specs,)

    if not isinstance(metric_specs, (list, tuple)):
        return []

    normalized = []
    valid_aggs = {
        "sum", "mean", "min", "max", "median", "integral",
        "abs_sum", "abs_integral", "first", "last",
    }

    for item in metric_specs:
        if isinstance(item, str):
            normalized.append((item, default_aggregation))
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 1:
            channel = item[0]
            if not isinstance(channel, str):
                continue
            aggregation = item[1] if len(item) >= 2 else default_aggregation
            if not isinstance(aggregation, str):
                aggregation = default_aggregation
            aggregation = aggregation.lower().strip()
            if aggregation not in valid_aggs:
                aggregation = default_aggregation
            normalized.append((channel, aggregation))

    return normalized


def aggregate_channel_for_bar(series, aggregation="last", sample_rate=100.0, time_series=None):
    """Aggregate a channel series into a scalar for grouped bar plots."""
    values = _to_numeric_safe(series).dropna()
    if values.empty:
        return np.nan

    agg = str(aggregation).lower().strip()

    if agg == "sum":
        return float(values.sum())
    if agg == "mean":
        return float(values.mean())
    if agg == "min":
        return float(values.min())
    if agg == "max":
        return float(values.max())
    if agg == "median":
        return float(values.median())
    if agg == "abs_sum":
        return float(values.abs().sum())
    if agg == "integral":
        if time_series is not None:
            times = _to_numeric_safe(pd.Series(time_series)).dropna()
            if len(times) == len(values):
                return float(np.trapz(values, times))
        dt = 1.0 / float(sample_rate) if sample_rate else 1.0
        return float(values.sum() * dt)
    if agg == "abs_integral":
        if time_series is not None:
            times = _to_numeric_safe(pd.Series(time_series)).dropna()
            if len(times) == len(values):
                return float(np.trapz(values.abs(), times))
        dt = 1.0 / float(sample_rate) if sample_rate else 1.0
        return float(values.abs().sum() * dt)
    if agg == "first":
        return float(values.iloc[0])
    if agg == "last":
        return float(values.iloc[-1])

    # fallback
    return float(values.iloc[-1])


# ================================================================
# SCATTER PLOTTING HELPERS
# ================================================================


def _decimate_xy(x_data, y_data, max_points):
    """Downsample evenly when data volume is large to keep plots responsive."""
    if max_points is None or max_points <= 0:
        return x_data, y_data
    if len(x_data) <= max_points:
        return x_data, y_data

    stride = max(1, int(np.ceil(len(x_data) / float(max_points))))
    return x_data[::stride], y_data[::stride]


def _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=45000):
    """Plot scatter points with optional decimation for dense data."""
    x_plot, y_plot = _decimate_xy(x_data, y_data, max_points=max_points)
    ax.scatter(x_plot, y_plot, alpha=alpha, s=size, color=color, label=label, edgecolors="none")


def plot_scatter(ax, x_data, y_data, label, color, alpha, size, x_var="", y_var="", max_points=45000):
    """Simple scatter plot."""
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for scatter: {label} ({x_var} vs {y_var}).")
        return False, None, None

    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)
    return True, None, None


def _plot_scatter_fit_line(ax, x_values, y_values, color, linestyle="-", linewidth=1.8):
    """Draw a run-colored fit line with a light halo and endpoint markers."""
    line = ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth + 0.4,
        alpha=0.99,
        zorder=7,
    )[0]
    line.set_path_effects(
        [pe.Stroke(linewidth=linewidth + 2.4, foreground=(1.0, 1.0, 1.0, 0.96)), pe.Normal()]
    )

    if len(x_values) >= 2:
        ax.plot(
            [x_values[0], x_values[-1]],
            [y_values[0], y_values[-1]],
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor=(1.0, 1.0, 1.0, 0.95),
            markeredgewidth=0.8,
            zorder=8,
            alpha=0.99,
        )


def plot_scatter_with_1fit(
    ax,
    x_data,
    y_data,
    label,
    color,
    alpha,
    size,
    x_var="",
    y_var="",
    FIT_LINE_X_LIMITS=None,
    max_points=45000,
):
    """Scatter + single linear trendline."""
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for single fit: {label} ({x_var} vs {y_var}).")
        return False, None, None, None, None

    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)

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
    _plot_scatter_fit_line(ax, xr, yr, color=color, linestyle="-", linewidth=1.6)

    sign = "−" if interc < 0 else "+"
    equation = f"y = {_fmt_g(slope)} x {sign} {_fmt_g(abs(interc))}"
    return True, slope, interc, equation, color


def plot_scatter_with_multi_fit(
    ax,
    x_data,
    y_data,
    label,
    color,
    alpha,
    size,
    x_var="",
    y_var="",
    fit_defs=None,
    fit_condition_data=None,
    max_points=45000,
):
    """Scatter + as many linear fit segments as provided.

    `fit_condition_data` may provide additional aligned channels used as
    fit-condition axes (for example, axis='SM').
    """
    if len(x_data) == 0:
        print(f"[WARNING][datafunctions] No data for multi-fit: {label} ({x_var} vs {y_var}).")
        return False, None, None, None, None

    if not fit_defs:
        return plot_scatter_with_1fit(
            ax, x_data, y_data, label, color, alpha, size, x_var, y_var, max_points=max_points,
        )

    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)

    slopes_list = []
    intercepts_list = []
    eq_lines = []
    line_styles = ["-", "-", "-"]

    def _format_bound(value, is_lower=True, fallback=None):
        """Format range bounds compactly; use data min/max when bound is open."""
        if value is None or not np.isfinite(value):
            if fallback is not None:
                return f"{float(fallback):.4g}"
            return "$-\\infty$" if is_lower else "$+\\infty$"
        return f"{float(value):.4g}"

    for idx, fit_def in enumerate(fit_defs):
        fit_mask_info = build_multi_fit_mask(
            fit_def,
            x_data=x_data,
            y_data=y_data,
            fit_condition_data=fit_condition_data,
            x_var=x_var,
            y_var=y_var,
        )
        if fit_mask_info["status"] == "invalid_definition":
            return plot_scatter_with_1fit(
                ax, x_data, y_data, label, color, alpha, size, x_var, y_var, max_points=max_points,
            )
        if fit_mask_info["status"] == "missing_condition_channel":
            print(
                f"[WARNING][datafunctions] No fit condition channel '{fit_mask_info['axis_name']}' "
                f"for run '{label}'. Segment skipped."
            )
            slopes_list.append(None)
            intercepts_list.append(None)
            continue

        axis_name = fit_mask_info["axis_name"]
        min_bound = fit_mask_info["min_bound"]
        max_bound = fit_mask_info["max_bound"]
        mask = fit_mask_info["mask"]

        if mask.sum() <= 1:
            slopes_list.append(None)
            intercepts_list.append(None)
            continue

        xb = x_data[mask]
        yb = y_data[mask]
        try:
            slope, interc, _, _, _ = linregress(xb, yb)
        except ValueError:
            print(
                f"[WARNING][datafunctions] Not enough data for fit segment {idx + 1} "
                f"of '{label}' ({x_var} vs {y_var}). Skipping segment."
            )
            slopes_list.append(None)
            intercepts_list.append(None)
            continue

        xr = np.linspace(np.min(xb), np.max(xb), 50)
        yr = slope * xr + interc
        _plot_scatter_fit_line(
            ax,
            xr,
            yr,
            color=color,
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=1.6,
        )
        # Determine fallback bounds from data when min/max_bound is None
        axis_key = fit_def[0].lower() if isinstance(fit_def[0], str) else ""
        if axis_key == "x":
            lo_fallback = float(np.nanmin(x_data[np.isfinite(x_data)])) if np.any(np.isfinite(x_data)) else None
            hi_fallback = float(np.nanmax(x_data[np.isfinite(x_data)])) if np.any(np.isfinite(x_data)) else None
        elif axis_key == "y":
            lo_fallback = float(np.nanmin(y_data[np.isfinite(y_data)])) if np.any(np.isfinite(y_data)) else None
            hi_fallback = float(np.nanmax(y_data[np.isfinite(y_data)])) if np.any(np.isfinite(y_data)) else None
        elif fit_condition_data and fit_def[0] in fit_condition_data:
            cond_arr = np.asarray(fit_condition_data[fit_def[0]], dtype=float)
            lo_fallback = float(np.nanmin(cond_arr[np.isfinite(cond_arr)])) if np.any(np.isfinite(cond_arr)) else None
            hi_fallback = float(np.nanmax(cond_arr[np.isfinite(cond_arr)])) if np.any(np.isfinite(cond_arr)) else None
        else:
            lo_fallback = None
            hi_fallback = None
        lo = _format_bound(min_bound, is_lower=True, fallback=lo_fallback)
        hi = _format_bound(max_bound, is_lower=False, fallback=hi_fallback)
        eq_sign = "−" if interc < 0 else "+"
        eq_lines.append(
            f"{axis_name} $\\in$ [{lo}, {hi}]   y = {_fmt_g(slope)} x {eq_sign} {_fmt_g(abs(interc))}"
        )
        slopes_list.append(slope)
        intercepts_list.append(interc)

    if not eq_lines:
        return False, tuple(slopes_list), tuple(intercepts_list), None, color

    return True, tuple(slopes_list), tuple(intercepts_list), "\n".join(eq_lines), color


def collect_multi_fit_condition_channels(fit_defs):
    """Collect channel names referenced as multi-fit condition axes.

    Conditions using axis 'x' or 'y' are excluded.
    """
    channels = set()
    if not isinstance(fit_defs, (list, tuple)):
        return channels

    for fit_def in fit_defs:
        if not isinstance(fit_def, (list, tuple)) or len(fit_def) != 3:
            continue
        axis = fit_def[0]
        if isinstance(axis, str) and axis.lower() not in {"x", "y"}:
            channels.add(axis)
    return channels


def build_fit_condition_data(df, index, fit_defs, plot_name="", run_name=""):
    """Build aligned numeric series for fit-condition channels.

    Returns a dict keyed by channel name. Missing channels are warned and omitted.
    """
    condition_channels = collect_multi_fit_condition_channels(fit_defs)
    data = {}

    for channel in condition_channels:
        if channel not in df.columns:
            print(
                f"[WARNING][datafunctions] Scatter plot '{plot_name}': "
                f"fit condition channel '{channel}' missing in run '{run_name}'."
            )
            continue
        series = pd.to_numeric(df[channel], errors="coerce").reindex(index)
        data[channel] = series.to_numpy(dtype=float)
    return data


# ================================================================
# BOX PLOT UTILITIES
# ================================================================

def apply_gate_to_dataframe(df, gate_spec):
    """Filter a dataframe by gate condition(s). Returns filtered copy."""
    if gate_spec is None:
        return df.copy()
    
    conditions = _normalize_gate_conditions(gate_spec)
    mask = pd.Series(True, index=df.index)

    for condition in conditions:
        if not isinstance(condition, (list, tuple)) or len(condition) != 3:
            print("[WARNING][datafunctions] Invalid gate condition. Skipping dataframe.")
            return df.iloc[0:0].copy()

        channel, operator, value = condition

        if channel not in df.columns:
            print(f"[WARNING][datafunctions] Gate channel '{channel}' missing. Skipping dataframe.")
            return df.iloc[0:0].copy()

        col = pd.to_numeric(df[channel], errors="coerce")

        if operator == '>':
            gate_mask = col > value
        elif operator == '<':
            gate_mask = col < value
        elif operator == '>=':
            gate_mask = col >= value
        elif operator == '<=':
            gate_mask = col <= value
        elif operator == '==':
            gate_mask = col == value
        elif operator == '!=':
            gate_mask = col != value
        elif operator == 'between' and isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            gate_mask = pd.Series(True, index=col.index)
            if low is not None:
                gate_mask &= col >= low
            if high is not None:
                gate_mask &= col <= high
        elif operator == 'outside' and isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            gate_mask = pd.Series(True, index=col.index)
            if low is not None and high is not None:
                gate_mask = (col < low) | (col > high)
            elif low is not None:
                gate_mask = col < low
            elif high is not None:
                gate_mask = col > high
            else:
                gate_mask = pd.Series(False, index=col.index)
        else:
            print(f"[WARNING][datafunctions] Unsupported gate condition for channel '{channel}'. Skipping dataframe.")
            return df.iloc[0:0].copy()

        mask &= gate_mask.fillna(False)

    return df[mask].copy()


def aggregate_channel_for_boxplot(
    run_data_dict,
    channels,
    aggregation_mode='per_run',
    gate_spec=None,
    filtered_run_data=None,
):
    """Aggregate channel data for box plotting. Returns per-run or aggregated dicts."""
    if isinstance(channels, str):
        channels = [channels]
    else:
        channels = list(channels)
    
    result = {}
    if filtered_run_data is None:
        filtered_run_data = {
            run_name: apply_gate_to_dataframe(df, gate_spec) if gate_spec is not None else df
            for run_name, df in run_data_dict.items()
        }
    
    if aggregation_mode == 'per_run':
        # Structure: {run_name: {channel: values_array}}
        for run_name, df in filtered_run_data.items():
            run_dict = {}
            for channel in channels:
                if channel in df.columns:
                    values = df[channel].dropna().values
                    run_dict[channel] = values
                else:
                    run_dict[channel] = np.array([])
            
            result[run_name] = run_dict
    
    elif aggregation_mode == 'aggregated':
        # Structure: {channel: aggregated_values_array}
        for channel in channels:
            all_values = []
            
            for run_name, df in filtered_run_data.items():
                if channel in df.columns:
                    values = df[channel].dropna().values
                    all_values.extend(values)
            
            result[channel] = np.array(all_values)
    
    return result


def build_multi_fit_mask(
    fit_def,
    x_data,
    y_data,
    fit_condition_data=None,
    x_var="",
    y_var="",
):
    """Build a boolean mask for one multi-fit definition.

    Fit definition format:
        (axis_key, min_val, max_val)

    axis_key may be:
        - 'x' or 'y'
        - a channel name present in fit_condition_data
    """
    result = {
        "status": "ok",
        "axis_name": None,
        "min_bound": None,
        "max_bound": None,
        "mask": None,
    }

    if not isinstance(fit_def, (list, tuple)) or len(fit_def) != 3:
        result["status"] = "invalid_definition"
        return result

    axis_key, min_val, max_val = fit_def
    if not isinstance(axis_key, str):
        result["status"] = "invalid_definition"
        return result

    axis_lower = axis_key.lower()
    use_inclusive_bounds = False
    if axis_lower == "x":
        condition_values = np.asarray(x_data, dtype=float)
        axis_name = x_var or "x"
    elif axis_lower == "y":
        condition_values = np.asarray(y_data, dtype=float)
        axis_name = y_var or "y"
    else:
        if not fit_condition_data or axis_key not in fit_condition_data:
            result["status"] = "missing_condition_channel"
            result["axis_name"] = axis_key
            result["mask"] = np.zeros(len(x_data), dtype=bool)
            return result
        condition_values = np.asarray(fit_condition_data[axis_key], dtype=float)
        axis_name = axis_key
        use_inclusive_bounds = True

    valid_axis = np.isfinite(condition_values)
    if valid_axis.any():
        min_bound = np.nanmin(condition_values) if min_val is None else min_val
        max_bound = np.nanmax(condition_values) if max_val is None else max_val
    else:
        min_bound = min_val
        max_bound = max_val

    lower_mask = (
        condition_values >= min_bound
        if (min_val is None or use_inclusive_bounds)
        else condition_values > min_bound
    )
    upper_mask = (
        condition_values <= max_bound
        if (max_val is None or use_inclusive_bounds)
        else condition_values < max_bound
    )

    xy_finite = np.isfinite(x_data) & np.isfinite(y_data)
    mask = valid_axis & xy_finite & lower_mask & upper_mask

    result.update(
        {
            "axis_name": axis_name,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "mask": mask,
        }
    )
    return result

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


# ================================================================
# SCATTER GATING HELPERS
# ================================================================

def is_gate_spec(value):
    """Return True if value matches supported scatter gate formats."""
    if isinstance(value, (list, tuple)) and len(value) == 3 and isinstance(value[0], str):
        return True
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(v, (list, tuple)) and len(v) == 3 and isinstance(v[0], str)
        for v in value
    ):
        return True
    return False


def _normalize_gate_conditions(gate_spec):
    """Normalize a gate spec into a list of 3-item conditions."""
    if gate_spec is None:
        return []
    if isinstance(gate_spec, (list, tuple)) and len(gate_spec) == 3 and isinstance(gate_spec[0], str):
        return [gate_spec]
    return list(gate_spec)


def collect_gate_channels(gate_spec):
    """Collect channel names referenced by a scatter gate specification."""
    channels = set()
    if gate_spec is None:
        return channels

    conditions = _normalize_gate_conditions(gate_spec)
    for condition in conditions:
        if (
            isinstance(condition, (list, tuple))
            and len(condition) == 3
            and isinstance(condition[0], str)
        ):
            channels.add(condition[0])
    return channels


def format_gate_text(gate_spec):
    """Format scatter gate condition(s) for display in a compact info box."""
    if gate_spec is None:
        return None

    conditions = _normalize_gate_conditions(gate_spec)
    lines = ["Gated For:"]

    for condition in conditions:
        if not isinstance(condition, (list, tuple)) or len(condition) != 3:
            continue
        channel, operator, value = condition
        if operator == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            lines.append(f"{channel} $\\in$ [{low}, {high}]")
        else:
            lines.append(f"{channel} {operator} {value}")

    return "\n".join(lines) if len(lines) > 1 else None
