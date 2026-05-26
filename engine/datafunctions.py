"""Shared data cleaning, filtering, and plotting helpers."""

from __future__ import annotations

import difflib

import pandas as pd
import numpy as np
from scipy.stats import linregress, theilslopes
from scipy.signal import butter, filtfilt, welch, resample_poly
from matplotlib import patheffects as pe

from .logger import log


def calc_channel(*deps):
    """Decorator to annotate a calculated-channel lambda with explicit deps (#5).

    Use when the channel body is too dynamic for the regex-based dependency
    extractor in :class:`DataPlotter` (e.g. f-string column names, indirect
    lookups). Example::

        CALCULATED = {
            "EngineEff": calc_channel("nEngine", "tThrottle")(
                lambda df: df["nEngine"] * df["tThrottle"] / 1000.0
            ),
        }

    The dependency list is attached as the ``__dls_deps__`` attribute and
    consumed by ``DataPlotter._extract_calculated_dependencies``.
    """
    def _wrap(fn):
        try:
            fn.__dls_deps__ = tuple(deps)
        except (AttributeError, TypeError):
            pass
        return fn
    return _wrap

# NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid``; keep both call sites working.
_np_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# Shared progress bar wrapper — import in generators as: from datafunctions import _tqdm
try:
    from tqdm import tqdm as _tqdm_raw
    def _tqdm(it, **kw):
        import sys
        return _tqdm_raw(it, file=sys.stderr, dynamic_ncols=True, **kw)
except ImportError:
    def _tqdm(iterable, **kwargs):
        return iterable


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
        log.debug("No %s found for %s - skipping.", config_name, source_type.upper())
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
        log.debug("Converted YES/NO to 1/0 in: %s", ', '.join(columns_converted))

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

def apply_channel_mappings(df: pd.DataFrame, channel_mappings: dict | None, source_type: str) -> pd.DataFrame:
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
        log.debug("Renamed %d channels for %s", len(rename_dict), source_type.upper())

    return df


def apply_transformations(df: pd.DataFrame, source_type: str, channel_transforms: dict | None) -> pd.DataFrame:
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
            log.debug("Applied 'all' transformations to %s data", source_type.upper())
            continue

        # normal single-column transformation
        if channel in df.columns:
            df[channel] = _to_numeric_safe(df[channel])
            df[channel] = func(df[channel])
            transformed_channels.append(channel)
        else:
            log.warning("Cannot transform missing channel '%s' for source '%s'.", channel, source_type.upper())

    log.debug("Applied transformations to %d channels for %s", len(transformed_channels), source_type.upper())
    return df


# ================================================================
# CALCULATED CHANNELS
# ================================================================

def apply_calculated_channels(df: pd.DataFrame, source_type: str, calculated_channels: dict | None,
                              required_channels: set | None = None) -> pd.DataFrame:
    """Compute derived channels from lambda(df) definitions.

    If ``required_channels`` is provided, only channels in that set (and their
    transitive dependencies on other calculated channels) are computed.  Missing
    dependency warnings are demoted to debug for unrequested channels.
    """
    if calculated_channels is None:
        return df

    if isinstance(calculated_channels, dict) and source_type in calculated_channels:
        calc_set = calculated_channels[source_type]
    else:
        calc_set = calculated_channels

    if not isinstance(calc_set, dict):
        return df

    # Restrict to requested channels (+ transitive calc-channel deps).
    target_names = None
    if required_channels is not None:
        target_names = {n for n in required_channels if n in calc_set}
        # Iteratively expand to include any calc-channel referenced by lambda source.
        if target_names:
            try:
                import inspect as _inspect, re as _re
                changed = True
                while changed:
                    changed = False
                    for name in list(target_names):
                        fn = calc_set.get(name)
                        if fn is None:
                            continue
                        try:
                            src = _inspect.getsource(fn)
                        except (OSError, TypeError):
                            continue
                        for tok in _re.findall(r"['\"]([A-Za-z_][\w]*)['\"]", src):
                            if tok in calc_set and tok not in target_names:
                                target_names.add(tok)
                                changed = True
            except Exception:
                pass

    calculated_channels_done = []

    for channel_name, func in calc_set.items():
        is_required = (target_names is None) or (channel_name in target_names)
        try:
            df[channel_name] = _to_numeric_safe(func(df))
            calculated_channels_done.append(channel_name)
        except KeyError as e:
            if is_required:
                log.warning("Missing dependency %s for calculated channel '%s'.", e, channel_name)
            else:
                log.debug("Skipped optional calc channel '%s' (missing dep %s).", channel_name, e)
        except Exception as e:
            if is_required:
                log.warning("Could not compute calculated channel '%s': %s", channel_name, e)
            else:
                log.debug("Skipped optional calc channel '%s': %s", channel_name, e)
    log.debug("Added %d calculated channels for %s", len(calculated_channels_done), source_type.upper())
    return df


# ================================================================
# BUTTERWORTH FILTERING
# ================================================================

def _apply_butterworth_filter_to_data(data, cutoff, order: int, sample_rate: float, btype: str = "low") -> tuple:
    """Apply Butterworth filter. Returns (filtered_data, success_flag).

    Parameters
    ----------
    cutoff : float or list[float]
        Cutoff frequency in Hz.  For bandpass, a two-element list [low, high].
    btype : str
        Filter type: ``"low"``, ``"high"``, or ``"bandpass"``.
    """
    if len(data) <= order * 3 or np.all(np.isnan(data)):
        return None, False

    nyquist = 0.5 * sample_rate

    if btype == "bandpass":
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            return None, False
        normal_cutoff = [c / nyquist for c in cutoff]
        if any(nc >= 1.0 or nc <= 0.0 for nc in normal_cutoff):
            return None, False
    else:
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
            return None, False

    b, a = butter(order, normal_cutoff, btype=btype, analog=False)

    # Interpolate missing data
    mask_nan = np.isnan(data)
    if mask_nan.any():
        interp = pd.Series(data).interpolate("linear", limit_direction="both").values
        filtered_data = filtfilt(b, a, interp)
        filtered_data[mask_nan] = np.nan
    else:
        filtered_data = filtfilt(b, a, data)

    return filtered_data, True


def apply_filters(df: pd.DataFrame, filters: dict | None, sample_rate: float, source_type: str,
                  required_channels: set | None = None) -> pd.DataFrame:
    """Apply Butterworth filters. Per-channel configs override the 'all' fallback.

    Each filter entry may contain an optional ``"type"`` key (``"low"``,
    ``"high"``, or ``"bandpass"``).  When omitted the filter defaults to
    low-pass for full backward compatibility.
    """
    if not filters:
        return df

    applied = []
    channels_to_skip = []
    filter_all = "all" in filters
    all_cfg = filters.get("all", None)

    # ------------------------------
    # First: specific channels
    # ------------------------------
    for channel, cfg in filters.items():
        if channel == "all":
            continue

        if channel not in df.columns:
            if required_channels is None or channel in required_channels:
                log.warning("Cannot filter missing channel '%s'.", channel)
            else:
                log.debug("Skipped filter on missing optional channel '%s'.", channel)
            continue

        channels_to_skip.append(channel)
        df[channel] = _to_numeric_safe(df[channel])

        # choose correct config
        if isinstance(cfg, dict) and source_type in cfg:
            config = cfg[source_type]
        else:
            config = cfg

        if "cutoff" not in config:
            log.warning("Invalid filter config for channel '%s'.", channel)
            continue

        cutoff = config["cutoff"]
        order = config.get("order", 2)
        btype = config.get("type", "low")

        # cutoff=0 disables the filter (unchanged convention)
        if isinstance(cutoff, (int, float)) and cutoff <= 0:
            continue

        filtered_data, success = _apply_butterworth_filter_to_data(
            df[channel].values, cutoff, order, sample_rate, btype=btype,
        )

        if not success:
            log.warning("Filter failed for channel '%s' (type=%s). Skipping.", channel, btype)
            continue

        df[channel] = filtered_data
        label = f"{channel}@{cutoff}Hz({btype})" if btype != "low" else f"{channel}@{cutoff}Hz"
        applied.append(label)

    # ------------------------------
    # Second: generic "all" channels
    # ------------------------------
    if filter_all and all_cfg:
        cutoff = all_cfg.get("cutoff", 0)
        order = all_cfg.get("order", 2)
        btype = all_cfg.get("type", "low")

        if isinstance(cutoff, (int, float)) and cutoff <= 0:
            pass  # disabled
        else:
            for col in df.columns:
                if col in channels_to_skip:
                    continue

                df[col] = _to_numeric_safe(df[col])

                filtered_data, success = _apply_butterworth_filter_to_data(
                    df[col].values, cutoff, order, sample_rate, btype=btype,
                )

                if success:
                    df[col] = filtered_data
                    label = f"{col}@{cutoff}Hz({btype})" if btype != "low" else f"{col}@{cutoff}Hz"
                    applied.append(label)

    if applied:
        log.debug("Applied %d filters for %s", len(applied), source_type.upper())

    return df


# ================================================================
# PSD CALCULATION
# ================================================================

def calculate_psd(signal, sample_rate: float, nperseg: int = 512) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    try:
        from .plot_definitions import _VALID_BAR_AGGS as valid_aggs
    except Exception:
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
                return float(_np_trapezoid(values, times))
        dt = 1.0 / float(sample_rate) if sample_rate else 1.0
        return float(values.sum() * dt)
    if agg == "abs_integral":
        if time_series is not None:
            times = _to_numeric_safe(pd.Series(time_series)).dropna()
            if len(times) == len(values):
                return float(_np_trapezoid(values.abs(), times))
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


_DECIMATE_CACHE = {}
_DECIMATE_CACHE_MAX_ENTRIES = 64


def _decimate_xy(x_data, y_data, max_points):
    """Downsample evenly when data volume is large to keep plots responsive.

    Decimated outputs for the same (x_array, y_array, max_points) inputs are
    memoized by array ``id()`` so that repeat scatter calls (e.g. when the same
    run/channel pair is plotted across multiple figures) reuse the work.
    """
    if max_points is None or max_points <= 0:
        return x_data, y_data
    if len(x_data) <= max_points:
        return x_data, y_data

    cache_key = (id(x_data), id(y_data), int(max_points), len(x_data))
    cached = _DECIMATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    stride = max(1, int(np.ceil(len(x_data) / float(max_points))))
    out = (x_data[::stride], y_data[::stride])

    # Bounded cache — drop oldest entries when above the soft cap.
    if len(_DECIMATE_CACHE) >= _DECIMATE_CACHE_MAX_ENTRIES:
        try:
            _DECIMATE_CACHE.pop(next(iter(_DECIMATE_CACHE)))
        except StopIteration:
            pass
    _DECIMATE_CACHE[cache_key] = out
    return out


def _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=45000):
    """Plot scatter points with optional decimation for dense data."""
    x_plot, y_plot = _decimate_xy(x_data, y_data, max_points=max_points)
    ax.scatter(x_plot, y_plot, alpha=alpha, s=size, color=color, label=label, edgecolors="none")


def plot_scatter(ax, x_data, y_data, label, color, alpha, size, x_var="", y_var="", max_points=45000):
    """Simple scatter plot."""
    if len(x_data) == 0:
        log.warning("No data for scatter: %s (%s vs %s).", label, x_var, y_var)
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
    robust=False,
    robust_threshold=3.0,
):
    """Scatter + single linear trendline.

    When ``robust=True``, uses Theil-Sen regression with MAD outlier
    rejection (#18). The 5th return value becomes a dict with diagnostics
    rather than just the colour; callers may inspect ``info["outlier_mask"]``
    to render the rejected samples.
    """
    if len(x_data) == 0:
        log.warning("No data for single fit: %s (%s vs %s).", label, x_var, y_var)
        return False, None, None, None, None

    if robust:
        info = fit_robust_theilsen(x_data, y_data, outlier_k=robust_threshold)
        if info is None:
            log.warning("Robust fit failed: %s (%s vs %s).", label, x_var, y_var)
            return False, None, None, None, None
        # Plot the inliers normally and outliers as faint grey 'x'.
        inlier_mask = ~info["outlier_mask"]
        _plot_scatter_layer(
            ax, np.asarray(x_data)[inlier_mask], np.asarray(y_data)[inlier_mask],
            label, color, alpha, size, max_points=max_points,
        )
        if info["n_outliers"] > 0:
            ax.scatter(
                np.asarray(x_data)[info["outlier_mask"]],
                np.asarray(y_data)[info["outlier_mask"]],
                s=max(size * 0.7, 8),
                marker="x", color="#9A9A9A", alpha=0.25, linewidth=0.8,
                zorder=1, label="_nolegend_",
            )
        if FIT_LINE_X_LIMITS:
            xmin, xmax = FIT_LINE_X_LIMITS
        else:
            xmin, xmax = np.min(x_data), np.max(x_data)
        slope, interc = info["slope"], info["intercept"]
        xr = np.linspace(xmin, xmax, 100)
        yr = slope * xr + interc
        _plot_scatter_fit_line(ax, xr, yr, color=color, linestyle="-", linewidth=1.6)
        sign = "−" if interc < 0 else "+"
        suffix = (
            f"   [robust: {info['n_outliers']}/{info['n_total']} outliers]"
            if info["n_outliers"] > 0 else "   [robust]"
        )
        equation = f"y = {_fmt_g(slope)} x {sign} {_fmt_g(abs(interc))}{suffix}"
        # Stuff the diagnostics dict into the slot historically used for colour.
        return True, slope, interc, equation, {"color": color, "robust_info": info}

    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)

    if FIT_LINE_X_LIMITS:
        xmin, xmax = FIT_LINE_X_LIMITS
    else:
        xmin, xmax = np.min(x_data), np.max(x_data)

    try:
        slope, interc, rval, _, _ = linregress(x_data, y_data)
    except ValueError:
        log.warning("Not enough data for fit: %s (%s vs %s).", label, x_var, y_var)
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
    robust=False,
    robust_threshold=3.0,
):
    """Scatter + as many linear fit segments as provided.

    `fit_condition_data` may provide additional aligned channels used as
    fit-condition axes (for example, axis='SM').
    When ``robust=True``, each segment uses Theil-Sen with MAD outlier
    rejection independently.
    """
    if len(x_data) == 0:
        log.warning("No data for multi-fit: %s (%s vs %s).", label, x_var, y_var)
        return False, None, None, None, None

    if not fit_defs:
        return plot_scatter_with_1fit(
            ax, x_data, y_data, label, color, alpha, size, x_var, y_var,
            max_points=max_points, robust=robust, robust_threshold=robust_threshold,
        )

    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)

    slopes_list = []
    intercepts_list = []
    eq_lines = []
    line_styles = ["-", "-", "-"]
    total_outliers = 0
    total_points = 0

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
        seg_outlier_count = 0
        if robust:
            info = fit_robust_theilsen(xb, yb, outlier_k=robust_threshold)
            if info is None:
                print(
                    f"[WARNING][datafunctions] Robust fit segment {idx + 1} failed for "
                    f"'{label}' ({x_var} vs {y_var}). Skipping segment."
                )
                slopes_list.append(None)
                intercepts_list.append(None)
                continue
            slope, interc = info["slope"], info["intercept"]
            seg_outlier_count = info["n_outliers"]
            total_outliers += seg_outlier_count
            total_points += info["n_total"]
            if seg_outlier_count > 0:
                ax.scatter(
                    xb[info["outlier_mask"]], yb[info["outlier_mask"]],
                    s=max(size * 0.7, 8),
                    marker="x", color="#9A9A9A", alpha=0.25, linewidth=0.8,
                    zorder=1, label="_nolegend_",
                )
        else:
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
        seg_eq = (
            f"{axis_name} $\\in$ [{lo}, {hi}]   y = {_fmt_g(slope)} x {eq_sign} {_fmt_g(abs(interc))}"
        )
        if robust and seg_outlier_count > 0:
            seg_eq += f"   ({seg_outlier_count} outliers rejected)"
        eq_lines.append(seg_eq)
        slopes_list.append(slope)
        intercepts_list.append(interc)

    if not eq_lines:
        return False, tuple(slopes_list), tuple(intercepts_list), None, color

    meta = {
        "color": color,
        "robust_info": (
            {"n_outliers": total_outliers, "n_total": total_points}
            if robust else None
        ),
    }
    return True, tuple(slopes_list), tuple(intercepts_list), "\n".join(eq_lines), meta


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

def compute_gate_mask(df: pd.DataFrame, gate_spec) -> pd.Series:
    """Build the boolean mask that ``apply_gate_to_dataframe`` would use.

    Returns a boolean ``pd.Series`` aligned to ``df.index``. Invalid or
    missing-channel gates return an all-False mask (matching the
    "skip dataframe" behaviour of ``apply_gate_to_dataframe``).
    """
    if gate_spec is None:
        return pd.Series(True, index=df.index)

    conditions = _normalize_gate_conditions(gate_spec)
    mask = pd.Series(True, index=df.index)

    for condition in conditions:
        if not isinstance(condition, (list, tuple)) or len(condition) != 3:
            log.warning("Invalid gate condition.")
            return pd.Series(False, index=df.index)

        channel, operator, value = condition

        if channel not in df.columns:
            log.warning("Gate channel '%s' missing.", channel)
            return pd.Series(False, index=df.index)

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
            if low is not None and high is not None:
                gate_mask = (col < low) | (col > high)
            elif low is not None:
                gate_mask = col < low
            elif high is not None:
                gate_mask = col > high
            else:
                gate_mask = pd.Series(False, index=col.index)
        elif operator == 'robust':
            try:
                k = float(value)
            except (TypeError, ValueError):
                log.warning("Gate 'robust' needs a numeric k for '%s'.", channel)
                return pd.Series(False, index=df.index)
            finite = col.dropna()
            if finite.empty:
                gate_mask = pd.Series(False, index=col.index)
            else:
                med = finite.median()
                mad = (finite - med).abs().median() * 1.4826
                if mad == 0 or not np.isfinite(mad):
                    gate_mask = (col == med)
                else:
                    gate_mask = (col - med).abs() <= (k * mad)
        else:
            log.warning("Unsupported gate condition for channel '%s'.", channel)
            return pd.Series(False, index=df.index)

        mask &= gate_mask.fillna(False)

    return mask


def apply_gate_to_dataframe(df: pd.DataFrame, gate_spec) -> pd.DataFrame:
    """Filter a dataframe by gate condition(s). Returns filtered copy (or view if no gate)."""
    if gate_spec is None:
        return df
    mask = compute_gate_mask(df, gate_spec)
    if not mask.any():
        return df.iloc[0:0].copy()
    return df[mask].copy()


def resolve_condition_marker(marker, df, x_channel):
    """Expand a condition-triggered Marker into concrete x-values for one run.

    Returns a list of floats \u2014 the x_channel values at each rising / falling /
    either edge of the gate condition. Honours ``marker.max_count``.
    Returns ``[]`` if the condition cannot be evaluated (missing channel etc.)
    or no transitions are found.
    """
    if marker.condition is None:
        return []
    if x_channel not in df.columns:
        return []

    mask = compute_gate_mask(df, marker.condition).astype(bool).to_numpy()
    if mask.size < 2:
        return []

    # True rising / falling edges: only detect transitions WITHIN the series.
    # A sample that is already True at index 0 does not count as a rising edge
    # (the condition didn't "become true" — it started true). Likewise the last
    # sample alone never counts as a falling edge.
    diff = np.diff(mask.astype(np.int8))  # length N-1, indexed by the *new* sample
    if marker.edge == "rising":
        idx = np.flatnonzero(diff == 1) + 1
    elif marker.edge == "falling":
        idx = np.flatnonzero(diff == -1) + 1
    else:  # both
        idx = np.flatnonzero(diff != 0) + 1

    if idx.size == 0:
        return []

    x_values = pd.to_numeric(df[x_channel], errors="coerce").to_numpy()
    # Drop any indices where x is NaN.
    x_hits = [float(x_values[i]) for i in idx if i < len(x_values) and np.isfinite(x_values[i])]

    if marker.max_count is not None and len(x_hits) > marker.max_count:
        x_hits = x_hits[: marker.max_count]
    return x_hits





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
        elif operator == "outside" and isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            lines.append(f"{channel} $\\notin$ [{low}, {high}]")
        elif operator == "robust":
            lines.append(f"|{channel} - median| $\\leq$ {value}$\\cdot$MAD")
        else:
            lines.append(f"{channel} {operator} {value}")

    return "\n".join(lines) if len(lines) > 1 else None


# ================================================================
# FUZZY CHANNEL NAME MATCHING (#10)
# ================================================================

def suggest_similar_channels(target, available, max_results=5, cutoff=0.5):
    """Suggest channel names from `available` that resemble `target`.

    Combines:
      * Substring / prefix matching (case-insensitive)
      * ``difflib.get_close_matches`` with a tunable cutoff

    Returns a deduplicated, ranked list of suggestions (best first).
    """
    if not target or not available:
        return []

    target_lc = target.lower()
    available_list = list(available)

    # 1. Substring / prefix scoring — fast and catches DLS naming patterns.
    scored = []
    for ch in available_list:
        ch_lc = ch.lower()
        if ch_lc == target_lc:
            continue
        if ch_lc.startswith(target_lc) or target_lc.startswith(ch_lc):
            scored.append((0, ch))
        elif target_lc in ch_lc or ch_lc in target_lc:
            scored.append((1, ch))

    scored.sort(key=lambda t: (t[0], t[1].lower()))
    substring_hits = [ch for _, ch in scored[:max_results]]

    # 2. difflib fuzzy match — catches typos.
    fuzzy_hits = difflib.get_close_matches(
        target, available_list, n=max_results, cutoff=cutoff
    )

    # Merge preserving order; substring hits first because they tend to be
    # more relevant for telemetry channel naming conventions.
    seen = set()
    merged = []
    for ch in substring_hits + fuzzy_hits:
        if ch not in seen and ch.lower() != target_lc:
            seen.add(ch)
            merged.append(ch)
        if len(merged) >= max_results:
            break
    return merged


# ================================================================
# ROBUST REGRESSION (#18)
# ================================================================

def fit_robust_theilsen(x, y, outlier_k=3.0):
    """Theil-Sen regression with MAD-based outlier rejection.

    Returns a dict with keys:
        slope, intercept, ci_low, ci_high   - Theil-Sen fit on inliers
        outlier_mask                        - boolean array (True = outlier)
        n_total, n_outliers                 - counts
        pseudo_r2                           - 1 - (MAD residuals / MAD total),
                                              clipped to [0, 1]; rough robust
                                              analogue of R².
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    if xf.size < 3:
        return None

    # First pass on all finite data to estimate residual scale.
    slope, intercept, lo, hi = theilslopes(yf, xf, 0.95)
    resid = yf - (slope * xf + intercept)
    mad_r = np.median(np.abs(resid - np.median(resid))) * 1.4826

    if mad_r > 0 and np.isfinite(mad_r):
        outlier_mask_f = np.abs(resid) > outlier_k * mad_r
        # Refit on inliers if we have enough points left.
        inliers = ~outlier_mask_f
        if inliers.sum() >= 3:
            slope, intercept, lo, hi = theilslopes(yf[inliers], xf[inliers], 0.95)
            resid_in = yf[inliers] - (slope * xf[inliers] + intercept)
            mad_r = np.median(np.abs(resid_in - np.median(resid_in))) * 1.4826
    else:
        outlier_mask_f = np.zeros_like(xf, dtype=bool)

    mad_total = np.median(np.abs(yf - np.median(yf))) * 1.4826
    if mad_total > 0 and np.isfinite(mad_total):
        pseudo_r2 = float(np.clip(1.0 - (mad_r / mad_total) ** 2, 0.0, 1.0))
    else:
        pseudo_r2 = 0.0

    # Re-expand outlier mask to full input length.
    full_mask = np.zeros_like(x, dtype=bool)
    full_mask[finite] = outlier_mask_f

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "outlier_mask": full_mask,
        "n_total": int(finite.sum()),
        "n_outliers": int(outlier_mask_f.sum()),
        "pseudo_r2": pseudo_r2,
    }


# ================================================================
# SAMPLE-RATE DETECTION (#17)
# ================================================================

def _rational_ratio(target: float, src: float,
                    max_denom: int = 1000) -> tuple[int, int]:
    """Approximate ``target / src`` as a reduced integer ratio (up, down)."""
    from math import gcd
    # Multiply by 1000 then round, then reduce. Good enough for sample rates.
    up = max(1, int(round(target * 1000)))
    down = max(1, int(round(src * 1000)))
    g = gcd(up, down) or 1
    up //= g
    down //= g
    if up > max_denom or down > max_denom:
        # Fall back to rounded integer Hz values
        up = max(1, int(round(target)))
        down = max(1, int(round(src)))
        g = gcd(up, down) or 1
        up //= g
        down //= g
    return up, down


def resample_to_uniform_rate(df: pd.DataFrame, target_rate: float,
                             time_col: str = "tLap",
                             run_name: str | None = None) -> pd.DataFrame:
    """Resample every numeric channel to a uniform ``target_rate`` (Hz).

    Pipeline:
      1. Estimate the source rate from ``time_col`` (median positive dt).
      2. If src ≈ target (within 0.5 %) the frame is returned unchanged.
      3. Otherwise each numeric column is resampled with
         :func:`scipy.signal.resample_poly`, which applies an FIR
         anti-alias / interpolation filter — preventing the aliasing
         that naive linear interpolation produces on downsampling and
         the stair-step harmonics it leaves on upsampling.

    NaNs are filled by linear interpolation before polyphase resampling
    so the FIR filter has no holes; an all-NaN column becomes an all-NaN
    column of the new length. Non-numeric columns use nearest-neighbour.

    Called BEFORE filter design so that Butterworth cutoffs at
    ``target_rate`` are consistent channel-to-channel and run-to-run.
    """
    if df is None or df.empty:
        return df
    try:
        target_rate = float(target_rate)
    except (TypeError, ValueError):
        return df
    if target_rate <= 0:
        return df
    if time_col not in df.columns:
        log.debug("resample: skipped (no '%s' column) for %s",
                  time_col, run_name or "<run>")
        return df

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    n_old = len(df)
    if n_old < 4 or not np.isfinite(t).any():
        return df

    # Estimate source rate from the median positive dt (handles per-lap resets).
    dt = np.diff(t)
    valid_dt = (dt > 0) & (dt < 1.0) & np.isfinite(dt)
    if valid_dt.sum() < 5:
        log.debug("resample: skipped (irregular '%s') for %s",
                  time_col, run_name or "<run>")
        return df
    med_dt = float(np.median(dt[valid_dt]))
    if not np.isfinite(med_dt) or med_dt <= 0:
        return df
    src_rate = 1.0 / med_dt

    # Already at target — skip work.
    if abs(src_rate - target_rate) / target_rate < 0.005:
        log.debug("resample: %s already at %.2f Hz (target %.2f) — skipped",
                  run_name or "<run>", src_rate, target_rate)
        return df

    up, down = _rational_ratio(target_rate, src_rate)
    if up == down:
        return df

    n_new = int(np.floor(n_old * up / down))
    if n_new < 2:
        return df

    # Columns that must NOT pass through the polyphase FIR.
    #   - Time/distance columns reset to 0 at lap boundaries (sawtooth) — the
    #     anti-alias filter would ring at every reset, distorting the time axis.
    #   - Lap counters are integer step functions — must stay integer-valued.
    # These are interpolated against the resampled time grid directly.
    _LINEAR_INTERP_COLS = {"tlap", "slap"}
    _NEAREST_COLS = {"nlap"}
    idx_old = np.arange(n_old)
    t_src_uniform = idx_old / src_rate
    t_new_uniform = np.arange(n_new) / target_rate

    out: dict[str, np.ndarray] = {}
    for col in df.columns:
        s = df[col]
        col_key = col.lower() if isinstance(col, str) else ""
        if pd.api.types.is_numeric_dtype(s):
            y = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            nan_mask = ~np.isfinite(y)
            if nan_mask.all():
                out[col] = np.full(n_new, np.nan)
                continue
            if nan_mask.any():
                # Fill holes so the polyphase filter has no NaN.
                valid = ~nan_mask
                y = np.interp(idx_old, idx_old[valid], y[valid])
            # Control columns bypass polyphase (see comment above).
            if col_key in _LINEAR_INTERP_COLS:
                y_new = np.interp(t_new_uniform, t_src_uniform, y)
                out[col] = y_new
                continue
            if col_key in _NEAREST_COLS:
                idx_new = np.clip(
                    np.round(t_new_uniform * src_rate).astype(int), 0, n_old - 1
                )
                out[col] = y[idx_new]
                continue
            try:
                y_new = resample_poly(y, up, down)
            except Exception as exc:  # pragma: no cover - extreme edge cases
                log.warning("resample: %s column '%s' fell back to linear (%s)",
                            run_name or "<run>", col, exc)
                y_new = np.interp(t_new_uniform, t_src_uniform, y)
            # resample_poly may return slightly different length; align.
            if len(y_new) != n_new:
                if len(y_new) > n_new:
                    y_new = y_new[:n_new]
                else:
                    y_new = np.concatenate(
                        (y_new, np.full(n_new - len(y_new), y_new[-1] if len(y_new) else 0.0))
                    )
            out[col] = y_new
        else:
            # Nearest-neighbour for non-numeric columns.
            if n_new == 1:
                idx_new = np.array([0])
            else:
                idx_new = np.round(
                    np.linspace(0, n_old - 1, n_new)
                ).astype(int)
            out[col] = s.to_numpy()[np.clip(idx_new, 0, n_old - 1)]

    new_df = pd.DataFrame(out)
    log.info("[%s] resampled %d -> %d rows (%.1f Hz -> %.1f Hz, ratio %d/%d)",
             run_name or "run", n_old, len(new_df), src_rate, target_rate, up, down)
    return new_df


def _parse_time_into_export(series: pd.Series) -> pd.Series:
    """Parse a TimeIntoExport-style column to seconds.

    Accepts numeric series (seconds), or strings in ``HH:MM:SS[.mmm]`` /
    ``MM:SS[.mmm]`` / ``SS[.mmm]`` form. Anything unparseable becomes NaN.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    # String → seconds.  Use pandas' timedelta parser, which handles
    # "HH:MM:SS.fff" and "MM:SS.fff" (after prepending "0:") robustly.
    s = series.astype(str).str.strip()
    td = pd.to_timedelta(s, errors="coerce")
    if td.notna().any():
        return td.dt.total_seconds()
    # Last resort: try plain float
    return pd.to_numeric(s, errors="coerce")


def detect_sample_rate(df, default=100.0):
    """Estimate samples-per-second from a loaded run dataframe.

    Tries (in order):
        1. ``tLap`` median diff
        2. ``TimeIntoExport`` (monotonic wall-clock, parsed as seconds
           or ``HH:MM:SS.mmm`` strings) median diff — the most reliable
           source on CAR exports where ``sLap`` is ZOH-held below the
           true grid rate.
        3. ``sLap``+``vCar`` (vCar in km/h)
        4. fallback ``default``
    Returns (rate_hz, source_label).
    """
    if "tLap" in df.columns:
        t = pd.to_numeric(df["tLap"], errors="coerce").dropna()
        if len(t) > 10:
            dt = t.diff().dropna()
            # Filter out lap-reset spikes (negative jumps) before taking median.
            dt = dt[(dt > 0) & (dt < 1.0)]
            if len(dt) > 5:
                med = dt.median()
                if med > 0 and np.isfinite(med):
                    return float(1.0 / med), "tLap"
    if "TimeIntoExport" in df.columns:
        t = _parse_time_into_export(df["TimeIntoExport"]).dropna()
        if len(t) > 10:
            dt = t.diff().dropna()
            dt = dt[(dt > 0) & (dt < 1.0)]
            if len(dt) > 5:
                med = dt.median()
                if med > 0 and np.isfinite(med):
                    return float(1.0 / med), "TimeIntoExport"
    if "sLap" in df.columns and "vCar" in df.columns:
        s = pd.to_numeric(df["sLap"], errors="coerce").dropna()
        v = pd.to_numeric(df["vCar"], errors="coerce").reindex(s.index).dropna()
        if len(v) > 50:
            # vCar in km/h -> m/s; dt = ds / (v * 1000/3600)
            ds = s.diff().dropna()
            v_mps = v.loc[ds.index] / 3.6
            dt = ds / v_mps.replace(0, np.nan)
            dt = dt[(dt > 0) & (dt < 1.0)].dropna()
            if len(dt) > 20:
                med = dt.median()
                if med > 0 and np.isfinite(med):
                    return float(1.0 / med), "sLap+vCar"
    return float(default), "default"

