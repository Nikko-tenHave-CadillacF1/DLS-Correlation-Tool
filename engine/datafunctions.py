from __future__ import annotations

import difflib

import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from scipy.signal import butter, filtfilt, resample_poly, welch
from scipy.stats import linregress, theilslopes

from .logger import log


def calc_channel(*deps):
    def _wrap(fn):
        try:
            fn.__dls_deps__ = tuple(deps)
        except (AttributeError, TypeError):
            pass
        return fn

    return _wrap


_np_trapezoid = getattr(np, "trapezoid", None) or np.trapz

try:
    from tqdm import tqdm as _tqdm_raw

    def _tqdm(it, **kw):
        import sys

        return _tqdm_raw(it, file=sys.stderr, dynamic_ncols=True, **kw)
except ImportError:

    def _tqdm(iterable, **kwargs):
        return iterable


def _fmt_g(v, sig=3):
    if v == 0:
        return "0"
    raw = f"{v:.{sig}g}"
    if "e" in raw or "E" in raw:
        abs_v = abs(v)
        if 1 <= abs_v < 1_000_000:
            decimals = max(0, sig - len(str(int(abs_v))))
            formatted = f"{v:,.{decimals}f}"
            if decimals > 0:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
    return raw


def _safe_get_config(config_dict, source_type: str, config_name: str) -> dict:
    if config_dict is None:
        return {}
    try:
        return config_dict.get(source_type, {})
    except Exception:
        log.debug("No %s found for %s - skipping.", config_name, source_type.upper())
        return {}


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def convert_yes_no_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    columns_converted = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "object" or dtype.name in ["string", "str"]:
            non_nan = df[col].dropna()
            if len(non_nan) == 0:
                continue
            str_values = [str(x).upper() for x in non_nan if isinstance(x, str)]
            if any(v in ["YES", "NO"] for v in str_values):
                # `copy=` kwarg to `infer_objects` is removed in pandas 3.0
                # (Copy-on-Write handles this automatically). Rely on default.
                df[col] = df[col].astype(str).str.upper().replace({"YES": 1, "NO": 0}).infer_objects()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                columns_converted.append(col)
    if columns_converted:
        log.debug("Converted YES/NO to 1/0 in: %s", ", ".join(columns_converted))
    return df


def sanitize_numeric_series(series: pd.Series) -> pd.Series:
    numeric = _to_numeric_safe(series)
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    numeric = numeric.replace([int64_min, int64_max, -np.inf, np.inf], np.nan)
    return numeric


def apply_channel_mappings(df: pd.DataFrame, channel_mappings: dict | None, source_type: str) -> pd.DataFrame:
    mapping = _safe_get_config(channel_mappings, source_type, "channel mappings")
    if not mapping:
        return df
    rename_dict = {src: tgt for src, tgt in mapping.items() if src in df.columns and tgt not in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
        log.debug("Renamed %d channels for %s", len(rename_dict), source_type.upper())
    return df


def apply_transformations(
    df: pd.DataFrame, source_type: str, channel_transforms: dict | None, missing_warned: set | None = None
) -> pd.DataFrame:
    transforms = _safe_get_config(channel_transforms, source_type, "channel transformations")
    if not transforms:
        return df
    transformed_channels = []
    for channel, func in transforms.items():
        if channel.lower() == "all":
            for col in df.columns:
                df[col] = _to_numeric_safe(df[col])
                df[col] = func(df[col])
            log.debug("Applied 'all' transformations to %s data", source_type.upper())
            continue
        if channel in df.columns:
            df[channel] = _to_numeric_safe(df[channel])
            df[channel] = func(df[channel])
            transformed_channels.append(channel)
        else:
            key = (source_type.upper(), channel)
            if missing_warned is None or key not in missing_warned:
                log.warning("Cannot transform missing channel '%s' for source '%s'.", channel, source_type.upper())
                if missing_warned is not None:
                    missing_warned.add(key)
    log.debug("Applied transformations to %d channels for %s", len(transformed_channels), source_type.upper())
    return df


def apply_calculated_channels(
    df: pd.DataFrame, source_type: str, calculated_channels: dict | None, required_channels: set | None = None
) -> pd.DataFrame:
    if calculated_channels is None:
        return df
    if isinstance(calculated_channels, dict) and source_type in calculated_channels:
        calc_set = calculated_channels[source_type]
    else:
        calc_set = calculated_channels
    if not isinstance(calc_set, dict):
        return df
    target_names = None
    if required_channels is not None:
        target_names = {n for n in required_channels if n in calc_set}
        if target_names:
            try:
                import inspect as _inspect
                import re as _re

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
    # Accumulate new columns in a dict and concat in batches to avoid
    # the PerformanceWarning about a fragmented DataFrame caused by repeated
    # single-column inserts (`df[name] = ...`).
    new_cols: dict[str, pd.Series] = {}
    working = df
    BATCH = 50

    def _flush(target, pending):
        if not pending:
            return target, {}
        extra = pd.DataFrame(pending, index=target.index)
        return pd.concat([target, extra], axis=1), {}

    for channel_name, func in calc_set.items():
        is_required = (target_names is None) or (channel_name in target_names)
        try:
            new_cols[channel_name] = _to_numeric_safe(func(working))
            calculated_channels_done.append(channel_name)
            # Rebuild lightweight view so subsequent calcs see prior new cols.
            working = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False)
            if len(new_cols) >= BATCH:
                df, new_cols = _flush(df, new_cols)
                working = df
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
    df, _ = _flush(df, new_cols)
    log.debug("Added %d calculated channels for %s", len(calculated_channels_done), source_type.upper())
    return df


def _apply_butterworth_filter_to_data(data, cutoff, order: int, sample_rate: float, btype: str = "low") -> tuple:
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
    mask_nan = np.isnan(data)
    if mask_nan.any():
        interp = pd.Series(data).interpolate("linear", limit_direction="both").values
        filtered_data = filtfilt(b, a, interp)
        filtered_data[mask_nan] = np.nan
    else:
        filtered_data = filtfilt(b, a, data)
    return filtered_data, True


def apply_filters(
    df: pd.DataFrame, filters: dict | None, sample_rate: float, source_type: str, required_channels: set | None = None
) -> pd.DataFrame:
    if not filters:
        return df
    applied = []
    channels_to_skip = []
    filter_all = "all" in filters
    all_cfg = filters.get("all", None)
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
        if isinstance(cutoff, (int, float)) and cutoff <= 0:
            continue
        filtered_data, success = _apply_butterworth_filter_to_data(
            df[channel].values,
            cutoff,
            order,
            sample_rate,
            btype=btype,
        )
        if not success:
            log.warning("Filter failed for channel '%s' (type=%s). Skipping.", channel, btype)
            continue
        df[channel] = filtered_data
        label = f"{channel}@{cutoff}Hz({btype})" if btype != "low" else f"{channel}@{cutoff}Hz"
        applied.append(label)
    if filter_all and all_cfg:
        cutoff = all_cfg.get("cutoff", 0)
        order = all_cfg.get("order", 2)
        btype = all_cfg.get("type", "low")
        if isinstance(cutoff, (int, float)) and cutoff <= 0:
            pass
        else:
            for col in df.columns:
                if col in channels_to_skip:
                    continue
                df[col] = _to_numeric_safe(df[col])
                filtered_data, success = _apply_butterworth_filter_to_data(
                    df[col].values,
                    cutoff,
                    order,
                    sample_rate,
                    btype=btype,
                )
                if success:
                    df[col] = filtered_data
                    label = f"{col}@{cutoff}Hz({btype})" if btype != "low" else f"{col}@{cutoff}Hz"
                    applied.append(label)
    if applied:
        log.debug("Applied %d filters for %s", len(applied), source_type.upper())
    return df


def calculate_psd(signal, sample_rate: float, nperseg: int = 512) -> tuple[np.ndarray | None, np.ndarray | None]:
    series = _to_numeric_safe(pd.Series(signal)).dropna()
    series = np.asarray(series, dtype=float)
    if len(series) < 8:
        return None, None
    nperseg = min(nperseg, len(series))
    if nperseg < 8:
        return None, None
    freq, power = welch(series, fs=sample_rate, nperseg=nperseg)
    return freq, power


def auto_nperseg(
    n_samples: int,
    sample_rate: float = 100.0,
    min_averages: int = 50,
    min_averages_target: int = 200,
    max_nperseg: int = 4096,
    target_resolution_hz: float = 0.1,
) -> int:
    """Choose a Welch nperseg for a given signal length.

    Three ceilings apply (nperseg cannot exceed any of them):
    - ``avg_floor_limit``: keeps at least ``min_averages`` segment averages
      (hard floor on statistical robustness).
    - ``max_nperseg``: absolute upper bound.
    - ``res_limit``: keeps Delta_f no finer than ``target_resolution_hz``
      (prevents wasting data on unnecessary frequency resolution).

    On top of that a soft preference: if ``min_averages_target > min_averages``
    and the data supports it, shrink ``nperseg`` below ``res_limit`` to reach
    ``min_averages_target`` averages. This buys tighter CIs on medium-length
    sessions (~2-8 minutes) that would otherwise sit at K=50-100 with the
    coarser resolution cap. Long sessions and very short sessions are
    unaffected (already at or below the target). ``min_averages_target`` is
    NEVER allowed to drop nperseg below 64 or below the hard floor cap.
    """
    if min_averages > 0:
        avg_floor_limit = int(2 * n_samples / (min_averages + 1))
    else:
        avg_floor_limit = max_nperseg
    if target_resolution_hz > 0 and sample_rate > 0:
        res_limit = int(sample_rate / target_resolution_hz)
    else:
        res_limit = max_nperseg
    limit = min(avg_floor_limit, max_nperseg, res_limit)
    if min_averages_target > max(min_averages, 0):
        avg_target_limit = int(2 * n_samples / (min_averages_target + 1))
        if avg_target_limit >= 64:
            limit = min(limit, avg_target_limit)
    if limit < 64:
        return 64
    return int(limit)


def calculate_segmented_psd(
    signal,
    mask,
    sample_rate: float,
    nperseg: int = 512,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    signal = np.asarray(signal, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if signal.size == 0 or mask.size != signal.size:
        return None, None, 0
    mask = mask & np.isfinite(signal)
    if not mask.any():
        return None, None, 0
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    noverlap = nperseg // 2
    step = nperseg - noverlap
    freq_ref = None
    weighted_sum = None
    total_segments = 0
    for s, e in zip(starts, ends):
        seg = signal[s:e]
        if len(seg) < nperseg:
            continue
        n_welch = 1 + (len(seg) - nperseg) // step
        freq, power = welch(seg, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        if freq_ref is None:
            freq_ref = freq
            weighted_sum = power.astype(float) * n_welch
        else:
            weighted_sum = weighted_sum + power.astype(float) * n_welch
        total_segments += n_welch
    if total_segments == 0 or weighted_sum is None:
        return None, None, 0
    return freq_ref, weighted_sum / total_segments, total_segments


def mask_waveform_discontinuities(x_values, y_values):
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
    units = ""
    if units_map:
        for key, value in units_map.items():
            if key.lower() == channel.lower():
                units = value
                break
    if units:
        return f"{channel} PSD (${units}^2$/Hz)"
    return f"{channel} PSD"


def compute_nice_histogram_bins(data, num_bins=30):
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
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = nice_fraction * (10**exponent)
    if step >= 1:
        step = max(1.0, float(np.round(step)))
    start = np.floor(data_min / step) * step
    end = np.ceil(data_max / step) * step
    bins = np.arange(start, end + step * 0.5, step)
    if bins.size < 2:
        bins = np.array([start, start + step])
    return bins


def compute_equal_width_bins_in_limits(xmin, xmax, reference_bins):
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


def normalize_bar_metric_specs(metric_specs, default_aggregation="last"):
    if isinstance(metric_specs, str):
        metric_specs = (metric_specs,)
    if not isinstance(metric_specs, (list, tuple)):
        return []
    normalized = []
    try:
        from .plot_definitions import _VALID_BAR_AGGS as valid_aggs
    except Exception:
        valid_aggs = {
            "sum",
            "mean",
            "min",
            "max",
            "median",
            "integral",
            "abs_sum",
            "abs_integral",
            "first",
            "last",
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
    return float(values.iloc[-1])


_DECIMATE_CACHE = {}
_DECIMATE_CACHE_MAX_ENTRIES = 64


def _decimate_xy(x_data, y_data, max_points):
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
    if len(_DECIMATE_CACHE) >= _DECIMATE_CACHE_MAX_ENTRIES:
        try:
            _DECIMATE_CACHE.pop(next(iter(_DECIMATE_CACHE)))
        except StopIteration:
            pass
    _DECIMATE_CACHE[cache_key] = out
    return out


def _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=45000):
    x_plot, y_plot = _decimate_xy(x_data, y_data, max_points=max_points)
    ax.scatter(x_plot, y_plot, alpha=alpha, s=size, color=color, label=label, edgecolors="none")


def plot_scatter(ax, x_data, y_data, label, color, alpha, size, x_var="", y_var="", max_points=45000):
    if len(x_data) == 0:
        log.warning("No data for scatter: %s (%s vs %s).", label, x_var, y_var)
        return False, None, None
    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)
    return True, None, None


def _plot_scatter_fit_line(ax, x_values, y_values, color, linestyle="-", linewidth=1.8):
    line = ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth + 0.4,
        alpha=0.99,
        zorder=7,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 2.4, foreground=(1.0, 1.0, 1.0, 0.96)), pe.Normal()])
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
    if len(x_data) == 0:
        log.warning("No data for single fit: %s (%s vs %s).", label, x_var, y_var)
        return False, None, None, None, None
    if robust:
        info = fit_robust_theilsen(x_data, y_data, outlier_k=robust_threshold)
        if info is None:
            log.warning("Robust fit failed: %s (%s vs %s).", label, x_var, y_var)
            return False, None, None, None, None
        inlier_mask = ~info["outlier_mask"]
        _plot_scatter_layer(
            ax,
            np.asarray(x_data)[inlier_mask],
            np.asarray(y_data)[inlier_mask],
            label,
            color,
            alpha,
            size,
            max_points=max_points,
        )
        if info["n_outliers"] > 0:
            ax.scatter(
                np.asarray(x_data)[info["outlier_mask"]],
                np.asarray(y_data)[info["outlier_mask"]],
                s=max(size * 0.7, 8),
                marker="x",
                color="#9A9A9A",
                alpha=0.25,
                linewidth=0.8,
                zorder=1,
                label="_nolegend_",
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
            f"   [robust: {info['n_outliers']}/{info['n_total']} outliers]" if info["n_outliers"] > 0 else "   [robust]"
        )
        equation = f"$y = {_fmt_g(slope)}x {sign} {_fmt_g(abs(interc))}${suffix}"
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
    equation = f"$y = {_fmt_g(slope)}x {sign} {_fmt_g(abs(interc))}$"
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
    if len(x_data) == 0:
        log.warning("No data for multi-fit: %s (%s vs %s).", label, x_var, y_var)
        return False, None, None, None, None
    if not fit_defs:
        return plot_scatter_with_1fit(
            ax,
            x_data,
            y_data,
            label,
            color,
            alpha,
            size,
            x_var,
            y_var,
            max_points=max_points,
            robust=robust,
            robust_threshold=robust_threshold,
        )
    _plot_scatter_layer(ax, x_data, y_data, label, color, alpha, size, max_points=max_points)
    slopes_list = []
    intercepts_list = []
    eq_lines = []
    line_styles = ["-", "-", "-"]
    total_outliers = 0
    total_points = 0

    def _format_bound(value, is_lower=True, fallback=None):
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
                ax,
                x_data,
                y_data,
                label,
                color,
                alpha,
                size,
                x_var,
                y_var,
                max_points=max_points,
            )
        if fit_mask_info["status"] == "missing_condition_channel":
            log.warning(
                "No fit condition channel %r for run %r. Segment skipped.",
                fit_mask_info["axis_name"], label,
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
                log.warning(
                    "Robust fit segment %d failed for %r (%s vs %s). Skipping segment.",
                    idx + 1, label, x_var, y_var,
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
                    xb[info["outlier_mask"]],
                    yb[info["outlier_mask"]],
                    s=max(size * 0.7, 8),
                    marker="x",
                    color="#9A9A9A",
                    alpha=0.25,
                    linewidth=0.8,
                    zorder=1,
                    label="_nolegend_",
                )
        else:
            try:
                slope, interc, _, _, _ = linregress(xb, yb)
            except ValueError:
                log.warning(
                    "Not enough data for fit segment %d of %r (%s vs %s). Skipping segment.",
                    idx + 1, label, x_var, y_var,
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
        seg_eq = f"{axis_name} $\\in$ [{lo}, {hi}]   $y = {_fmt_g(slope)}x {eq_sign} {_fmt_g(abs(interc))}$"
        if robust and seg_outlier_count > 0:
            seg_eq += f"   ({seg_outlier_count} outliers rejected)"
        eq_lines.append(seg_eq)
        slopes_list.append(slope)
        intercepts_list.append(interc)
    if not eq_lines:
        return False, tuple(slopes_list), tuple(intercepts_list), None, color
    meta = {
        "color": color,
        "robust_info": ({"n_outliers": total_outliers, "n_total": total_points} if robust else None),
    }
    return True, tuple(slopes_list), tuple(intercepts_list), "\n".join(eq_lines), meta


def collect_multi_fit_condition_channels(fit_defs):
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
    condition_channels = collect_multi_fit_condition_channels(fit_defs)
    data = {}
    for channel in condition_channels:
        if channel not in df.columns:
            log.warning(
                "Scatter plot %r: fit condition channel %r missing in run %r.",
                plot_name, channel, run_name,
            )
            continue
        series = pd.to_numeric(df[channel], errors="coerce").reindex(index)
        data[channel] = series.to_numpy(dtype=float)
    return data


def compute_gate_mask(df: pd.DataFrame, gate_spec) -> pd.Series:
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
        if operator == ">":
            gate_mask = col > value
        elif operator == "<":
            gate_mask = col < value
        elif operator == ">=":
            gate_mask = col >= value
        elif operator == "<=":
            gate_mask = col <= value
        elif operator == "==":
            gate_mask = col == value
        elif operator == "!=":
            gate_mask = col != value
        elif operator == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            gate_mask = pd.Series(True, index=col.index)
            if low is not None:
                gate_mask &= col >= low
            if high is not None:
                gate_mask &= col <= high
        elif operator == "outside" and isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            if low is not None and high is not None:
                gate_mask = (col < low) | (col > high)
            elif low is not None:
                gate_mask = col < low
            elif high is not None:
                gate_mask = col > high
            else:
                gate_mask = pd.Series(False, index=col.index)
        elif operator == "robust":
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
                    gate_mask = col == med
                else:
                    gate_mask = (col - med).abs() <= (k * mad)
        else:
            log.warning("Unsupported gate condition for channel '%s'.", channel)
            return pd.Series(False, index=df.index)
        mask &= gate_mask.fillna(False)
    return mask


def apply_gate_to_dataframe(df: pd.DataFrame, gate_spec) -> pd.DataFrame:
    if gate_spec is None:
        return df
    mask = compute_gate_mask(df, gate_spec)
    if not mask.any():
        return df.iloc[0:0].copy()
    return df[mask].copy()


def resolve_condition_marker(marker, df, x_channel):
    if marker.condition is None:
        return []
    if x_channel not in df.columns:
        return []
    mask = compute_gate_mask(df, marker.condition).astype(bool).to_numpy()
    if mask.size < 2:
        return []
    diff = np.diff(mask.astype(np.int8))
    if marker.edge == "rising":
        idx = np.flatnonzero(diff == 1) + 1
    elif marker.edge == "falling":
        idx = np.flatnonzero(diff == -1) + 1
    else:
        idx = np.flatnonzero(diff != 0) + 1
    if idx.size == 0:
        return []
    x_values = pd.to_numeric(df[x_channel], errors="coerce").to_numpy()
    x_hits = [float(x_values[i]) for i in idx if i < len(x_values) and np.isfinite(x_values[i])]
    if marker.max_count is not None and len(x_hits) > marker.max_count:
        x_hits = x_hits[: marker.max_count]
    return x_hits


def aggregate_channel_for_boxplot(
    run_data_dict,
    channels,
    aggregation_mode="per_run",
    gate_spec=None,
    filtered_run_data=None,
):
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
    if aggregation_mode == "per_run":
        for run_name, df in filtered_run_data.items():
            run_dict = {}
            for channel in channels:
                if channel in df.columns:
                    values = df[channel].dropna().values
                    run_dict[channel] = values
                else:
                    run_dict[channel] = np.array([])
            result[run_name] = run_dict
    elif aggregation_mode == "aggregated":
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
        condition_values >= min_bound if (min_val is None or use_inclusive_bounds) else condition_values > min_bound
    )
    upper_mask = (
        condition_values <= max_bound if (max_val is None or use_inclusive_bounds) else condition_values < max_bound
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


def add_units_to_label(var_name: str, units_map: dict):
    key = var_name.lower()
    for k, v in units_map.items():
        if k.lower() == key:
            return f"{var_name} ({v})"
    return var_name


def _running_std(values):
    valid = np.isfinite(values)
    counts = np.cumsum(valid)
    clean = np.where(valid, values, 0.0)
    sums = np.cumsum(clean)
    sumsq = np.cumsum(clean * clean)
    out = np.full(len(values), np.nan)
    ok = counts > 1
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    variance = (sumsq - counts * mean * mean) / np.maximum(counts - 1, 1)
    out[ok] = np.sqrt(np.maximum(variance[ok], 0.0))
    return out


def calculate_cplv(df, axle, sample_rate=100.0, highpass_freq=2.0, highpass_order=4):
    fz_cols = ("FzTyreFL", "FzTyreFR", "FzTyreRL", "FzTyreRR")
    for col in fz_cols:
        if col not in df.columns:
            raise KeyError(col)
    throttle_col = next((c for c in ("rThrottle", "rThrottlePedal") if c in df.columns), None)
    if throttle_col is None:
        raise KeyError("rThrottle")
    lap_col = next((c for c in ("nLap", "NLap", "_nLap") if c in df.columns), None)
    time_col = next((c for c in ("tLap", "sLap") if c in df.columns), None)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    min_samples = 3 * (highpass_order + 1) + 1
    fs = float(sample_rate or 100.0)
    if highpass_freq >= fs / 2:
        return result
    groups = df.groupby(lap_col, sort=False) if lap_col else [(None, df)]
    for _, group in groups:
        lap = group.sort_values(time_col) if time_col else group
        if len(lap) < min_samples:
            continue
        hp = {}
        usable = np.ones(len(lap), dtype=bool)
        for col in fz_cols:
            values = pd.to_numeric(lap[col], errors="coerce")
            usable &= values.notna().to_numpy()
            filled = values.interpolate("linear", limit_direction="both")
            if filled.notna().sum() < min_samples:
                break
            filtered, ok = _apply_butterworth_filter_to_data(
                filled.to_numpy(dtype=float), highpass_freq, highpass_order, fs, btype="high"
            )
            if not ok:
                break
            hp[col] = filtered
        else:
            throttle = pd.to_numeric(lap[throttle_col], errors="coerce").to_numpy(dtype=float)
            gls = (throttle < 98.0) & usable
            if axle == "front":
                vals = _running_std(np.where(gls, hp["FzTyreFL"], np.nan)) + _running_std(
                    np.where(gls, hp["FzTyreFR"], np.nan)
                )
            else:
                vals = _running_std(np.where(gls, hp["FzTyreRL"], np.nan)) + _running_std(
                    np.where(gls, hp["FzTyreRR"], np.nan)
                )
            result.loc[lap.index] = pd.Series(vals, index=lap.index).ffill()
    return result


def _normalize_gate_conditions(gate_spec):
    if gate_spec is None:
        return []
    if isinstance(gate_spec, (list, tuple)) and len(gate_spec) == 3 and isinstance(gate_spec[0], str):
        return [gate_spec]
    return list(gate_spec)


def collect_gate_channels(gate_spec):
    channels = set()
    if gate_spec is None:
        return channels
    conditions = _normalize_gate_conditions(gate_spec)
    for condition in conditions:
        if isinstance(condition, (list, tuple)) and len(condition) == 3 and isinstance(condition[0], str):
            channels.add(condition[0])
    return channels


def format_gate_text(gate_spec):
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


def suggest_similar_channels(target, available, max_results=5, cutoff=0.5):
    if not target or not available:
        return []
    target_lc = target.lower()
    available_list = list(available)
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
    fuzzy_hits = difflib.get_close_matches(target, available_list, n=max_results, cutoff=cutoff)
    seen = set()
    merged = []
    for ch in substring_hits + fuzzy_hits:
        if ch not in seen and ch.lower() != target_lc:
            seen.add(ch)
            merged.append(ch)
        if len(merged) >= max_results:
            break
    return merged


def fit_robust_theilsen(x, y, outlier_k=3.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    if xf.size < 3:
        return None
    slope, intercept, lo, hi = theilslopes(yf, xf, 0.95)
    resid = yf - (slope * xf + intercept)
    mad_r = np.median(np.abs(resid - np.median(resid))) * 1.4826
    if mad_r > 0 and np.isfinite(mad_r):
        outlier_mask_f = np.abs(resid) > outlier_k * mad_r
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


def _rational_ratio(target: float, src: float, max_denom: int = 1000) -> tuple[int, int]:
    from math import gcd

    up = max(1, int(round(target * 1000)))
    down = max(1, int(round(src * 1000)))
    g = gcd(up, down) or 1
    up //= g
    down //= g
    if up > max_denom or down > max_denom:
        up = max(1, int(round(target)))
        down = max(1, int(round(src)))
        g = gcd(up, down) or 1
        up //= g
        down //= g
    return up, down


def resample_to_uniform_rate(
    df: pd.DataFrame, target_rate: float, time_col: str = "tLap", run_name: str | None = None
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        target_rate = float(target_rate)
    except (TypeError, ValueError):
        return df
    if target_rate <= 0:
        return df
    candidate_cols = []
    if time_col:
        candidate_cols.append(time_col)
    for fallback_col in ("tLap", "TimeIntoExport"):
        if fallback_col not in candidate_cols:
            candidate_cols.append(fallback_col)
    chosen_col = None
    t = None
    for col in candidate_cols:
        if col not in df.columns:
            continue
        if col == "TimeIntoExport":
            t_candidate = _parse_time_into_export(df[col]).to_numpy(dtype=float)
        else:
            t_candidate = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t_candidate).sum() >= 5:
            chosen_col = col
            t = t_candidate
            break
    if chosen_col is None or t is None:
        log.debug("resample: skipped (no usable time column among %s) for %s", candidate_cols, run_name or "<run>")
        return df
    n_old = len(df)
    if n_old < 4 or not np.isfinite(t).any():
        return df
    dt = np.diff(t)
    valid_dt = (dt > 0) & (dt < 1.0) & np.isfinite(dt)
    if valid_dt.sum() < 5:
        log.debug("resample: skipped (irregular '%s') for %s", chosen_col, run_name or "<run>")
        return df
    med_dt = float(np.median(dt[valid_dt]))
    if not np.isfinite(med_dt) or med_dt <= 0:
        return df
    src_rate = 1.0 / med_dt
    if abs(src_rate - target_rate) / target_rate < 0.005:
        log.debug("resample: %s already at %.2f Hz (target %.2f) — skipped", run_name or "<run>", src_rate, target_rate)
        return df
    up, down = _rational_ratio(target_rate, src_rate)
    if up == down:
        return df
    n_new = int(np.floor(n_old * up / down))
    if n_new < 2:
        return df
    _LINEAR_INTERP_COLS = {"tlap", "slap", "timeintoexport"}
    _NEAREST_COLS = {"nlap"}
    idx_old = np.arange(n_old)
    t_src_uniform = idx_old / src_rate
    t_new_uniform = np.arange(n_new) / target_rate
    finite_t = t[np.isfinite(t)]
    t0 = float(finite_t[0]) if finite_t.size else 0.0
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
                valid = ~nan_mask
                y = np.interp(idx_old, idx_old[valid], y[valid])
            if col_key in _LINEAR_INTERP_COLS:
                y_new = np.interp(t_new_uniform, t_src_uniform, y)
                out[col] = y_new
                continue
            if col_key in _NEAREST_COLS:
                idx_new = np.clip(np.round(t_new_uniform * src_rate).astype(int), 0, n_old - 1)
                out[col] = y[idx_new]
                continue
            try:
                y_new = resample_poly(y, up, down, padtype="line")
            except Exception as exc:  # pragma: no cover - extreme edge cases
                log.warning("resample: %s column '%s' fell back to linear (%s)", run_name or "<run>", col, exc)
                y_new = np.interp(t_new_uniform, t_src_uniform, y)
            if len(y_new) != n_new:
                if len(y_new) > n_new:
                    y_new = y_new[:n_new]
                else:
                    y_new = np.concatenate((y_new, np.full(n_new - len(y_new), y_new[-1] if len(y_new) else 0.0)))
            out[col] = y_new
        else:
            if col == chosen_col:
                out[col] = t_new_uniform + t0
                continue
            if n_new == 1:
                idx_new = np.array([0])
            else:
                idx_new = np.round(np.linspace(0, n_old - 1, n_new)).astype(int)
            out[col] = s.to_numpy()[np.clip(idx_new, 0, n_old - 1)]
    new_df = pd.DataFrame(out)
    log.info(
        "[%s] resampled %d -> %d rows (%.1f Hz -> %.1f Hz, ratio %d/%d)",
        run_name or "run",
        n_old,
        len(new_df),
        src_rate,
        target_rate,
        up,
        down,
    )
    return new_df


def _parse_time_into_export(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    td = pd.to_timedelta(s, errors="coerce")
    if td.notna().any():
        return td.dt.total_seconds()
    return pd.to_numeric(s, errors="coerce")


def detect_sample_rate(df, default=100.0):
    if "tLap" in df.columns:
        t = pd.to_numeric(df["tLap"], errors="coerce").dropna()
        if len(t) > 10:
            dt = t.diff().dropna()
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
            ds = s.diff().dropna()
            v_mps = v.loc[ds.index] / 3.6
            dt = ds / v_mps.replace(0, np.nan)
            dt = dt[(dt > 0) & (dt < 1.0)].dropna()
            if len(dt) > 20:
                med = dt.median()
                if med > 0 and np.isfinite(med):
                    return float(1.0 / med), "sLap+vCar"
    return float(default), "default"
