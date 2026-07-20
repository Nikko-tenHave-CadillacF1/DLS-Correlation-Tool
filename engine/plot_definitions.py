"""Typed dataclasses describing every plot the engine can render.

One class per plot kind (``WaveformPlot``, ``ScatterPlot``, ``PsdPlot`` …).
Each class validates its arguments in ``__post_init__`` and exposes a
``kind`` classvar that the runtime dispatch uses to route the plot definition
to the correct generator. :class:`Marker` is the annotation primitive shared
by every kind that supports vertical guides.

The single source of truth for valid ``BarPlot`` aggregation names lives here
(``_VALID_BAR_AGGS``) — ``engine.datafunctions.normalize_bar_metric_specs``
imports it so the validation and normalisation stay in sync.

Users typically build these via keyword arguments in the top-level
``Run_*.py`` scripts:

    from engine import WaveformPlot, ScatterPlot
    WAVEFORMS = [WaveformPlot(name="Driver Input", channels=("rThrottle", "pBrakeF"))]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence, Union

_VALID_BAR_AGGS = {
    "integral",
    "abs_integral",
    "sum",
    "abs_sum",
    "mean",
    "median",
    "max",
    "min",
    "first",
    "last",
}
_VALID_BOX_MODES = {"per_run", "aggregated", "per_run_aggregated"}
_VALID_GATE_OPS = {">", "<", ">=", "<=", "==", "!=", "between", "outside", "robust"}
_VALID_LEGEND_POS = {"top", "right"}
_VALID_HEATMAP_AGGS = {"mean", "median", "std", "count", "sum", "max", "min"}


def _require_str(value: Any, where: str, allow_blank: bool = False) -> None:
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise TypeError(f"{where}: expected non-empty string, got {value!r}.")


def _require_nonempty(value: Any, where: str) -> None:
    if not value:
        raise ValueError(f"{where}: must not be empty (got {value!r}).")


def _validate_gate(value: Any, where: str) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{where}: gate must be a tuple or list of tuples.")
    if len(value) == 3 and isinstance(value[0], str):
        _validate_one_gate(value, where)
        return
    for i, cond in enumerate(value):
        if not isinstance(cond, (list, tuple)) or len(cond) != 3 or not isinstance(cond[0], str):
            raise TypeError(f"{where}: condition #{i} must be (channel, operator, value); got {cond!r}.")
        _validate_one_gate(cond, f"{where} cond#{i}")


def _validate_one_gate(cond: Sequence[Any], where: str) -> None:
    _, op, _ = cond
    if op not in _VALID_GATE_OPS:
        raise ValueError(f"{where}: unknown gate operator {op!r}. Expected one of {sorted(_VALID_GATE_OPS)}.")


@dataclass
class Marker:
    """Vertical guide annotation shared by every 2-D plot kind.

    A marker is defined either at a fixed axis position (``x``) or by a
    data-dependent boolean ``condition`` that triggers on rising/falling/both
    edges of the referenced series. Exactly one of ``x`` and ``condition``
    must be set.

    Parameters
    ----------
    x : float, optional
        Fixed axis position (in the plot's x-axis units) for a static marker.
    label : str, optional
        Text drawn beside the vertical line. ``None`` renders no label.
    show_label : bool, default True
        Suppress just the label text while still drawing the line.
    color : str, optional
        Matplotlib colour spec; defaults to a neutral grey.
    linestyle : str, default ':'
        Matplotlib line-style spec.
    row : int, optional
        Sub-row index for waveform plots; ``None`` = all rows.
    condition : (channel, op, value) tuple, optional
        Data-triggered marker; e.g. ``("SM", "==", 1)``.
    edge : {'rising', 'falling', 'both'}, default 'rising'
        Edge type on which the ``condition`` fires.
    max_count : int, optional
        Cap on how many condition-triggered markers to draw per run.
    """

    x: float | None = None
    label: str | None = None
    show_label: bool = True
    color: str | None = None
    linestyle: str = ":"
    row: int | None = None
    condition: Any = None
    edge: str = "rising"
    max_count: int | None = None

    def __post_init__(self) -> None:
        has_x = self.x is not None
        has_cond = self.condition is not None
        if has_x == has_cond:
            raise ValueError(
                "Marker requires exactly one of 'x' or 'condition' to be set "
                f"(got x={self.x!r}, condition={self.condition!r})."
            )
        if has_x:
            try:
                self.x = float(self.x)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Marker.x must be numeric, got {self.x!r}.") from exc
        if has_cond:
            if self.edge not in ("rising", "falling", "both"):
                raise ValueError(f"Marker.edge must be 'rising', 'falling', or 'both'. Got {self.edge!r}.")
            if self.max_count is not None:
                try:
                    self.max_count = int(self.max_count)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"Marker.max_count must be int or None, got {self.max_count!r}.") from exc
                if self.max_count <= 0:
                    raise ValueError(f"Marker.max_count must be positive, got {self.max_count}.")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(f"Marker.label must be str or None, got {self.label!r}.")


def _coerce_markers(value: Any, where: str) -> list[Marker]:
    if value is None:
        return []
    if isinstance(value, Marker):
        return [value]
    if isinstance(value, dict):
        return [Marker(**value)]
    if isinstance(value, (list, tuple)):
        out: list[Marker] = []
        for i, item in enumerate(value):
            if isinstance(item, Marker):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Marker(**item))
            elif isinstance(item, (int, float)):
                out.append(Marker(x=float(item)))
            else:
                raise TypeError(f"{where}: marker #{i} must be a Marker, dict, or number; got {item!r}.")
        return out
    raise TypeError(f"{where}: markers must be None, Marker, dict, or list. Got {value!r}.")


def _coerce_flat_reference_lines(value: Any, where: str) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise TypeError(f"{where}: entry #{i} must be a number; got {item!r}.")
            out.append(float(item))
        return out
    raise TypeError(f"{where} must be None, a number, or a list of numbers. Got {value!r}.")


def _coerce_lorentz_fit(value: Any, where: str) -> list[tuple[float, float]] | None:
    def _coerce_window(item: Any, idx_label: str) -> tuple[float, float]:
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, (int, float)) for v in item):
            lo = float(item[0])
            hi = float(item[1])
            if not (0 < lo < hi):
                raise ValueError(f"{where}: {idx_label} must be (f_lo, f_hi) with 0 < f_lo < f_hi.")
            return (lo, hi)
        raise TypeError(f"{where}: {idx_label} must be an (f_lo, f_hi) tuple; got {item!r}.")

    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return [_coerce_window(value, "value")]
    if isinstance(value, list):
        return [_coerce_window(item, f"entry #{i}") for i, item in enumerate(value)]
    raise TypeError(f"{where} must be an (f_lo, f_hi) tuple or list of such tuples. Got {value!r}.")


@dataclass
class WaveformPlot:
    """Time- or lap-domain waveform with one row per channel entry.

    Each entry in ``channels`` becomes one horizontal panel. An entry may be
    a single channel name (single y-axis) or a ``(primary, secondary)`` pair
    (two-y-axis row). The optional ``axis_limits``, ``reference_lines``, and
    ``subplot_heights`` tuples must be the same length as ``channels``.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    channels : tuple
        One entry per row; each entry is a channel name or a
        ``(primary, secondary)`` pair.
    axis_limits : tuple, optional
        Per-row ``(y1_limits, y2_limits)`` tuples.
    reference_lines : tuple, optional
        Per-row lists of horizontal reference y-values.
    subplot_heights : tuple of float, optional
        Per-row relative height ratios.
    x_limits : (float, float), optional
        Shared x-axis limits for every row.
    x_channel : str, default 'sLap'
        Column used as the shared x-axis.
    highlight_zones : Any, optional
        Shaded background zones per row.
    normalise : bool, default False
        If True, normalise each trace to its own peak-absolute value.
    legend_position : {'top', 'right'}, default 'top'
        Placement of the figure-level legend.
    show_delta : bool or tuple of bool, default False
        Per-row toggle to draw a delta-vs-reference-run trace.
    markers : list of Marker, optional
        Vertical guides applied per row (see :class:`Marker.row`).
    annotate_at : number or tuple of numbers, optional
        X-values at which to annotate every trace's amplitude.
    """

    name: str
    channels: tuple[Any, ...]
    axis_limits: tuple[Any, ...] | None = None
    reference_lines: tuple[Any, ...] | None = None
    subplot_heights: tuple[float, ...] | None = None
    x_limits: tuple[float | None, float | None] | None = None
    x_channel: str = "sLap"
    highlight_zones: Any = None
    normalise: bool = False
    legend_position: str = "top"
    show_delta: Union[bool, tuple[bool, ...], list[bool]] = False
    markers: list[Marker] = field(default_factory=list)
    annotate_at: Union[tuple[float, ...], list[float], float] | None = None
    kind: ClassVar[str] = "waveform"

    def __post_init__(self) -> None:
        where = f"WaveformPlot {self.name!r}"
        _require_str(self.name, "WaveformPlot.name")
        _require_nonempty(self.channels, f"{where}.channels")
        if not isinstance(self.channels, (list, tuple)):
            raise TypeError(f"{where}.channels must be tuple/list.")
        if self.legend_position not in _VALID_LEGEND_POS:
            raise ValueError(f"{where}.legend_position must be one of {sorted(_VALID_LEGEND_POS)}.")
        n = len(self.channels)
        for attr in ("axis_limits", "reference_lines", "subplot_heights"):
            value = getattr(self, attr)
            if value is not None and len(value) != n:
                raise ValueError(f"{where}.{attr} has length {len(value)} but channels has {n}.")
        if not isinstance(self.x_channel, str) or not self.x_channel.strip():
            raise TypeError(f"{where}.x_channel must be a non-empty string.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        if isinstance(self.show_delta, bool):
            self.show_delta = tuple(self.show_delta for _ in range(n))
        elif isinstance(self.show_delta, (list, tuple)):
            if len(self.show_delta) != n:
                raise ValueError(f"{where}.show_delta has length {len(self.show_delta)} but channels has {n}.")
            self.show_delta = tuple(bool(v) for v in self.show_delta)
        else:
            raise TypeError(f"{where}.show_delta must be bool or a tuple/list of bools per row.")
        self.normalise = bool(self.normalise)
        if self.annotate_at is not None:
            if isinstance(self.annotate_at, (int, float)):
                self.annotate_at = (float(self.annotate_at),)
            elif isinstance(self.annotate_at, (list, tuple)):
                self.annotate_at = tuple(float(v) for v in self.annotate_at)
            else:
                raise TypeError(f"{where}.annotate_at must be a number or tuple/list of numbers.")


@dataclass
class ScatterPlot:
    """2-D scatter with optional best-fit line and gate filtering.

    ``best_fit`` selects the fit style:

    * ``0`` — no fit,
    * ``1`` — one linear regression across the full data,
    * ``2`` — one quadratic polynomial,
    * list of ``(axis, lo, hi)`` segments — piecewise fits per x-range.

    ``color_gate`` is a 4-tuple ``(channel, op, value, '#hex')`` that recolours
    the subset of points matching the gate to the given colour, keeping the
    rest at the run's base colour.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    x_channel, y_channel : str
        Column names for the two axes.
    axis_limits : list of (float, float), optional
        Per-axis ``(x_limits, y_limits)`` tuples.
    best_fit : int or list of segments, default 0
        Fit selector (see above).
    gate : (channel, op, value) or list of such, optional
        Row filter applied before scatter and fit.
    show_equations : bool, default True
        Emit fit-equation text on the plot.
    show_error : bool, default True
        Show per-segment %-error vs the first run's fit.
    error_as_factor : bool, default False
        Render errors as multiplicative factors (×1.05) instead of %.
    color_gate : 4-tuple, optional
        See above.
    annotate_fit_at : Any, optional
        X-values where fit predictions are annotated.
    markers : list of Marker, optional
        Vertical guides.
    reference_lines : list of float, optional
        Horizontal reference y-values.
    robust : bool, default False
        Use Theil-Sen instead of OLS for fits.
    robust_threshold : float, default 3.0
        Sigma cutoff for the robust outlier flag.
    """

    name: str
    x_channel: str
    y_channel: str
    axis_limits: list[tuple[float | None, float | None]] | None = None
    best_fit: Union[int, list[tuple[str, float | None, float | None]], None] = 0
    gate: Any = None
    show_equations: bool = True
    show_error: bool = True
    error_as_factor: bool = False
    color_gate: Any = None
    annotate_fit_at: Any = None
    markers: list[Marker] = field(default_factory=list)
    reference_lines: list[float] | None = None
    robust: bool = False
    robust_threshold: float = 3.0
    kind: ClassVar[str] = "scatter"

    def __post_init__(self) -> None:
        where = f"ScatterPlot {self.name!r}"
        _require_str(self.name, "ScatterPlot.name")
        _require_str(self.x_channel, f"{where}.x_channel")
        _require_str(self.y_channel, f"{where}.y_channel")
        _validate_gate(self.gate, f"{where}.gate")
        if self.color_gate is not None:
            if (
                not isinstance(self.color_gate, (list, tuple))
                or len(self.color_gate) < 4
                or not isinstance(self.color_gate[0], str)
                or not isinstance(self.color_gate[3], str)
            ):
                raise TypeError(f"{where}.color_gate must be ('channel', 'op', value, '#hexcolor').")
            _validate_one_gate(self.color_gate[:3], f"{where}.color_gate")
        if isinstance(self.best_fit, (list, tuple)):
            for i, seg in enumerate(self.best_fit):
                if not isinstance(seg, (list, tuple)) or len(seg) != 3 or not isinstance(seg[0], str):
                    raise TypeError(f"{where}.best_fit[{i}] must be (axis, lo, hi); got {seg!r}.")
        elif self.best_fit not in (None, 0, 1, 2):
            raise ValueError(f"{where}.best_fit must be 0/1/2/None or a list of segments.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.robust = bool(self.robust)
        self.robust_threshold = float(self.robust_threshold)
        if self.robust_threshold <= 0:
            raise ValueError(f"{where}.robust_threshold must be > 0.")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


@dataclass
class Scatter3DPlot:
    name: str
    x_channel: str
    y_channel: str
    z_channel: str
    gate: Any = None
    axis_limits: list[tuple[float | None, float | None]] | None = None
    kind: ClassVar[str] = "scatter3d"

    def __post_init__(self) -> None:
        where = f"Scatter3DPlot {self.name!r}"
        _require_str(self.name, "Scatter3DPlot.name")
        _require_str(self.x_channel, f"{where}.x_channel")
        _require_str(self.y_channel, f"{where}.y_channel")
        _require_str(self.z_channel, f"{where}.z_channel")
        _validate_gate(self.gate, f"{where}.gate")
        if self.axis_limits is not None:
            if not isinstance(self.axis_limits, (list, tuple)) or len(self.axis_limits) != 3:
                raise ValueError(f"{where}.axis_limits must be a length-3 list of (lo, hi) tuples.")


@dataclass
class PsdPlot:
    """Welch Power Spectral Density plot.

    ``channel`` may be a single string or a list to overlay several channels
    per run. ``nperseg`` may be an integer window size, ``None`` for auto
    selection (see ``engine.datafunctions.auto_nperseg`` — sample-rate-
    aware, respects the plotter's ``PSD_MIN_AVERAGES_TARGET``), or the string
    ``"auto"`` (equivalent to ``None``).

    Optional ``lorentz_fit`` is a list of ``(f0, half_width_hz)`` tuples;
    each pair triggers a single-degree-of-freedom Lorentz fit around that
    peak whose amplitude, damping ratio (ζ), and uncertainty are drawn
    inline on the spectrum.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    channel : str or list of str
        Channel(s) whose PSD is drawn.
    axis_limits : list of (float, float), optional
        ``[(f_min, f_max), (S_min, S_max)]`` limits.
    log_scale : bool, default True
        Semi-log Y axis (spectral density is log-scaled by default).
    nperseg : int, 'auto', or None, optional
        Welch window size; ``None``/``'auto'`` uses the sample-rate-aware policy.
    annotate_at : Any, optional
        Frequencies at which per-curve amplitudes are annotated.
    markers : list of Marker, optional
        Vertical guides.
    gate : gate spec, optional
        Row filter applied before spectrogram segmentation.
    show_envelope : bool, default False
        Draw a max-envelope band per run.
    reference_lines : list of float, optional
        Horizontal reference power values.
    lorentz_fit : list of (f0, half_width_hz), optional
        See above.
    """

    name: str
    channel: Union[str, list[str], tuple[str, ...]]
    axis_limits: list[tuple[float | None, float | None]] | None = None
    log_scale: bool = True
    nperseg: Union[int, str] | None = None
    annotate_at: Any = None
    markers: list[Marker] = field(default_factory=list)
    gate: Any = None
    show_envelope: bool = False
    reference_lines: list[float] | None = None
    lorentz_fit: Any = None
    kind: ClassVar[str] = "psd"

    def __post_init__(self) -> None:
        where = f"PsdPlot {self.name!r}"
        _require_str(self.name, "PsdPlot.name")
        if isinstance(self.channel, (list, tuple)):
            if not self.channel:
                raise ValueError(f"{where}.channel list must not be empty.")
            for ch in self.channel:
                if not isinstance(ch, str) or not ch.strip():
                    raise TypeError(f"{where}: channel entries must be non-empty strings.")
        else:
            _require_str(self.channel, f"{where}.channel")
        if self.nperseg is not None:
            if isinstance(self.nperseg, str):
                if self.nperseg.strip().lower() == "auto":
                    # Reuse existing runtime auto path (None means auto-select).
                    self.nperseg = None
                else:
                    self.nperseg = int(self.nperseg)
            else:
                self.nperseg = int(self.nperseg)
            if self.nperseg is not None and self.nperseg < 8:
                raise ValueError(f"{where}.nperseg must be >= 8.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.log_scale = bool(self.log_scale)
        _validate_gate(self.gate, f"{where}.gate")
        self.show_envelope = bool(self.show_envelope)
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")
        self.lorentz_fit = _coerce_lorentz_fit(self.lorentz_fit, f"{where}.lorentz_fit")


@dataclass
class HistogramPlot:
    """1-D histogram of a single channel across runs.

    All loaded runs are overlaid on the same axes with transparency; the
    per-run colour is taken from the run dict.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    channel : str
        Column to histogram.
    axis_limits : list of (float, float), optional
        ``[(x_min, x_max), (y_min, y_max)]`` limits.
    log_scale : bool, default False
        Log-scale the Y (count) axis.
    markers : list of Marker, optional
        Vertical guides.
    gate : gate spec, optional
        Row filter.
    reference_lines : list of float, optional
        Horizontal reference values.
    """

    name: str
    channel: str
    axis_limits: list[tuple[float | None, float | None]] | None = None
    log_scale: bool = False
    markers: list[Marker] = field(default_factory=list)
    gate: Any = None
    reference_lines: list[float] | None = None
    kind: ClassVar[str] = "histogram"

    def __post_init__(self) -> None:
        where = f"HistogramPlot {self.name!r}"
        _require_str(self.name, "HistogramPlot.name")
        _require_str(self.channel, f"{where}.channel")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.log_scale = bool(self.log_scale)
        _validate_gate(self.gate, f"{where}.gate")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


@dataclass
class BarPlot:
    """Grouped bar chart, one bar per run per metric.

    Each entry in ``metrics`` is either a channel name (aggregated with
    ``default_aggregation``) or a ``(channel, aggregation)`` tuple whose
    aggregation overrides the default. Valid aggregations: ``integral``,
    ``abs_integral``, ``sum``, ``abs_sum``, ``mean``, ``median``, ``max``,
    ``min``, ``first``, ``last``.

    When ``secondary_axis`` is True and one metric's magnitude exceeds
    ``PlotJobConfig.bar_secondary_axis_ratio`` × the smallest, that metric
    is drawn on a right-hand y-axis to prevent scale flattening.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    metrics : tuple
        Channel names or ``(channel, aggregation)`` tuples.
    default_aggregation : str, default 'last'
        Aggregation for entries that don't override it.
    axis_limits : (float, float), optional
        Y-axis ``(y_min, y_max)``.
    gate : gate spec, optional
        Row filter applied before aggregation.
    reference_lines : list of float, optional
        Horizontal reference values.
    error_metrics : tuple, optional
        Per-metric channel names whose values become the errorbar half-
        widths. Length must match ``metrics``; entries may be ``None`` to
        skip errorbars on individual metrics.
    secondary_axis : bool, default True
        Auto-split large-scale-ratio metrics onto a right axis.
    """

    name: str
    metrics: tuple[Any, ...]
    default_aggregation: str = "last"
    axis_limits: tuple[float | None, float | None] | None = None
    gate: Any = None
    reference_lines: list[float] | None = None
    error_metrics: tuple[Any, ...] | None = None
    secondary_axis: bool = True
    kind: ClassVar[str] = "bar"

    def __post_init__(self) -> None:
        where = f"BarPlot {self.name!r}"
        _require_str(self.name, "BarPlot.name")
        _require_nonempty(self.metrics, f"{where}.metrics")
        if self.default_aggregation not in _VALID_BAR_AGGS:
            raise ValueError(f"{where}.default_aggregation must be one of {sorted(_VALID_BAR_AGGS)}.")
        _validate_gate(self.gate, f"{where}.gate")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")
        if self.error_metrics is not None:
            if not isinstance(self.error_metrics, (list, tuple)):
                raise TypeError(f"{where}.error_metrics must be a tuple/list (one entry per metric) or None.")
            if len(self.error_metrics) != len(self.metrics):
                raise ValueError(
                    f"{where}.error_metrics has length {len(self.error_metrics)} but metrics has {len(self.metrics)}."
                )
            for i, em in enumerate(self.error_metrics):
                if em is not None and (not isinstance(em, str) or not em.strip()):
                    raise TypeError(f"{where}.error_metrics[{i}] must be a channel name string or None; got {em!r}.")


@dataclass
class BoxPlot:
    """Box-and-whisker for one or more channels across runs.

    ``aggregation_mode`` selects how boxes are grouped:

    * ``per_run`` — one box per (channel, run),
    * ``aggregated`` — one box per channel with all runs pooled,
    * ``per_run_aggregated`` — both per-run boxes and a pooled overlay.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    channels : str or list of str
        Channels to render. A single string is coerced to a one-element list.
    aggregation_mode : {'per_run', 'aggregated', 'per_run_aggregated'}, default 'per_run'
        See above.
    axis_limits : (float, float), optional
        Y-axis ``(y_min, y_max)``.
    gate : gate spec, optional
        Row filter.
    reference_lines : list of float, optional
        Horizontal reference values.
    options : dict, optional
        Per-plot overrides for the shared ``BOX_PLOT_SETTINGS`` (colours,
        whisker style, outlier appearance, etc.).
    """

    name: str
    channels: Union[str, list[str], tuple[str, ...]]
    aggregation_mode: str = "per_run"
    axis_limits: tuple[float | None, float | None] | None = None
    gate: Any = None
    reference_lines: list[float] | None = None
    options: dict | None = None
    kind: ClassVar[str] = "box"

    def __post_init__(self) -> None:
        where = f"BoxPlot {self.name!r}"
        _require_str(self.name, "BoxPlot.name")
        if self.aggregation_mode not in _VALID_BOX_MODES:
            raise ValueError(f"{where}.aggregation_mode must be one of {sorted(_VALID_BOX_MODES)}.")
        _validate_gate(self.gate, f"{where}.gate")
        if isinstance(self.channels, str):
            self.channels = [self.channels]
        elif isinstance(self.channels, (list, tuple)):
            for ch in self.channels:
                if not isinstance(ch, str) or not ch.strip():
                    raise TypeError(f"{where}: channel entries must be non-empty strings.")
            self.channels = list(self.channels)
        else:
            raise TypeError(f"{where}.channels must be a string or list of strings.")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


_VALID_GRID_RENDER_MODES = {"expand", "grid"}


@dataclass
class BoxPlotGrid:
    """Matrix of :class:`BoxPlot` cells partitioned by row and column gates.

    ``rows`` and ``cols`` are ``{label: gate}`` dicts. For every
    ``(row, col)`` pair the grid expands into a child ``BoxPlot`` whose gate
    is the AND of the row and column gates and whose name is
    ``"{name} - {row_label} {col_label}"``.

    ``render_mode='expand'`` (default) causes :func:`build_plot_groups` to
    call :meth:`expand` and inject the child plots into the ``boxes`` group
    at build time. ``render_mode='grid'`` reserves the entry for future
    single-figure grid rendering (not implemented).

    Parameters
    ----------
    name : str
        Grid title; per-cell names append the row and column labels.
    channels : str or list of str
        Channels shared by every cell.
    rows, cols : dict
        ``{label: gate}`` partitions; each gate follows the same schema as
        :class:`BoxPlot.gate`.
    aggregation_mode : {'per_run', 'aggregated', 'per_run_aggregated'}, default 'per_run'
        Passed to every child BoxPlot.
    axis_limits : (float, float), optional
        Shared Y-axis limits.
    options : dict, optional
        Shared per-cell style overrides.
    render_mode : {'expand', 'grid'}, default 'expand'
        See above.
    """

    name: str
    channels: Union[str, list[str], tuple[str, ...]]
    rows: dict
    cols: dict
    aggregation_mode: str = "per_run"
    axis_limits: tuple[float | None, float | None] | None = None
    options: dict | None = None
    render_mode: str = "expand"
    kind: ClassVar[str] = "box_grid"

    def __post_init__(self) -> None:
        where = f"BoxPlotGrid {self.name!r}"
        _require_str(self.name, "BoxPlotGrid.name")
        if self.render_mode not in _VALID_GRID_RENDER_MODES:
            raise ValueError(
                f"{where}.render_mode must be one of {sorted(_VALID_GRID_RENDER_MODES)}; got {self.render_mode!r}."
            )
        if self.aggregation_mode not in _VALID_BOX_MODES:
            raise ValueError(f"{where}.aggregation_mode must be one of {sorted(_VALID_BOX_MODES)}.")
        if not isinstance(self.rows, dict) or not self.rows:
            raise TypeError(f"{where}.rows must be a non-empty dict.")
        if not isinstance(self.cols, dict) or not self.cols:
            raise TypeError(f"{where}.cols must be a non-empty dict.")
        for label, gate in self.rows.items():
            _validate_gate(gate, f"{where}.rows[{label!r}]")
        for label, gate in self.cols.items():
            _validate_gate(gate, f"{where}.cols[{label!r}]")
        if isinstance(self.channels, str):
            self.channels = [self.channels]
        elif isinstance(self.channels, (list, tuple)):
            for ch in self.channels:
                if not isinstance(ch, str) or not ch.strip():
                    raise TypeError(f"{where}: channel entries must be non-empty strings.")
            self.channels = list(self.channels)
        else:
            raise TypeError(f"{where}.channels must be a string or list of strings.")

    def expand(self) -> list[BoxPlot]:
        plots = []
        for row_label, row_gate in self.rows.items():
            for col_label, col_gate in self.cols.items():
                combined_gate = _normalise_gate_list(row_gate) + _normalise_gate_list(col_gate)
                cell_name = f"{self.name} - {row_label} {col_label}"
                plots.append(
                    BoxPlot(
                        name=cell_name,
                        channels=list(self.channels),
                        aggregation_mode=self.aggregation_mode,
                        axis_limits=self.axis_limits,
                        gate=combined_gate,
                        options=self.options,
                    )
                )
        return plots


def _normalise_gate_list(gate) -> list:
    if gate is None:
        return []
    if isinstance(gate, (list, tuple)) and len(gate) == 3 and isinstance(gate[0], str):
        return [tuple(gate)]
    return [tuple(c) for c in gate]


@dataclass
class HeatmapPlot:
    """2-D binned heatmap with optional per-bin z-aggregation.

    When ``z_channel`` is ``None`` the heatmap shows point density (count per
    bin). When ``z_channel`` is set, each bin's colour reflects the chosen
    aggregation over the z-values inside that bin. Bins with fewer than
    ``min_count`` samples are drawn transparent.

    Parameters
    ----------
    name : str
        Plot title and filename stem.
    x_channel, y_channel : str
        Column names for the two spatial axes.
    z_channel : str, optional
        Column aggregated per bin; ``None`` → density.
    aggregation : {'mean', 'median', 'std', 'count', 'sum', 'max', 'min'}, default 'mean'
        Per-bin aggregator applied to ``z_channel``.
    bins : int or (int, int), default 40
        Global bin count, or ``(x_bins, y_bins)``.
    axis_limits : list of (float, float), optional
        ``[(x_lo, x_hi), (y_lo, y_hi)]`` limits.
    cmap : str, default 'viridis'
        Matplotlib colormap name.
    z_limits : (float, float), optional
        Colorbar clip.
    gate : gate spec, optional
        Row filter.
    markers : list of Marker, optional
        Vertical guides.
    min_count : int, default 3
        Bins with fewer samples are masked out.
    """

    name: str
    x_channel: str
    y_channel: str
    z_channel: str | None = None
    aggregation: str = "mean"
    bins: Union[int, tuple[int, int]] = 40
    axis_limits: list[tuple[float | None, float | None]] | None = None
    cmap: str = "viridis"
    z_limits: tuple[float | None, float | None] | None = None
    gate: Any = None
    markers: list[Marker] = field(default_factory=list)
    min_count: int = 3
    kind: ClassVar[str] = "heatmap"

    def __post_init__(self) -> None:
        where = f"HeatmapPlot {self.name!r}"
        _require_str(self.name, "HeatmapPlot.name")
        _require_str(self.x_channel, f"{where}.x_channel")
        _require_str(self.y_channel, f"{where}.y_channel")
        if self.z_channel is not None and (not isinstance(self.z_channel, str) or not self.z_channel.strip()):
            raise TypeError(f"{where}.z_channel must be a non-empty string or None.")
        if self.aggregation not in _VALID_HEATMAP_AGGS:
            raise ValueError(f"{where}.aggregation must be one of {sorted(_VALID_HEATMAP_AGGS)}.")
        if isinstance(self.bins, int):
            if self.bins < 4:
                raise ValueError(f"{where}.bins must be >= 4.")
        elif isinstance(self.bins, (list, tuple)) and len(self.bins) == 2:
            self.bins = (int(self.bins[0]), int(self.bins[1]))
        else:
            raise TypeError(f"{where}.bins must be int or (int, int); got {self.bins!r}.")
        if self.min_count < 1:
            raise ValueError(f"{where}.min_count must be >= 1.")
        _validate_gate(self.gate, f"{where}.gate")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")


PLOT_TYPE_ORDER: tuple[str, ...] = (
    "waveform",
    "scatter",
    "psd",
    "histogram",
    "bar",
    "box",
    "heatmap",
)
"""Canonical group order — used to index ``DataPlotter.PLOT_DEFINITIONS``."""

_PLOT_KIND_TO_INDEX = {kind: i for i, kind in enumerate(PLOT_TYPE_ORDER)}


def plot_group_index(kind: str) -> int:
    return _PLOT_KIND_TO_INDEX[kind]


PLOT_KIND_BY_DATACLASS = {
    WaveformPlot: "waveform",
    ScatterPlot: "scatter",
    PsdPlot: "psd",
    HistogramPlot: "histogram",
    BarPlot: "bar",
    BoxPlot: "box",
    HeatmapPlot: "heatmap",
}
