"""Typed plot definitions.

Each plot type is a frozen-ish dataclass with ``__post_init__`` validation
so configuration errors are surfaced immediately at workflow startup rather
than deep inside matplotlib. Field semantics are documented inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union

# --- Allowed value sets -------------------------------------------------------

_VALID_BAR_AGGS = {
    "integral", "abs_integral", "sum", "abs_sum",
    "mean", "median", "max", "min", "first", "last",
}
_VALID_BOX_MODES = {"per_run", "aggregated", "per_run_aggregated"}
_VALID_GATE_OPS = {">", "<", ">=", "<=", "==", "!=", "between", "outside", "robust"}
_VALID_LEGEND_POS = {"top", "right"}
_VALID_HEATMAP_AGGS = {"mean", "median", "std", "count", "sum", "max", "min"}


# --- Validation helpers -------------------------------------------------------


def _require_str(value: Any, where: str, allow_blank: bool = False) -> None:
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise TypeError(f"{where}: expected non-empty string, got {value!r}.")


def _require_nonempty(value: Any, where: str) -> None:
    if not value:
        raise ValueError(f"{where}: must not be empty (got {value!r}).")


def _validate_gate(value: Any, where: str) -> None:
    """Accept None, a single 3-tuple gate, or a list/tuple of 3-tuple gates."""
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{where}: gate must be a tuple or list of tuples.")
    # Single condition
    if len(value) == 3 and isinstance(value[0], str):
        _validate_one_gate(value, where)
        return
    # List of conditions
    for i, cond in enumerate(value):
        if not isinstance(cond, (list, tuple)) or len(cond) != 3 or not isinstance(cond[0], str):
            raise TypeError(
                f"{where}: condition #{i} must be (channel, operator, value); got {cond!r}."
            )
        _validate_one_gate(cond, f"{where} cond#{i}")


def _validate_one_gate(cond: Sequence[Any], where: str) -> None:
    _, op, _ = cond
    if op not in _VALID_GATE_OPS:
        raise ValueError(
            f"{where}: unknown gate operator {op!r}. "
            f"Expected one of {sorted(_VALID_GATE_OPS)}."
        )


# ---------------------------------------------------------------------------
# Markers (generalised annotation primitive — used across plot types)
# ---------------------------------------------------------------------------


@dataclass
class Marker:
    """A vertical reference line on a plot.

    Two modes:
      * **Static** — supply ``x`` directly. Drawn once at that x-value, on every
        plot type that supports markers (waveform / scatter / psd / histogram /
        heatmap). Uses ``color`` if given, otherwise neutral grey.
      * **Condition-triggered** — supply ``condition`` (same format as
        ``ScatterPlot.gate``: ``('channel', 'op', value)`` or a list of those
        AND-ed together). Resolved **per run** by the waveform generator: the
        rising/falling edges of the boolean condition are detected and one
        marker is emitted at the x-channel value of each transition. Drawn in
        the run's colour unless ``color`` is set explicitly. Other plot types
        ignore condition markers (their x-axis is not a time/distance series).

    Fields:
      x:           x-value for static markers. Mutually exclusive with ``condition``.
      condition:   gate spec (tuple or list of tuples) for condition markers.
      edge:        which transition triggers a marker: 'rising' (False\u2192True),
                   'falling' (True\u2192False), or 'both'. Default 'rising'.
      max_count:   cap the number of markers emitted per run (most recent N
                   transitions are kept). None = unlimited.
      label:       optional text label drawn at the top.
      show_label:  if False, suppress drawing the label text (the line is
                   still drawn). Useful to reduce clutter when the condition
                   being annotated is obvious from context. Default True.
      color:       hex colour. ``None`` \u2192 grey for static markers, run colour
                   for condition markers.
      linestyle:   matplotlib linestyle (default dotted ':' so it visually
                   distinguishes from reference lines).
      row:         waveform-only \u2014 row index to limit the marker to;
                   ``None`` draws on every row.
    """

    x: Optional[float] = None
    label: Optional[str] = None
    show_label: bool = True
    color: Optional[str] = None
    linestyle: str = ":"
    row: Optional[int] = None
    condition: Any = None
    edge: str = "rising"
    max_count: Optional[int] = None

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
                raise ValueError(
                    f"Marker.edge must be 'rising', 'falling', or 'both'. Got {self.edge!r}."
                )
            if self.max_count is not None:
                try:
                    self.max_count = int(self.max_count)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"Marker.max_count must be int or None, got {self.max_count!r}."
                    ) from exc
                if self.max_count <= 0:
                    raise ValueError(
                        f"Marker.max_count must be positive, got {self.max_count}."
                    )
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(f"Marker.label must be str or None, got {self.label!r}.")


def _coerce_markers(value: Any, where: str) -> List[Marker]:
    """Accept None, a single Marker, a dict, a tuple/list of any of those."""
    if value is None:
        return []
    if isinstance(value, Marker):
        return [value]
    if isinstance(value, dict):
        return [Marker(**value)]
    if isinstance(value, (list, tuple)):
        out: List[Marker] = []
        for i, item in enumerate(value):
            if isinstance(item, Marker):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Marker(**item))
            elif isinstance(item, (int, float)):
                out.append(Marker(x=float(item)))
            else:
                raise TypeError(
                    f"{where}: marker #{i} must be a Marker, dict, or number; got {item!r}."
                )
        return out
    raise TypeError(f"{where}: markers must be None, Marker, dict, or list. Got {value!r}.")


def _coerce_flat_reference_lines(value: Any, where: str) -> Optional[List[float]]:
    """Accept None, a single number, or an iterable of numbers; return a list.

    Used by 2-D plot types (scatter, PSD, histogram, bar, box) whose
    ``reference_lines`` field is a flat list of y-axis benchmark values
    drawn as horizontal dashed lines. WaveformPlot uses a different
    per-row schema and does NOT use this helper.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"{where}: entry #{i} must be a number; got {item!r}."
                )
            out.append(float(item))
        return out
    raise TypeError(f"{where} must be None, a number, or a list of numbers. Got {value!r}.")


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------


@dataclass
class WaveformPlot:
    """Stacked time/distance traces, one subplot row per channel.

    Channel rows may be a single 'ch' or a ('left', 'right') overlay.

    The ``show_delta`` field controls per-row delta subplots. When 2+ runs
    are loaded, a half-height row showing each non-reference run minus the
    reference run is appended below the primary row. Accepts:
      - ``False`` — no delta rows (default)
      - ``True`` — delta row below every channel row
      - tuple/list of bools — per-row control; length must match ``channels``

    The reference run is configured workflow-wide on the run entry itself
    (``{"name": ..., "reference": True, ...}``). If no run is flagged, the
    first loaded run is used.
    """

    name: str
    channels: Tuple[Any, ...]
    axis_limits: Optional[Tuple[Any, ...]] = None
    reference_lines: Optional[Tuple[Any, ...]] = None
    subplot_heights: Optional[Tuple[float, ...]] = None
    x_limits: Optional[Tuple[Optional[float], Optional[float]]] = None
    x_channel: str = "sLap"
    highlight_zones: Any = None
    normalise: bool = False
    legend_position: str = "top"
    show_delta: Union[bool, Tuple[bool, ...], List[bool]] = False
    markers: List[Marker] = field(default_factory=list)

    kind: ClassVar[str] = "waveform"

    def __post_init__(self) -> None:
        where = f"WaveformPlot {self.name!r}"
        _require_str(self.name, "WaveformPlot.name")
        _require_nonempty(self.channels, f"{where}.channels")
        if not isinstance(self.channels, (list, tuple)):
            raise TypeError(f"{where}.channels must be tuple/list.")
        if self.legend_position not in _VALID_LEGEND_POS:
            raise ValueError(
                f"{where}.legend_position must be one of {sorted(_VALID_LEGEND_POS)}."
            )
        # Per-row length consistency — catches the classic mismatch foot-gun.
        n = len(self.channels)
        for attr in ("axis_limits", "reference_lines", "subplot_heights"):
            value = getattr(self, attr)
            if value is not None and len(value) != n:
                raise ValueError(
                    f"{where}.{attr} has length {len(value)} but channels has {n}."
                )
        if not isinstance(self.x_channel, str) or not self.x_channel.strip():
            raise TypeError(f"{where}.x_channel must be a non-empty string.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        # Normalise show_delta: bool → per-row tuple of bools.
        if isinstance(self.show_delta, bool):
            self.show_delta = tuple(self.show_delta for _ in range(n))
        elif isinstance(self.show_delta, (list, tuple)):
            if len(self.show_delta) != n:
                raise ValueError(
                    f"{where}.show_delta has length {len(self.show_delta)} but channels has {n}."
                )
            self.show_delta = tuple(bool(v) for v in self.show_delta)
        else:
            raise TypeError(
                f"{where}.show_delta must be bool or a tuple/list of bools per row."
            )
        self.normalise = bool(self.normalise)


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------


@dataclass
class ScatterPlot:
    """XY correlation plot with optional segmented fits and outlier handling."""

    name: str
    x_channel: str
    y_channel: str
    axis_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    best_fit: Union[int, List[Tuple[str, Optional[float], Optional[float]]], None] = 0
    gate: Any = None
    show_equations: bool = True
    show_error: bool = True
    # When True, display the gradient delta as an absolute multiplicative
    # factor (e.g. "x 1.10" meaning "multiply this run's slope by 1.10 to get
    # the baseline slope") instead of a signed percentage (+10.0%).
    error_as_factor: bool = False
    color_gate: Any = None
    annotate_fit_at: Any = None
    markers: List[Marker] = field(default_factory=list)
    reference_lines: Optional[List[float]] = None
    # Robust mode (#18): Theil-Sen regression instead of OLS, plus MAD-based
    # outlier rejection on the fit. Outliers are still plotted but as a faint
    # grey 'x' overlay so engineers can see what was excluded.
    robust: bool = False
    # Multiples of MAD used as outlier threshold during robust fitting.
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
                raise TypeError(
                    f"{where}.color_gate must be ('channel', 'op', value, '#hexcolor')."
                )
            _validate_one_gate(self.color_gate[:3], f"{where}.color_gate")
        if isinstance(self.best_fit, (list, tuple)):
            for i, seg in enumerate(self.best_fit):
                if not isinstance(seg, (list, tuple)) or len(seg) != 3 or not isinstance(seg[0], str):
                    raise TypeError(
                        f"{where}.best_fit[{i}] must be (axis, lo, hi); got {seg!r}."
                    )
        elif self.best_fit not in (None, 0, 1, 2):
            raise ValueError(f"{where}.best_fit must be 0/1/2/None or a list of segments.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.robust = bool(self.robust)
        self.robust_threshold = float(self.robust_threshold)
        if self.robust_threshold <= 0:
            raise ValueError(f"{where}.robust_threshold must be > 0.")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------


@dataclass
class PsdPlot:
    name: str
    channel: Union[str, List[str], Tuple[str, ...]]
    axis_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    log_scale: bool = True
    nperseg: Optional[int] = None
    annotate_at: Any = None
    markers: List[Marker] = field(default_factory=list)
    gate: Any = None
    show_envelope: bool = False
    reference_lines: Optional[List[float]] = None

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
            self.nperseg = int(self.nperseg)
            if self.nperseg < 8:
                raise ValueError(f"{where}.nperseg must be >= 8.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.log_scale = bool(self.log_scale)
        _validate_gate(self.gate, f"{where}.gate")
        self.show_envelope = bool(self.show_envelope)
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


@dataclass
class HistogramPlot:
    name: str
    channel: str
    axis_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    log_scale: bool = False
    markers: List[Marker] = field(default_factory=list)
    gate: Any = None
    reference_lines: Optional[List[float]] = None

    kind: ClassVar[str] = "histogram"

    def __post_init__(self) -> None:
        where = f"HistogramPlot {self.name!r}"
        _require_str(self.name, "HistogramPlot.name")
        _require_str(self.channel, f"{where}.channel")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.log_scale = bool(self.log_scale)
        _validate_gate(self.gate, f"{where}.gate")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------


@dataclass
class BarPlot:
    name: str
    metrics: Tuple[Any, ...]
    default_aggregation: str = "last"
    axis_limits: Optional[Tuple[Optional[float], Optional[float]]] = None
    gate: Any = None
    reference_lines: Optional[List[float]] = None

    kind: ClassVar[str] = "bar"

    def __post_init__(self) -> None:
        where = f"BarPlot {self.name!r}"
        _require_str(self.name, "BarPlot.name")
        _require_nonempty(self.metrics, f"{where}.metrics")
        if self.default_aggregation not in _VALID_BAR_AGGS:
            raise ValueError(
                f"{where}.default_aggregation must be one of {sorted(_VALID_BAR_AGGS)}."
            )
        _validate_gate(self.gate, f"{where}.gate")
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


@dataclass
class BoxPlot:
    name: str
    channels: Union[str, List[str], Tuple[str, ...]]
    aggregation_mode: str = "per_run"
    axis_limits: Optional[Tuple[Optional[float], Optional[float]]] = None
    gate: Any = None
    reference_lines: Optional[List[float]] = None
    options: Optional[dict] = None

    kind: ClassVar[str] = "box"

    def __post_init__(self) -> None:
        where = f"BoxPlot {self.name!r}"
        _require_str(self.name, "BoxPlot.name")
        if self.aggregation_mode not in _VALID_BOX_MODES:
            raise ValueError(
                f"{where}.aggregation_mode must be one of {sorted(_VALID_BOX_MODES)}."
            )
        _validate_gate(self.gate, f"{where}.gate")
        # Normalise channels to a list of strings.
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


# ---------------------------------------------------------------------------
# Box Grid (composite — expands into BoxPlot or renders as subplot matrix)
# ---------------------------------------------------------------------------

_VALID_GRID_RENDER_MODES = {"expand", "grid"}


@dataclass
class BoxPlotGrid:
    """A grid of box plots defined by two gate dimensions (rows × cols).

    Each cell in the grid combines the row gate with the column gate (AND-ed).
    The grid can either expand into individual ``BoxPlot`` objects (``render_mode='expand'``)
    or be rendered as a single subplot-matrix figure (``render_mode='grid'``).

    Parameters
    ----------
    name : str
        Base name for the plot (cell names are auto-generated as "{name} - {row_key} {col_key}").
    channels : str or list of str
        Channel(s) to plot in each cell.
    rows : dict
        Ordered mapping of row labels to gate conditions (list of tuples).
        Example: {"LS": [("vCar", '<', 120)], "MS": [("vCar", '>', 120), ("vCar", '<', 200)]}
    cols : dict
        Ordered mapping of column labels to gate conditions (list of tuples).
        Example: {"Entry": [("CosPhi_Calc", 'between', (-0.7, -0.3))], ...}
    aggregation_mode : str
        Same as BoxPlot — "per_run", "aggregated", or "per_run_aggregated".
    axis_limits : tuple, optional
        Y-axis limits applied to all cells.
    options : dict, optional
        Styling overrides passed to each cell.
    render_mode : str
        "expand" — produces one BoxPlot per cell (default).
        "grid" — renders a single figure with a rows×cols subplot matrix.
    """

    name: str
    channels: Union[str, List[str], Tuple[str, ...]]
    rows: dict
    cols: dict
    aggregation_mode: str = "per_run"
    axis_limits: Optional[Tuple[Optional[float], Optional[float]]] = None
    options: Optional[dict] = None
    render_mode: str = "expand"

    kind: ClassVar[str] = "box_grid"

    def __post_init__(self) -> None:
        where = f"BoxPlotGrid {self.name!r}"
        _require_str(self.name, "BoxPlotGrid.name")
        if self.render_mode not in _VALID_GRID_RENDER_MODES:
            raise ValueError(
                f"{where}.render_mode must be one of {sorted(_VALID_GRID_RENDER_MODES)}; "
                f"got {self.render_mode!r}."
            )
        if self.aggregation_mode not in _VALID_BOX_MODES:
            raise ValueError(
                f"{where}.aggregation_mode must be one of {sorted(_VALID_BOX_MODES)}."
            )
        # Validate rows and cols
        if not isinstance(self.rows, dict) or not self.rows:
            raise TypeError(f"{where}.rows must be a non-empty dict.")
        if not isinstance(self.cols, dict) or not self.cols:
            raise TypeError(f"{where}.cols must be a non-empty dict.")
        for label, gate in self.rows.items():
            _validate_gate(gate, f"{where}.rows[{label!r}]")
        for label, gate in self.cols.items():
            _validate_gate(gate, f"{where}.cols[{label!r}]")
        # Normalise channels
        if isinstance(self.channels, str):
            self.channels = [self.channels]
        elif isinstance(self.channels, (list, tuple)):
            for ch in self.channels:
                if not isinstance(ch, str) or not ch.strip():
                    raise TypeError(f"{where}: channel entries must be non-empty strings.")
            self.channels = list(self.channels)
        else:
            raise TypeError(f"{where}.channels must be a string or list of strings.")

    def expand(self) -> List["BoxPlot"]:
        """Expand into individual BoxPlot instances (one per grid cell)."""
        plots = []
        for row_label, row_gate in self.rows.items():
            for col_label, col_gate in self.cols.items():
                # Combine row + col gates (normalise single-condition tuples to lists)
                combined_gate = _normalise_gate_list(row_gate) + _normalise_gate_list(col_gate)
                cell_name = f"{self.name} - {row_label} {col_label}"
                plots.append(BoxPlot(
                    name=cell_name,
                    channels=list(self.channels),
                    aggregation_mode=self.aggregation_mode,
                    axis_limits=self.axis_limits,
                    gate=combined_gate,
                    options=self.options,
                ))
        return plots


def _normalise_gate_list(gate) -> list:
    """Normalise a gate spec to a list of 3-tuple conditions."""
    if gate is None:
        return []
    if isinstance(gate, (list, tuple)) and len(gate) == 3 and isinstance(gate[0], str):
        return [tuple(gate)]
    return [tuple(c) for c in gate]


# ---------------------------------------------------------------------------
# Heatmap (NEW)
# ---------------------------------------------------------------------------


@dataclass
class HeatmapPlot:
    """2-D binned aggregate plot (one panel per run).

    For each (x_bin, y_bin) cell, computes the aggregate of z_channel
    (or counts when z_channel is None).
    """

    name: str
    x_channel: str
    y_channel: str
    z_channel: Optional[str] = None
    aggregation: str = "mean"
    bins: Union[int, Tuple[int, int]] = 40
    axis_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    cmap: str = "viridis"
    z_limits: Optional[Tuple[Optional[float], Optional[float]]] = None
    gate: Any = None
    markers: List[Marker] = field(default_factory=list)
    min_count: int = 3  # cells with fewer points are masked

    kind: ClassVar[str] = "heatmap"

    def __post_init__(self) -> None:
        where = f"HeatmapPlot {self.name!r}"
        _require_str(self.name, "HeatmapPlot.name")
        _require_str(self.x_channel, f"{where}.x_channel")
        _require_str(self.y_channel, f"{where}.y_channel")
        if self.z_channel is not None and (not isinstance(self.z_channel, str) or not self.z_channel.strip()):
            raise TypeError(f"{where}.z_channel must be a non-empty string or None.")
        if self.aggregation not in _VALID_HEATMAP_AGGS:
            raise ValueError(
                f"{where}.aggregation must be one of {sorted(_VALID_HEATMAP_AGGS)}."
            )
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


# ---------------------------------------------------------------------------
# Helpers used across the codebase
# ---------------------------------------------------------------------------


PLOT_TYPE_ORDER: Tuple[str, ...] = (
    "waveform", "scatter", "psd", "histogram", "bar", "box", "heatmap",
)
"""Canonical group order — used to index ``DataPlotter.PLOT_DEFINITIONS``."""


_PLOT_KIND_TO_INDEX = {kind: i for i, kind in enumerate(PLOT_TYPE_ORDER)}


def plot_group_index(kind: str) -> int:
    """Group-tuple index for a plot kind string."""
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
