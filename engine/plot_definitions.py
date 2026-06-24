
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union

_VALID_BAR_AGGS = {
    "integral", "abs_integral", "sum", "abs_sum",
    "mean", "median", "max", "min", "first", "last",
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

@dataclass
class Marker:

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

def _coerce_lorentz_fit(value: Any, where: str) -> Optional[List[Tuple[float, float]]]:
    def _coerce_window(item: Any, idx_label: str) -> Tuple[float, float]:
        if isinstance(item, (list, tuple)) and len(item) == 2 \
                and all(isinstance(v, (int, float)) for v in item):
            lo = float(item[0])
            hi = float(item[1])
            if not (0 < lo < hi):
                raise ValueError(
                    f"{where}: {idx_label} must be (f_lo, f_hi) with 0 < f_lo < f_hi."
                )
            return (lo, hi)
        raise TypeError(
            f"{where}: {idx_label} must be an (f_lo, f_hi) tuple; got {item!r}."
        )
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2 \
            and all(isinstance(v, (int, float)) for v in value):
        return [_coerce_window(value, "value")]
    if isinstance(value, list):
        return [_coerce_window(item, f"entry #{i}") for i, item in enumerate(value)]
    raise TypeError(
        f"{where} must be an (f_lo, f_hi) tuple or list of such tuples. Got {value!r}."
    )

@dataclass
class WaveformPlot:

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
    annotate_at: Optional[Union[Tuple[float, ...], List[float], float]] = None
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
        if self.annotate_at is not None:
            if isinstance(self.annotate_at, (int, float)):
                self.annotate_at = (float(self.annotate_at),)
            elif isinstance(self.annotate_at, (list, tuple)):
                self.annotate_at = tuple(float(v) for v in self.annotate_at)
            else:
                raise TypeError(
                    f"{where}.annotate_at must be a number or tuple/list of numbers."
                )

@dataclass
class ScatterPlot:

    name: str
    x_channel: str
    y_channel: str
    axis_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    best_fit: Union[int, List[Tuple[str, Optional[float], Optional[float]]], None] = 0
    gate: Any = None
    show_equations: bool = True
    show_error: bool = True
    error_as_factor: bool = False
    color_gate: Any = None
    annotate_fit_at: Any = None
    markers: List[Marker] = field(default_factory=list)
    reference_lines: Optional[List[float]] = None
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
            self.nperseg = int(self.nperseg)
            if self.nperseg < 8:
                raise ValueError(f"{where}.nperseg must be >= 8.")
        self.markers = _coerce_markers(self.markers, f"{where}.markers")
        self.log_scale = bool(self.log_scale)
        _validate_gate(self.gate, f"{where}.gate")
        self.show_envelope = bool(self.show_envelope)
        self.reference_lines = _coerce_flat_reference_lines(self.reference_lines, f"{where}.reference_lines")
        self.lorentz_fit = _coerce_lorentz_fit(self.lorentz_fit, f"{where}.lorentz_fit")

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

@dataclass
class BarPlot:
    name: str
    metrics: Tuple[Any, ...]
    default_aggregation: str = "last"
    axis_limits: Optional[Tuple[Optional[float], Optional[float]]] = None
    gate: Any = None
    reference_lines: Optional[List[float]] = None
    error_metrics: Optional[Tuple[Any, ...]] = None
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
        if self.error_metrics is not None:
            if not isinstance(self.error_metrics, (list, tuple)):
                raise TypeError(
                    f"{where}.error_metrics must be a tuple/list (one entry per metric) "
                    "or None."
                )
            if len(self.error_metrics) != len(self.metrics):
                raise ValueError(
                    f"{where}.error_metrics has length {len(self.error_metrics)} "
                    f"but metrics has {len(self.metrics)}."
                )
            for i, em in enumerate(self.error_metrics):
                if em is not None and (not isinstance(em, str) or not em.strip()):
                    raise TypeError(
                        f"{where}.error_metrics[{i}] must be a channel name string or None; "
                        f"got {em!r}."
                    )

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
    def expand(self) -> List["BoxPlot"]:
        plots = []
        for row_label, row_gate in self.rows.items():
            for col_label, col_gate in self.cols.items():
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
    if gate is None:
        return []
    if isinstance(gate, (list, tuple)) and len(gate) == 3 and isinstance(gate[0], str):
        return [tuple(gate)]
    return [tuple(c) for c in gate]

@dataclass
class HeatmapPlot:

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

PLOT_TYPE_ORDER: Tuple[str, ...] = (
    "waveform", "scatter", "psd", "histogram", "bar", "box", "heatmap",
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
