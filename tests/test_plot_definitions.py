"""Tests for `engine.plot_definitions` dataclass validation."""
from __future__ import annotations

import pytest

from engine.plot_definitions import (
    _VALID_BAR_AGGS,
    _VALID_BOX_MODES,
    _VALID_GATE_OPS,
    BarPlot,
    BoxPlot,
    BoxPlotGrid,
    HeatmapPlot,
    HistogramPlot,
    Marker,
    PsdPlot,
    Scatter3DPlot,
    ScatterPlot,
    WaveformPlot,
)


class TestMarker:
    def test_requires_x_or_condition(self):
        with pytest.raises(ValueError, match="exactly one of 'x' or 'condition'"):
            Marker()
        with pytest.raises(ValueError, match="exactly one of 'x' or 'condition'"):
            Marker(x=1.0, condition=("SM", ">", 0.5))

    def test_x_coerced_to_float(self):
        m = Marker(x=7)
        assert isinstance(m.x, float)
        assert m.x == 7.0

    def test_condition_edge_validation(self):
        Marker(condition=("SM", ">", 0.5), edge="rising")  # ok
        with pytest.raises(ValueError, match="edge"):
            Marker(condition=("SM", ">", 0.5), edge="sideways")

    def test_condition_max_count_positive(self):
        Marker(condition=("SM", ">", 0.5), max_count=3)  # ok
        with pytest.raises(ValueError, match="max_count must be positive"):
            Marker(condition=("SM", ">", 0.5), max_count=0)


class TestScatterPlot:
    def test_minimal_valid(self):
        s = ScatterPlot("t", "x", "y")
        assert s.kind == "scatter"
        assert s.best_fit == 0
        assert s.gate is None

    def test_empty_channel_rejected(self):
        with pytest.raises(TypeError):
            ScatterPlot("t", "", "y")

    def test_gate_shape(self):
        ScatterPlot("t", "x", "y", gate=("SM", "<", 0.5))
        ScatterPlot("t", "x", "y", gate=[("SM", "<", 0.5), ("pBrakeF", "<", 1)])
        with pytest.raises(ValueError, match="unknown gate operator"):
            ScatterPlot("t", "x", "y", gate=("SM", "??", 0.5))

    def test_best_fit_shapes(self):
        ScatterPlot("t", "x", "y", best_fit=1)
        ScatterPlot("t", "x", "y", best_fit=[("x", 0.0, 1.0)])
        with pytest.raises(ValueError):
            ScatterPlot("t", "x", "y", best_fit=99)
        with pytest.raises(TypeError):
            ScatterPlot("t", "x", "y", best_fit=[("bad",)])

    def test_color_gate_shape(self):
        ScatterPlot("t", "x", "y", color_gate=("SM", ">", 0.5, "#ff0000"))
        with pytest.raises(TypeError):
            ScatterPlot("t", "x", "y", color_gate=("SM", ">", 0.5))

    def test_robust_threshold_positive(self):
        with pytest.raises(ValueError, match="robust_threshold"):
            ScatterPlot("t", "x", "y", robust_threshold=0)


class TestScatter3DPlot:
    def test_minimal_valid(self):
        s = Scatter3DPlot("t", "x", "y", "z")
        assert s.kind == "scatter3d"
        assert s.gate is None
        assert s.axis_limits is None

    def test_missing_z_channel_rejected(self):
        with pytest.raises(TypeError):
            Scatter3DPlot("t", "x", "y", "")

    def test_axis_limits_shape(self):
        Scatter3DPlot("t", "x", "y", "z", axis_limits=[(0, 1), (0, 1), (None, None)])
        with pytest.raises(ValueError, match="axis_limits must be a length-3"):
            Scatter3DPlot("t", "x", "y", "z", axis_limits=[(0, 1), (0, 1)])

    def test_gate_validated(self):
        with pytest.raises(ValueError, match="unknown gate operator"):
            Scatter3DPlot("t", "x", "y", "z", gate=("SM", "??", 0.5))


class TestPsdPlot:
    def test_single_or_multi_channel(self):
        assert PsdPlot("t", "gVertF").kind == "psd"
        assert PsdPlot("t", ["gVertF", "gVertR"]).channel == ["gVertF", "gVertR"]

    def test_empty_multi_channel_rejected(self):
        with pytest.raises(ValueError, match="channel list"):
            PsdPlot("t", [])

    def test_nperseg_coerced(self):
        p = PsdPlot("t", "gVertF", nperseg="512")
        assert p.nperseg == 512
        p = PsdPlot("t", "gVertF", nperseg="auto")
        assert p.nperseg is None
        with pytest.raises(ValueError, match="nperseg must be >= 8"):
            PsdPlot("t", "gVertF", nperseg=4)

    def test_lorentz_fit_shape(self):
        p = PsdPlot("t", "gVertF", lorentz_fit=(6, 12))
        assert p.lorentz_fit == [(6.0, 12.0)]
        p = PsdPlot("t", "gVertF", lorentz_fit=[(3, 7), (5, 10)])
        assert len(p.lorentz_fit) == 2
        with pytest.raises(ValueError):
            PsdPlot("t", "gVertF", lorentz_fit=(0, 1))  # f_lo must be > 0


class TestBarPlot:
    def test_default_aggregation_in_valid_set(self):
        BarPlot("t", (("PMGUK", "integral"),), default_aggregation="last")
        with pytest.raises(ValueError, match="default_aggregation"):
            BarPlot("t", (("PMGUK", "integral"),), default_aggregation="totally_made_up")

    def test_error_metrics_length_matches(self):
        BarPlot("t", (("a", "sum"), ("b", "sum")), error_metrics=("a_sig", None))
        with pytest.raises(ValueError, match="error_metrics has length"):
            BarPlot("t", (("a", "sum"), ("b", "sum")), error_metrics=("a_sig",))

    def test_all_valid_aggs_accepted_via_normalise(self):
        # Sanity: every entry in _VALID_BAR_AGGS builds a bar with that agg.
        for agg in _VALID_BAR_AGGS:
            BarPlot("t", (("PMGUK", agg),))


class TestBoxPlotGrid:
    def test_expand_produces_row_x_col_boxplots(self):
        grid = BoxPlotGrid(
            name="G",
            channels="vCar",
            rows={"lo": ("SM", "<", 0.5), "hi": ("SM", ">=", 0.5)},
            cols={"str": ("pBrakeF", "<", 1), "brk": ("pBrakeF", ">=", 1)},
        )
        expanded = grid.expand()
        assert len(expanded) == 4
        assert all(isinstance(p, BoxPlot) for p in expanded)
        names = sorted(p.name for p in expanded)
        assert names == [
            "G - hi brk",
            "G - hi str",
            "G - lo brk",
            "G - lo str",
        ]

    def test_invalid_render_mode(self):
        with pytest.raises(ValueError, match="render_mode"):
            BoxPlotGrid(
                name="G", channels="vCar",
                rows={"a": None}, cols={"b": None},
                render_mode="waterfall",
            )


class TestHeatmapPlot:
    def test_bins_int_or_pair(self):
        HeatmapPlot("t", "x", "y", bins=40)
        HeatmapPlot("t", "x", "y", bins=(20, 30))
        with pytest.raises(TypeError):
            HeatmapPlot("t", "x", "y", bins="lots")
        with pytest.raises(ValueError, match=r"bins must be >= 4"):
            HeatmapPlot("t", "x", "y", bins=2)

    def test_aggregation_validated(self):
        with pytest.raises(ValueError, match="aggregation"):
            HeatmapPlot("t", "x", "y", aggregation="wat")


class TestWaveformPlot:
    def test_axis_limits_length_must_match_channels(self):
        with pytest.raises(ValueError, match=r"axis_limits.*length"):
            WaveformPlot(
                name="w",
                channels=("a", "b", "c"),
                axis_limits=(None, None),  # wrong length
            )

    def test_legend_position_validated(self):
        with pytest.raises(ValueError, match="legend_position"):
            WaveformPlot(name="w", channels=("a",), legend_position="bottom")


def test_all_gate_ops_accepted_in_scatter():
    for op in _VALID_GATE_OPS:
        # Non-numeric operands are fine — validator only checks the op string.
        ScatterPlot("t", "x", "y", gate=("SM", op, 0.5))


def test_all_box_modes_accepted():
    for mode in _VALID_BOX_MODES:
        BoxPlot(name="b", channels="vCar", aggregation_mode=mode)


def test_histogram_minimal():
    h = HistogramPlot("t", "x")
    assert h.kind == "histogram"
    assert h.log_scale is False
