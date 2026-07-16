"""Tests for pure helpers in `engine.datafunctions`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.datafunctions import (
    apply_gate_to_dataframe,
    collect_gate_channels,
    compute_gate_mask,
    convert_yes_no_to_binary,
    normalize_bar_metric_specs,
    sanitize_numeric_series,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "SM": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "pBrakeF": [0.0, 0.0, 20.0, 60.0, 80.0, 100.0],
            "vCar": [10.0, 40.0, 90.0, 150.0, 220.0, 280.0],
        }
    )


class TestComputeGateMask:
    def test_none_returns_all_true(self, df):
        mask = compute_gate_mask(df, None)
        assert mask.all()

    @pytest.mark.parametrize(
        "op,val,expected",
        [
            (">", 0.5, [False, False, False, True, True, True]),
            (">=", 0.4, [False, False, True, True, True, True]),
            ("<", 0.4, [True, True, False, False, False, False]),
            ("<=", 0.4, [True, True, True, False, False, False]),
            ("==", 0.4, [False, False, True, False, False, False]),
            ("!=", 0.4, [True, True, False, True, True, True]),
        ],
    )
    def test_simple_comparison_operators(self, df, op, val, expected):
        mask = compute_gate_mask(df, ("SM", op, val))
        assert list(mask) == expected

    def test_between_inclusive(self, df):
        mask = compute_gate_mask(df, ("SM", "between", [0.2, 0.6]))
        assert list(mask) == [False, True, True, True, False, False]

    def test_outside_two_sided(self, df):
        mask = compute_gate_mask(df, ("SM", "outside", [0.2, 0.6]))
        # (col < 0.2) OR (col > 0.6)
        assert list(mask) == [True, False, False, False, True, True]

    def test_multiple_conditions_are_anded(self, df):
        mask = compute_gate_mask(df, [("SM", ">=", 0.4), ("pBrakeF", "<", 80)])
        assert list(mask) == [False, False, True, True, False, False]

    def test_missing_channel_returns_all_false(self, df):
        mask = compute_gate_mask(df, ("doesNotExist", ">", 0))
        assert not mask.any()

    def test_invalid_operator_returns_all_false(self, df):
        # `compute_gate_mask` uses the same operator whitelist as
        # `_validate_gate` but is more forgiving at runtime.
        mask = compute_gate_mask(df, ("SM", "??", 0.5))
        assert not mask.any()

    def test_robust_operator(self, df):
        mask = compute_gate_mask(df, ("SM", "robust", 100))  # k huge -> keep all
        assert mask.all()


class TestApplyGateToDataframe:
    def test_none_returns_original(self, df):
        out = apply_gate_to_dataframe(df, None)
        assert out is df

    def test_filters_rows(self, df):
        out = apply_gate_to_dataframe(df, ("SM", ">", 0.5))
        assert len(out) == 3
        assert out["SM"].min() > 0.5

    def test_empty_result_returns_empty_dataframe(self, df):
        out = apply_gate_to_dataframe(df, ("SM", ">", 999))
        assert isinstance(out, pd.DataFrame)
        assert out.empty
        # Columns preserved even on empty result.
        assert list(out.columns) == list(df.columns)


class TestCollectGateChannels:
    def test_single_condition(self):
        assert collect_gate_channels(("SM", "<", 0.5)) == {"SM"}

    def test_multiple_conditions(self):
        channels = collect_gate_channels([("SM", "<", 0.5), ("pBrakeF", ">", 1)])
        assert channels == {"SM", "pBrakeF"}

    def test_none_returns_empty(self):
        assert collect_gate_channels(None) == set()


class TestNormalizeBarMetricSpecs:
    def test_string_becomes_default_agg(self):
        assert normalize_bar_metric_specs("PMGUK") == [("PMGUK", "last")]

    def test_tuple_pair_kept(self):
        assert normalize_bar_metric_specs([("PMGUK", "integral")]) == [
            ("PMGUK", "integral"),
        ]

    def test_invalid_agg_falls_back_to_default(self):
        assert normalize_bar_metric_specs([("PMGUK", "spline")], default_aggregation="max") == [("PMGUK", "max")]

    def test_non_string_channel_dropped(self):
        assert normalize_bar_metric_specs([(123, "sum")]) == []

    def test_scalar_non_sequence_returns_empty(self):
        assert normalize_bar_metric_specs(42) == []


class TestConvertYesNoToBinary:
    def test_converts_yes_no_columns(self):
        df = pd.DataFrame({"flag": ["Yes", "No", "yes", "no"], "x": [1, 2, 3, 4]})
        out = convert_yes_no_to_binary(df.copy())
        assert list(out["flag"]) == [1, 0, 1, 0]
        assert list(out["x"]) == [1, 2, 3, 4]

    def test_leaves_pure_numeric_alone(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = convert_yes_no_to_binary(df.copy())
        pd.testing.assert_frame_equal(out, df)


class TestSanitizeNumericSeries:
    def test_strips_inf_and_int64_sentinels(self):
        s = pd.Series([1.0, np.inf, -np.inf, float(np.iinfo(np.int64).min), 2.5])
        out = sanitize_numeric_series(s)
        assert out.iloc[0] == 1.0
        assert np.isnan(out.iloc[1])
        assert np.isnan(out.iloc[2])
        assert np.isnan(out.iloc[3])
        assert out.iloc[4] == 2.5

    def test_coerces_strings(self):
        s = pd.Series(["1.5", "not a number", "3"])
        out = sanitize_numeric_series(s)
        assert out.iloc[0] == 1.5
        assert np.isnan(out.iloc[1])
        assert out.iloc[2] == 3.0
