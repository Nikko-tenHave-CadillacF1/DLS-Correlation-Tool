"""Smoke tests for `engine.modal_plots.plot_modal_evolution`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from engine.modal_plots import plot_modal_evolution

MODES = ("Heave", "Pitch", "Roll", "Warp")


def _result(shift: float, with_amps: bool = True) -> dict:
    fn = np.array([5.5, 9.1, 5.3, 10.3]) + shift
    res = {
        "mode_labels": list(MODES),
        "fn": fn,
        "zeta": np.array([0.09, 0.08, 0.11, 0.11]),
        "sigma_fn": np.full(4, 0.05),
        "sigma_zeta": np.full(4, np.nan),
        "fn_ci": (fn - 0.1, fn + 0.2),
    }
    if with_amps:
        res |= {"amp_front": np.full(4, 2e5), "amp_rear": np.full(4, 3e5), "sigma_amp_front": np.full(4, 1e4)}
    return res


@pytest.fixture
def plotter(tmp_path):
    runs, results = [], {}
    for i, (group, session) in enumerate([("RED", "P1"), ("RED", "Q"), ("BLUE", "P1"), ("BLUE", "Q")]):
        name = f"26R07BCN_{group}_{session}"
        runs.append({"name": name, "group": group})
        results[name.lower()] = _result(0.1 * i, with_amps=group == "RED")
    return SimpleNamespace(runs=runs, modal_results=results, plots_dir=tmp_path, output_dpi=50)


def test_compare_by_session_writes_all_figures(plotter):
    saved = plot_modal_evolution(plotter, modes=("Heave",), group_by="group", compare_by="session", name_suffix="heave")
    names = sorted(p.name for p in saved)
    assert names == sorted(
        f"modal_evolution_{kind}_{stem}.png" for kind in ("line", "bar") for stem in ("f0_heave", "zeta_heave", "amp_heave")
    )
    assert all(p.exists() for p in saved)


def test_flat_layout_and_empty_amplitudes_skipped(plotter):
    for res in plotter.modal_results.values():
        res.pop("amp_front", None)
        res.pop("amp_rear", None)
    saved = plot_modal_evolution(plotter, bars=False)
    assert sorted(p.name for p in saved) == ["modal_evolution_line_f0.png", "modal_evolution_line_zeta.png"]


def test_unknown_compare_by_rejected(plotter):
    with pytest.raises(ValueError):
        plot_modal_evolution(plotter, compare_by="driver")
