from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PlotContext:
    """Runtime state shared between :class:`DataPlotter` and its plot-generator
    mixins.

    Introduced as a first step toward disentangling DataPlotter state from
    DataPlotter behaviour. The plotter still exposes each field via a
    ``self.<field>`` shim property so existing ``self.run_data`` /
    ``self._psd_cache`` access patterns keep working unchanged during the
    incremental refactor.
    """

    run_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    run_units: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_filepaths: dict[str, Path] = field(default_factory=dict)
    run_required_cols: dict[str, Any] = field(default_factory=dict)
    run_sample_rates: dict[str, tuple[float, str]] = field(default_factory=dict)
    psd_cache: dict[tuple, Any] = field(default_factory=dict)
    gated_data_cache: dict[tuple, Any] = field(default_factory=dict)
    outlier_log: list[Any] = field(default_factory=list)
    # Lazy per-plot-invocation accumulator used by generate_psd_plots to
    # collect Lorentz-fit records that _log_lorentz_fit_summary later
    # reports. Previously a lazy-initialised ``self._lorentz_fit_records``
    # attribute on DataPlotter (Prompt 10 phase 1 promoted it here so the
    # helper extraction can be state-free).
    lorentz_fit_records: list[dict] = field(default_factory=list)
