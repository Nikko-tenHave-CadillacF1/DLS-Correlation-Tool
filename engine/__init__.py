
from engine.datafunctions import calc_channel  # noqa: F401
from engine.modal_plots import plot_modal_evolution  # noqa: F401
from engine.plot_runtime import (  # noqa: F401
    BarPlot,
    BoxPlot,
    BoxPlotGrid,
    HeatmapPlot,
    HistogramPlot,
    Marker,
    PlotJobConfig,
    PsdPlot,
    ScatterPlot,
    Slide,
    WaveformPlot,
    build_plot_groups,
    parse_plot_cli,
    run_from_config,
    run_workflow,
    workflow_config,
)

__all__ = [
    # Workflow entrypoints
    "run_workflow", "run_from_config", "parse_plot_cli",
    "build_plot_groups", "workflow_config", "PlotJobConfig",
    # Plot dataclasses (re-exported from plot_definitions via plot_runtime)
    "Slide", "Marker",
    "WaveformPlot", "ScatterPlot", "PsdPlot", "HistogramPlot",
    "BarPlot", "BoxPlot", "BoxPlotGrid", "HeatmapPlot",
    # Helpers
    "calc_channel",
    "plot_modal_evolution",
]

# NOTE: `engine.vibrations_io` is NOT re-exported here because it transitively
# imports `channel_config`, which imports `engine.datafunctions` — eager
# re-export creates a circular import while `engine/__init__.py` itself is
# still mid-load. Import it explicitly when needed:
#     from engine.vibrations_io import run_fit, plot_comparison, expand_runs
# The lazy import inside `DataPlotter._run_modal_fits` is unaffected because
# it executes at runtime, after `channel_config` has fully loaded.
#
# The two pure-math fit modules (`engine.vibrations_lorentz` and
# `engine.vibrations_body4dof`) are standalone and have no such dependency,
# but are also not re-exported here — import them directly if you want to
# call the fit kernels without the I/O / plotting glue.
