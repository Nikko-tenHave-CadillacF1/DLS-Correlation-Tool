"""DLS Correlation Tool — plotting engine package.

This package contains the data-loading, processing, and plot-generation
backend. User-facing Run_*.py scripts import from here via:

    from engine import run_workflow, WaveformPlot, ScatterPlot, ...
"""

from engine.plot_runtime import (  # noqa: F401
    run_workflow,
    run_from_config,
    parse_plot_cli,
    build_plot_groups,
    workflow_config,
    PlotJobConfig,
    Slide,
    Marker,
    WaveformPlot,
    ScatterPlot,
    PsdPlot,
    HistogramPlot,
    BarPlot,
    BoxPlot,
    BoxPlotGrid,
    HeatmapPlot,
)
from engine.datafunctions import calc_channel  # noqa: F401
