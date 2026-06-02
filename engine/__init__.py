"""DLS Correlation Tool — plotting engine package.

This package contains the data-loading, processing, and plot-generation
backend. User-facing Run_*.py scripts import from here via:

    from engine import (
        run_workflow, Slide, Marker, calc_channel,
        WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot,
        BarPlot, BoxPlot, BoxPlotGrid, HeatmapPlot,
    )

Exports
-------
run_workflow      — One-call entry point for Run_*.py scripts.
run_from_config   — Lower-level: run from a pre-built PlotJobConfig.
parse_plot_cli    — Parse CLI args (--only, --types, --dry-run, etc.).
build_plot_groups — Assemble plot definition lists into the tuple format.
workflow_config   — Build a PlotJobConfig from a named workflow.
PlotJobConfig     — Dataclass bundling all job parameters.
Slide             — Declarative PowerPoint slide definition helper.
Marker            — Vertical reference line (static or condition-triggered).
calc_channel      — Decorator for declaring calculated-channel dependencies.
WaveformPlot, ScatterPlot, PsdPlot, HistogramPlot, BarPlot, BoxPlot,
BoxPlotGrid, HeatmapPlot — Plot definition dataclasses.
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
