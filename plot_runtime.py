"""Shared helpers for plot-entry scripts."""

import os
import traceback
import json

from dataplotter import DataPlotter
from powerpointexporter import export_report_to_powerpoint, get_template_plot_aspect_ratios


DEFAULT_FIG_SIZE = [(15.5, 6.4), (10, 8), (10, 8), (10, 8), (10, 6)]
_ASPECT_RATIO_CACHE = {}


def build_plot_groups(*groups):
    """Normalize plot-definition groups into the tuple expected by DataPlotter."""
    return tuple(group or [] for group in groups)


def build_plotter(
    *,
    root_folder,
    runs,
    plot_definitions,
    channel_mappings=None,
    channel_transforms=None,
    calculated_channels=None,
    low_pass_filters=None,
    units_map=None,
    template_path=None,
    export_map=None,
    fig_size=None,
    scatter_render_mode="auto",
    scatter_density_threshold=25000,
    scatter_max_points=45000,
    scatter_hexbin_gridsize=70,
    bar_secondary_axis_ratio=20.0,
    box_plot_settings=None,
    output_dir=None,
):
    """Build a DataPlotter with optional PowerPoint aspect-ratio hints."""
    plot_aspect_ratios = {}
    if template_path and export_map:
        cache_key = (
            str(template_path),
            json.dumps(export_map, sort_keys=True),
        )
        plot_aspect_ratios = _ASPECT_RATIO_CACHE.get(cache_key)
        if plot_aspect_ratios is None:
            plot_aspect_ratios = get_template_plot_aspect_ratios(template_path, export_map)
            _ASPECT_RATIO_CACHE[cache_key] = plot_aspect_ratios

    return DataPlotter(
        root_folder=root_folder,
        runs=runs,
        plot_definitions=plot_definitions,
        channel_mappings=channel_mappings,
        channel_transforms=channel_transforms,
        calculated_channels=calculated_channels,
        low_pass_filters=low_pass_filters,
        fig_size=fig_size or DEFAULT_FIG_SIZE,
        units_map=units_map,
        plot_aspect_ratios=plot_aspect_ratios,
        scatter_render_mode=scatter_render_mode,
        scatter_density_threshold=scatter_density_threshold,
        scatter_max_points=scatter_max_points,
        scatter_hexbin_gridsize=scatter_hexbin_gridsize,
        bar_secondary_axis_ratio=bar_secondary_axis_ratio,
        box_plot_settings=box_plot_settings,
        output_dir=output_dir,
    )


def run_plot_job(
    *,
    title,
    plotter,
    plot_method="plot_all",
    generate_message="Generating plots...",
    powerpoint_template=None,
    powerpoint_output=None,
    export_map=None,
):
    """Run a plotting job with consistent console output and optional PPT export."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")

    print(f"\n{generate_message}")
    getattr(plotter, plot_method)()

    if powerpoint_template and powerpoint_output and export_map:
        print("\nExporting to PowerPoint...")
        try:
            powerpoint_output.parent.mkdir(parents=True, exist_ok=True)
            export_report_to_powerpoint(
                template_path=powerpoint_template,
                output_path=powerpoint_output,
                plots_dir=plotter.plots_dir,
                export_map=export_map,
                visible=False,
            )
            try:
                os.startfile(powerpoint_output)
            except Exception as open_err:
                print(f"[WARNING] Could not auto-open PowerPoint file: {open_err}")
                print(f"File saved to: {powerpoint_output}")
        except Exception as export_err:
            print(f"[ERROR] PowerPoint export failed: {export_err}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"{'PROCESSING COMPLETE':^80}")
    print("=" * 80 + "\n")
