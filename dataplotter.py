import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import font_manager
from pathlib import Path
import datafunctions
from collections import Counter

def make_unique(names):
    """Make column names unique by appending suffixes to duplicates"""
    counts = Counter(names)
    unique_names = []
    seen = {}
    for name in names:
        if counts[name] > 1:
            if name not in seen:
                seen[name] = 1
                unique_names.append(name)
            else:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
        else:
            unique_names.append(name)
    return unique_names

class DataPlotter:
    """Main class for loading, processing, and plotting multi-run data"""


    def __init__(self, root_folder, dls_run, track_run, plot_definitions=None, channel_mappings={'dls': None,'track': None}, channel_transforms={'dls': None,'track': None}, calculated_channels={'dls': None,'track': None}, low_pass_filters=None, fig_size=[(15.5, 6.4), (10, 8), (10, 8)], units_map=None, plot_aspect_ratios=None, sample_rate=100, scatter_dot_size=5, scatter_transparency=0.8):

        self.dls_run = dls_run 
        self.track_run = track_run 
        self.dls_label = dls_run.get('display_name', dls_run['name'])
        self.track_label = track_run.get('display_name', 'CAR' if track_run['name'].lower() == 'track' else track_run['name'])
        self._configure_plot_style()

        # Load DLS data
        dls_file_path = Path(root_folder) / dls_run['file']
        self.dls_data, header_dls, self.units_dls = self._load_run_data(dls_file_path, use_python_engine=False)

        # Load Track data
        track_file_path = Path(root_folder) / track_run['file']
        self.track_data, header_track, self.units_track = self._load_run_data(track_file_path, use_python_engine=True)

        # Combine units, preferring track units when different from DLS
        self.units = {}
        for col in set(header_dls + header_track):
            if col in self.units_track and self.units_track[col] != self.units_dls.get(col, ''):
                self.units[col] = self.units_track[col]
            elif col in self.units_dls:
                self.units[col] = self.units_dls[col]
            elif col in self.units_track:
                self.units[col] = self.units_track[col]

        self.PLOT_DEFINITIONS = plot_definitions
        self.CHANNEL_MAPPINGS = channel_mappings
        self.CHANNEL_TRANSFORMS = channel_transforms
        self.CALCULATED_CHANNELS = calculated_channels
        self.units_map = units_map or self.units

        self.FILTER_SAMPLE_RATE = sample_rate  # Hz - sampling rate of your data
        self.LOW_PASS_FILTERS = low_pass_filters 
        
        self.SCATTER_DOT_SIZE = scatter_dot_size
        self.SCATTER_TRANSPARENCY = scatter_transparency
        self.waveform_figsize = fig_size[0]  # Size for waveform plots
        self.scatter_FIGSIZE = fig_size[1]  # Size for individual scatter plots
        self.psd_FIGSIZE = fig_size[2]  # Size for PSD plots
        self.plot_aspect_ratios = plot_aspect_ratios or {}

        # Create plots directory
        self.plots_dir = Path(root_folder) / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Clean data: remove columns with strings, interpolate NaNs, and apply transformations
        self.apply_channel_mappings()
        self.apply_transformations()
        self.clean_data()
        self.apply_calculated_channels()
        self.apply_lowpass_filters()

    def _configure_plot_style(self):
        """Apply a consistent font and baseline styling to all plots."""
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        preferred_font = 'Montserrat' if 'Montserrat' in available_fonts else 'DejaVu Sans'
        plt.rcParams.update({
            'font.family': preferred_font,
            'font.sans-serif': ['Montserrat', 'DejaVu Sans', 'Arial', 'sans-serif'],
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.titleweight': 'bold',
        })

    def _load_run_data(self, file_path, use_python_engine=False):
        """Load either legacy text exports or parquet files."""
        try:
            if file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
                df.columns = make_unique([str(col) for col in df.columns])
                header = list(df.columns)
                units = {col: '' for col in header}
                return df, header, units

            with open(file_path, 'r') as f:
                lines = f.readlines()
            header = make_unique(lines[1].strip().split(','))
            units_row = lines[2].strip().split(',')
            read_csv_kwargs = {
                'sep': r',',
                'skiprows': 3,
                'header': None,
                'names': header,
                'on_bad_lines': 'skip',
            }
            if use_python_engine:
                read_csv_kwargs['engine'] = 'python'
            else:
                read_csv_kwargs['low_memory'] = False

            df = pd.read_csv(file_path, **read_csv_kwargs)
            units = dict(zip(header, units_row))
            return df, header, units
        except Exception as e:
            print(f"Error loading data file {file_path}: {e}")
            raise
    
    def clean_data(self):
        """Remove columns containing strings and interpolate NaN values"""
        # Convert YES/NO to 1/0 before other cleaning
        self.dls_data = datafunctions.convert_yes_no_to_binary(self.dls_data)
        self.track_data = datafunctions.convert_yes_no_to_binary(self.track_data)

        # Clean track data
        for col in list(self.track_data.columns):
            print(f"Column {col} dtype: {self.track_data[col].dtype}")
            if self.track_data[col].dtype == 'object' or self.track_data[col].dtype.name in ['string', 'str']:
                non_nan = self.track_data[col].dropna()
                if any(isinstance(x, str) for x in non_nan):
                    self.track_data.drop(col, axis=1, inplace=True)
                    print(f"Dropped column {col} from track data due to strings")
                else:
                    self.track_data[col] = datafunctions.sanitize_numeric_series(self.track_data[col])
                    self.track_data[col] = self.track_data[col].interpolate(method='linear')
            else:
                self.track_data[col] = datafunctions.sanitize_numeric_series(self.track_data[col])
                self.track_data[col] = self.track_data[col].interpolate(method='linear')

        # Clean DLS data
        for col in list(self.dls_data.columns):
            if self.dls_data[col].dtype == 'object' or self.dls_data[col].dtype.name in ['string', 'str']:
                non_nan = self.dls_data[col].dropna()
                if any(isinstance(x, str) for x in non_nan):
                    self.dls_data.drop(col, axis=1, inplace=True)
                    print(f"Dropped column {col} from DLS data due to strings")
                else:
                    self.dls_data[col] = datafunctions.sanitize_numeric_series(self.dls_data[col])
                    self.dls_data[col] = self.dls_data[col].interpolate(method='linear')
            else:
                self.dls_data[col] = datafunctions.sanitize_numeric_series(self.dls_data[col])
                self.dls_data[col] = self.dls_data[col].interpolate(method='linear')

    def apply_channel_mappings(self):
        self.dls_data = datafunctions.apply_channel_mappings(self.dls_data, self.CHANNEL_MAPPINGS, source_type=self.dls_run['name'].lower())
        self.track_data = datafunctions.apply_channel_mappings(self.track_data, self.CHANNEL_MAPPINGS, source_type=self.track_run['name'].lower())

    def apply_transformations(self):
        datafunctions.apply_transformations(self.dls_data, self.dls_run['name'].lower(), self.CHANNEL_TRANSFORMS)
        datafunctions.apply_transformations(self.track_data, self.track_run['name'].lower(), self.CHANNEL_TRANSFORMS)

    def apply_calculated_channels(self):
        datafunctions.apply_calculated_channels(self.dls_data, self.dls_run['name'].lower(), self.CALCULATED_CHANNELS)
        datafunctions.apply_calculated_channels(self.track_data, self.track_run['name'].lower(), self.CALCULATED_CHANNELS)

    def apply_lowpass_filters(self):
        """Apply low-pass Butterworth filters to specified channels        
        Args:
            df: DataFrame to filter
            source_type: 'dls' or 'track' to determine which filter config to use
        """
        datafunctions.apply_lowpass_filters(self.dls_data, self.LOW_PASS_FILTERS, self.FILTER_SAMPLE_RATE, 'dls')
        datafunctions.apply_lowpass_filters(self.track_data, self.LOW_PASS_FILTERS, self.FILTER_SAMPLE_RATE, 'track')

    def _get_plot_group(self, index, plot_type):
        """Safely fetch a configured plot-definition group."""
        if not self.PLOT_DEFINITIONS or len(self.PLOT_DEFINITIONS) <= index:
            return []
        plots = self.PLOT_DEFINITIONS[index]
        if plots is None:
            return []
        return plots

    def _normalise_reference_lines(self, reference_line_config):
        """Return reference lines as a list, regardless of scalar/tuple/list input."""
        if reference_line_config is None:
            return []
        if np.isscalar(reference_line_config):
            return [reference_line_config]
        return list(reference_line_config)

    def _resolve_waveform_axes(self):
        """Use sLap on both datasets when available, otherwise fall back to sample index."""
        if 'sLap' in self.dls_data.columns and 'sLap' in self.track_data.columns:
            return self.dls_data['sLap'], self.track_data['sLap'], 'sLap (m)'
        return self.dls_data.index, self.track_data.index, 'Sample'

    def _mask_waveform_discontinuities(self, x_values, y_values):
        """Break plotted lines where the x-axis resets and hide pre-lap samples."""
        x_series = pd.Series(x_values).reset_index(drop=True)
        y_series = pd.Series(y_values).reset_index(drop=True).copy()

        # Do not plot waveform data before the lap start when using sLap on the x-axis.
        negative_x_mask = x_series < 0
        x_series.loc[negative_x_mask] = np.nan
        y_series.loc[negative_x_mask] = np.nan

        if x_series.notna().sum() > 1:
            x_diff = x_series.diff()
            reset_mask = x_diff < 0
            y_series.loc[reset_mask] = np.nan

        return x_series, y_series

    def _style_waveform_x_axis(self, axes, xlabel):
        """Apply compact, professional x-axis styling to shared waveform axes."""
        bottom_axis = axes[-1]
        bottom_axis.set_xlabel(xlabel, fontsize=11, fontweight='bold', labelpad=10)
        bottom_axis.tick_params(axis='x', labelsize=10)

        if xlabel == 'sLap (m)':
            lap_end_candidates = []
            for ax in axes:
                x_min, x_max = ax.get_xlim()
                if np.isfinite(x_max) and x_max > 0:
                    lap_end_candidates.append(x_max)

            if lap_end_candidates:
                lap_end = max(lap_end_candidates)
                lap_end = np.ceil(lap_end / 100) * 100
                for ax in axes:
                    ax.set_xlim(0, lap_end)

            for ax in axes:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
                ax.grid(True, which='major', axis='x', alpha=0.32, linestyle='-', linewidth=0.5)
                ax.grid(True, which='minor', axis='x', alpha=0.16, linestyle='-', linewidth=0.35)

    def _sanitize_plot_filename(self, prefix, plot_name, suffix=""):
        """Create a filesystem-friendly filename for saved plots."""
        safe_name = (
            plot_name.replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('/', '_')
            .replace('\\', '_')
        )
        return f"{prefix}_{safe_name}{suffix}.png"

    def _resolve_plot_figsize(self, plot_filename, default_size, *, min_height=None):
        """Use template-derived aspect ratios when available, otherwise keep defaults."""
        default_width, default_height = default_size
        target_aspect = self.plot_aspect_ratios.get(plot_filename)

        if isinstance(target_aspect, (list, tuple)) and len(target_aspect) > 0:
            target_aspect = sum(target_aspect) / len(target_aspect)

        if target_aspect is None:
            width = default_width
            height = default_height
        else:
            height = default_height
            width = height * target_aspect

        if min_height is not None:
            height = max(height, min_height)
            if target_aspect is not None:
                width = height * target_aspect
            else:
                width = max(width, height * (default_width / default_height))

        return (width, height)

    def _add_axis_edge_padding(self, ax, x_pad_ratio=0.02, y_pad_ratio=0.03):
        """Add a small amount of whitespace around current axis limits."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            x_pad = (x_max - x_min) * x_pad_ratio
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            y_pad = (y_max - y_min) * y_pad_ratio
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _format_trendline_text(self, label, equation):
        """Format trendline text into a compact block without a label header."""
        formatted_lines = []
        for line in str(equation).splitlines():
            cleaned_line = line.strip()
            label_prefix = f"{label} "
            if cleaned_line.startswith(label_prefix):
                cleaned_line = cleaned_line[len(label_prefix):]
            if ": y =" in cleaned_line:
                cleaned_line = cleaned_line.replace(": y =", ":\ny =")
            formatted_lines.append(cleaned_line)
        return "\n".join(formatted_lines)

    def _format_gradient_error_text(self, equations_list, x_var=None, fit_split=None, y_var=None):
        """Create gradient error text using CAR as the baseline."""
        slope_map = {label: slopes for label, _, _, _, _, slopes in equations_list}
        dls_slopes = slope_map.get(self.dls_label)
        car_slopes = slope_map.get(self.track_label)

        if dls_slopes is None or car_slopes is None:
            return None

        def percent_error(dls_value, car_value):
            if car_value == 0:
                return None
            return ((dls_value - car_value) / car_value) * 100

        lines = ["Error in DLS vs CAR"]

        if isinstance(dls_slopes, tuple) and isinstance(car_slopes, tuple) and fit_split is not None:
            split_axis, split_value = fit_split
            axis_name = x_var if split_axis == 'x' else y_var
            conditions = [f"{axis_name} < {split_value}", f"{axis_name} >= {split_value}"]
            for condition, dls_value, car_value in zip(conditions, dls_slopes, car_slopes):
                error = percent_error(dls_value, car_value)
                if error is None:
                    lines.append(f"{condition}: undefined")
                else:
                    lines.append(f"{condition}: {error:+.1f}%")
        else:
            error = percent_error(float(dls_slopes), float(car_slopes))
            if error is None:
                lines.append("undefined")
            else:
                lines.append(f"{error:+.1f}%")

        return "\n".join(lines)

    def _select_trendline_anchor(self, ax, equations_list):
        """Choose the least crowded corner for vertically stacked trendline boxes."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_span = x_max - x_min
        y_span = y_max - y_min

        if x_span == 0 or y_span == 0:
            return (0.03, 0.97, 'left', 'top')

        point_sets = [(np.asarray(x_vals), np.asarray(y_vals)) for _, _, _, x_vals, y_vals, _ in equations_list]
        all_x = np.concatenate([x_vals for x_vals, _ in point_sets if len(x_vals) > 0])
        all_y = np.concatenate([y_vals for _, y_vals in point_sets if len(y_vals) > 0])

        box_width = 0.34
        stack_height = min(0.16 * max(len(equations_list), 1), 0.42)
        candidates = [
            (0.03, 0.97, 'left', 'top'),
            (0.97, 0.97, 'right', 'top'),
            (0.03, 0.03, 'left', 'bottom'),
            (0.97, 0.03, 'right', 'bottom'),
        ]

        best_candidate = candidates[0]
        best_score = None

        for x_anchor, y_anchor, h_align, v_align in candidates:
            if h_align == 'left':
                x0 = x_min + (x_anchor * x_span)
                x1 = x_min + ((x_anchor + box_width) * x_span)
            else:
                x0 = x_min + ((x_anchor - box_width) * x_span)
                x1 = x_min + (x_anchor * x_span)

            if v_align == 'top':
                y0 = y_min + ((y_anchor - stack_height) * y_span)
                y1 = y_min + (y_anchor * y_span)
            else:
                y0 = y_min + (y_anchor * y_span)
                y1 = y_min + ((y_anchor + stack_height) * y_span)

            covered_points = ((all_x >= x0) & (all_x <= x1) & (all_y >= y0) & (all_y <= y1)).sum()
            score = (covered_points, 0 if v_align == 'top' else 1)

            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (x_anchor, y_anchor, h_align, v_align)

        return best_candidate

    def _colorize_legend_labels(self, legend):
        """Match legend text colors to the plotted series colors."""
        if legend is None:
            return

        for text, handle in zip(legend.get_texts(), legend.legend_handles):
            color = None
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
            elif hasattr(handle, 'get_facecolor'):
                facecolor = handle.get_facecolor()
                if len(facecolor) > 0:
                    color = facecolor[0]

            if color is not None:
                text.set_color(color)

    def _prepare_waveform_channels(self, channels, axis_limits=None, reference_lines=None, subplot_heights=None):
        """Collect channels and per-axis options that exist in both datasets."""
        available_channels = []
        available_axis_limits = []
        available_ref_lines = []
        available_subplot_heights = []

        for ax_idx, channel in enumerate(channels):
            if channel not in self.dls_data.columns or channel not in self.track_data.columns:
                print(f"  Warning: Channel {channel} not found in both datasets, skipping")
                continue

            available_channels.append(channel)
            available_axis_limits.append(axis_limits[ax_idx] if axis_limits and ax_idx < len(axis_limits) else None)
            available_ref_lines.append(reference_lines[ax_idx] if reference_lines and ax_idx < len(reference_lines) else None)
            available_subplot_heights.append(subplot_heights[ax_idx] if subplot_heights and ax_idx < len(subplot_heights) else 1.0)

        return available_channels, available_axis_limits, available_ref_lines, available_subplot_heights
    
    def generate_waveform_plots(self):
        """Create waveform plots based on WAVEFORM_PLOT_DEFINITIONS"""
        plots = self._get_plot_group(0, 'waveform')

        for plot_def in plots:
            if len(plot_def) == 4:
                plot_name, channels, axis_limits, reference_lines = plot_def
                subplot_heights = None
            elif len(plot_def) == 5:
                plot_name, channels, axis_limits, reference_lines, subplot_heights = plot_def
            else:
                raise ValueError(
                    "Waveform plot definitions must be "
                    "[name, channels, axis_limits, reference_lines] or "
                    "[name, channels, axis_limits, reference_lines, subplot_heights]."
                )

            print(f"Creating waveform plot: {plot_name}")
            available_channels, available_axis_limits, available_ref_lines, available_subplot_heights = self._prepare_waveform_channels(
                channels,
                axis_limits=axis_limits,
                reference_lines=reference_lines,
                subplot_heights=subplot_heights
            )

            if len(available_channels) == 0:
                print(f"  Warning: No valid channels found for {plot_name}, skipping plot")
                continue

            plot_filename = self._sanitize_plot_filename("waveform", plot_name)
            min_waveform_height = 1.6 * sum(available_subplot_heights)
            figsize = self._resolve_plot_figsize(
                plot_filename,
                self.waveform_figsize,
                min_height=min_waveform_height
            )

            fig, axes = plt.subplots(
                len(available_channels),
                1,
                figsize=figsize,
                squeeze=False,
                sharex=True,
                gridspec_kw={'height_ratios': available_subplot_heights}
            )
            axes = axes.flatten()

            x_dls, x_track, xlabel = self._resolve_waveform_axes()

            # Plot each available channel
            for plot_idx, channel in enumerate(available_channels):
                ax = axes[plot_idx]
                x_dls_plot, y_dls_plot = self._mask_waveform_discontinuities(x_dls, self.dls_data[channel])
                x_track_plot, y_track_plot = self._mask_waveform_discontinuities(x_track, self.track_data[channel])

                ax.plot(x_dls_plot, y_dls_plot, linewidth=1.8, color=self.dls_run['color'],
                        label=self.dls_label, alpha=0.85, zorder=2)
                ax.plot(x_track_plot, y_track_plot, linewidth=1.8, color=self.track_run['color'],
                        label=self.track_label, alpha=0.85, zorder=2)

                

                ax.set_ylabel(
                    datafunctions.add_units_to_label(channel, units_map=self.units_map),
                    fontsize=8.2,
                    fontweight='bold',
                    rotation=0,
                    ha='right',
                    va='center'
                )
                ax.yaxis.set_label_coords(-0.035, 0.5)
                ax.grid(True, which='major', axis='y', alpha=0.28, linestyle='-', linewidth=0.45)
                ax.set_axisbelow(True)

                # Set axis limits if provided
                if available_axis_limits[plot_idx] is not None:
                    y_min, y_max = available_axis_limits[plot_idx]
                    if y_min is not None and y_max is not None:
                        ax.set_ylim(y_min, y_max)

                # Add reference lines if provided
                if available_ref_lines[plot_idx] is not None:
                    for ref_val in self._normalise_reference_lines(available_ref_lines[plot_idx]):
                        ax.axhline(y=ref_val, color='gray', linestyle='--', alpha=0.4, linewidth=0.9)

                # Style axes
                ax.tick_params(labelsize=10)

                if plot_idx < len(available_channels) - 1:
                    ax.tick_params(labelbottom=False)

            self._style_waveform_x_axis(axes, xlabel)

            plt.tight_layout(pad=0.3, h_pad=-0.8)

            # Add legend
            handles, labels = axes[0].get_legend_handles_labels()
            legend = axes[-1].legend(
                handles,
                labels,
                loc='lower left',
                bbox_to_anchor=(0.015, 0.02),
                fontsize=10.4,
                framealpha=0.99,
                borderpad=0.45,
                handlelength=2.2,
                labelspacing=0.35,
                handletextpad=0.55,
                prop={'family': 'Montserrat', 'weight': 'bold', 'size': 10.4}
            )
            self._colorize_legend_labels(legend)

            # Save plot
            fig.savefig(self.plots_dir / plot_filename, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.03)
            print(f"  Saved: {plot_filename}")
            plt.close(fig)
    
    def generate_scatter_plots(self):
        """Create scatter plots based on SCATTER_PLOT_DEFINITIONS"""
        plots = self._get_plot_group(1, 'scatter')

        for plot_def in plots:
            plot_name, (x_var, y_var), axis_limits, best_fit, fit_split = plot_def
            print(f"Creating scatter plot: {plot_name} ({x_var} vs {y_var})")

            plot_filename = self._sanitize_plot_filename("scatter", plot_name)
            fig, ax = plt.subplots(figsize=self._resolve_plot_figsize(plot_filename, self.scatter_FIGSIZE))
            ax.set_xlabel(datafunctions.add_units_to_label(x_var, self.units_map), fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(datafunctions.add_units_to_label(y_var, self.units_map), fontsize=14, fontweight='bold', labelpad=10)

            equations_list = []

            # Plot DLS data
            if x_var in self.dls_data.columns and y_var in self.dls_data.columns:
                x_data = self.dls_data[x_var].dropna()
                y_data = self.dls_data[y_var][x_data.index].dropna()
                x_data = x_data[y_data.index]

                if best_fit == 0:
                    # No fit line
                    datafunctions.plot_scatter(ax, x_data.values, y_data.values,
                                             self.dls_label, self.dls_run['color'],
                                             self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                                             x_var, y_var)
                elif best_fit == 1:
                    # Single fit line
                    result = datafunctions.plot_scatter_with_1fit(ax, x_data.values, y_data.values,
                                                        self.dls_label, self.dls_run['color'],
                                                        self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                                                        x_var, y_var, FIT_LINE_X_LIMITS=None)
                    if result[0] and result[3]:
                        equations_list.append((self.dls_label, result[3], self.dls_run['color'], x_data.values, y_data.values, result[1]))
                elif best_fit == 2:
                    # Double fit line at configured split point
                    result = datafunctions.plot_scatter_with_double_fit(ax, x_data.values, y_data.values,
                                                              self.dls_label, self.dls_run['color'],
                                                              self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                                                              x_var, y_var, fit_split=fit_split)
                    if result[0] and result[3]:
                        equations_list.append((self.dls_label, result[3], self.dls_run['color'], x_data.values, y_data.values, result[1]))
            else:
                print(f"  Warning: Channels {x_var} and/or {y_var} not found in DLS data, skipping DLS plot")

            # Plot Track data
            if x_var in self.track_data.columns and y_var in self.track_data.columns:
                x_data = self.track_data[x_var].dropna()
                y_data = self.track_data[y_var][x_data.index].dropna()
                x_data = x_data[y_data.index]

                if best_fit == 0:
                    # No fit line
                    datafunctions.plot_scatter(ax, x_data.values, y_data.values,
                                             self.track_label, self.track_run['color'],
                                             self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                                             x_var, y_var)
                elif best_fit == 1:
                    # Single fit line
                    result = datafunctions.plot_scatter_with_1fit(ax, x_data.values, y_data.values,
                                                        self.track_label, self.track_run['color'],
                                                        self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                                                        x_var, y_var, FIT_LINE_X_LIMITS=None)
                    if result[0] and result[3]:
                        equations_list.append((self.track_label, result[3], self.track_run['color'], x_data.values, y_data.values, result[1]))
                elif best_fit == 2:
                    # Double fit line at configured split point
                    result = datafunctions.plot_scatter_with_double_fit(ax, x_data.values, y_data.values,
                                                              self.track_label, self.track_run['color'],
                                                              self.SCATTER_TRANSPARENCY, self.SCATTER_DOT_SIZE,
                                                              x_var, y_var, fit_split=fit_split)
                    if result[0] and result[3]:
                        equations_list.append((self.track_label, result[3], self.track_run['color'], x_data.values, y_data.values, result[1]))
            else:
                print(f"  Warning: Channels {x_var} and/or {y_var} not found in Track data, skipping Track plot")

            # Set axis limits if provided
            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None and xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)

            self._add_axis_edge_padding(ax, x_pad_ratio=0.02, y_pad_ratio=0.03)

            ax.grid(True, which='major', alpha=0.26, linestyle='-', linewidth=0.4)
            ax.set_axisbelow(True)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            if y_min <= 0 <= y_max:
                ax.axhline(0, color="#5E5E5E", linewidth=1, alpha=0.8, zorder=1.5)
            if x_min <= 0 <= x_max:
                ax.axvline(0, color="#5E5E5E", linewidth=1, alpha=0.8, zorder=1.5)
            ax.tick_params(labelsize=10, direction='out', length=4, width=0.8, pad=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.9)
            ax.spines['bottom'].set_linewidth(0.9)

            if equations_list:
                anchor_info = self._display_equations(ax, equations_list)
                gradient_error_text = self._format_gradient_error_text(equations_list, x_var=x_var, fit_split=fit_split, y_var=y_var)
                if gradient_error_text:
                    self._display_gradient_error(ax, gradient_error_text, anchor_info)

            # Only add legend if there are artists to display
            if ax.get_legend_handles_labels()[0]:
                legend = ax.legend(
                    fontsize=10,
                    framealpha=1,
                    loc='best',
                    borderpad=0.35,
                    handlelength=1.8,
                    prop={'family': 'Montserrat', 'weight': 'bold', 'size': 10}
                )
                self._colorize_legend_labels(legend)

            plt.tight_layout(pad=0.25)

            # Save plot
            fig.savefig(
                self.plots_dir / plot_filename,
                dpi=300,
                pad_inches=0.05,
                facecolor='white'
            )
            print(f"  Saved: {plot_filename}")
            plt.close(fig)

    def generate_psd_plots(self):
        """Create PSD plots based on PSD_PLOT_DEFINITIONS"""
        plots = self._get_plot_group(2, 'psd')

        for plot_def in plots:
            if len(plot_def) == 3:
                plot_name, channel, axis_limits = plot_def
                nperseg = 256
                log_scale = True
            elif len(plot_def) == 4:
                plot_name, channel, axis_limits, nperseg = plot_def
                log_scale = True
            elif len(plot_def) == 5:
                plot_name, channel, axis_limits, nperseg, log_scale = plot_def
            else:
                raise ValueError(
                    "PSD plot definitions must be [name, channel, axis_limits], "
                    "[name, channel, axis_limits, nperseg], or "
                    "[name, channel, axis_limits, nperseg, log_scale]."
                )

            print(f"Creating PSD plot: {plot_name} ({channel})")

            if channel not in self.dls_data.columns or channel not in self.track_data.columns:
                print(f"  Warning: Channel {channel} not found in both datasets, skipping PSD plot")
                continue

            plot_filename = self._sanitize_plot_filename("psd", plot_name)
            fig, ax = plt.subplots(figsize=self._resolve_plot_figsize(plot_filename, self.psd_FIGSIZE))
            ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold', labelpad=10)
            ax.set_ylabel(f'{datafunctions.add_units_to_label(channel, self.units_map)} PSD', fontsize=13, fontweight='bold', labelpad=10)

            dls_freq, dls_power = datafunctions.calculate_psd(self.dls_data[channel], self.FILTER_SAMPLE_RATE, nperseg=nperseg)
            track_freq, track_power = datafunctions.calculate_psd(self.track_data[channel], self.FILTER_SAMPLE_RATE, nperseg=nperseg)

            if dls_freq is None or track_freq is None:
                print(f"  Warning: Not enough valid data for PSD plot {plot_name}, skipping")
                plt.close(fig)
                continue

            plot_func = ax.semilogy if log_scale else ax.plot
            plot_func(dls_freq, dls_power, linewidth=1.8, color=self.dls_run['color'], label=self.dls_label, alpha=0.9)
            plot_func(track_freq, track_power, linewidth=1.8, color=self.track_run['color'], label=self.track_label, alpha=0.9)

            if axis_limits:
                (xmin, xmax), (ymin, ymax) = axis_limits
                if xmin is not None and xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)

            self._add_axis_edge_padding(ax, x_pad_ratio=0.02, y_pad_ratio=0.04)

            ax.grid(True, which='major', alpha=0.26, linestyle='-', linewidth=0.4)
            ax.grid(True, which='minor', alpha=0.14, linestyle='-', linewidth=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=10, direction='out', length=4, width=0.8, pad=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.9)
            ax.spines['bottom'].set_linewidth(0.9)

            legend = ax.legend(
                fontsize=10,
                framealpha=1,
                loc='best',
                borderpad=0.35,
                handlelength=1.8,
                prop={'family': 'Montserrat', 'weight': 'bold', 'size': 10}
            )
            self._colorize_legend_labels(legend)

            plt.tight_layout(pad=0.25)

            fig.savefig(
                self.plots_dir / plot_filename,
                dpi=300,
                pad_inches=0.05,
                facecolor='white'
            )
            print(f"  Saved: {plot_filename}")
            plt.close(fig)

    def _display_equations(self, ax, equations_list):
        """Display vertically stacked trendline text in the least crowded corner."""
        if len(equations_list) == 0:
            return None

        x_anchor, y_anchor, h_align, v_align = self._select_trendline_anchor(ax, equations_list)
        y_step = 0.118
        box_edges = []

        for idx, (label, equation, color, _, _, _) in enumerate(equations_list):
            y_pos = y_anchor - (idx * y_step) if v_align == 'top' else y_anchor + (idx * y_step)
            trendline_text = self._format_trendline_text(label, equation)
            line_count = trendline_text.count('\n') + 1
            estimated_height = 0.03 + (0.036 * line_count)

            if v_align == 'top':
                box_edges.append((y_pos - estimated_height, y_pos))
            else:
                box_edges.append((y_pos, y_pos + estimated_height))

            ax.text(
                x_anchor,
                y_pos,
                trendline_text,
                transform=ax.transAxes,
                fontsize=9.1,
                verticalalignment=v_align,
                horizontalalignment=h_align,
                bbox=dict(
                    boxstyle='round,pad=0.28',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=1.6
                ),
                color=color,
                fontweight='bold',
                family='Montserrat',
                linespacing=1.08
            )

        return x_anchor, h_align, v_align, box_edges

    def _display_gradient_error(self, ax, gradient_error_text, anchor_info):
        """Display gradient error summary directly below the stacked trendline boxes."""
        if anchor_info is None:
            return

        x_anchor, h_align, v_align, box_edges = anchor_info
        padding = 0.04
        if v_align == 'top':
            lowest_edge = min(bottom for bottom, _ in box_edges)
            y_pos = lowest_edge - padding
        else:
            highest_edge = max(top for _, top in box_edges)
            y_pos = highest_edge + padding

        ax.text(
            x_anchor,
            y_pos,
            gradient_error_text,
            transform=ax.transAxes,
            fontsize=8.8,
            verticalalignment=v_align,
            horizontalalignment=h_align,
            bbox=dict(
                boxstyle='round,pad=0.26',
                facecolor='white',
                alpha=0.9,
                edgecolor='#6E6E6E',
                linewidth=1.2
            ),
            color='#3F3F3F',
            fontweight='bold',
            family='Montserrat',
            linespacing=1.08
        )

    def plot_all(self):
        """Generate all plots based on definitions"""
        self.generate_waveform_plots()
        self.generate_scatter_plots()
        self.generate_psd_plots()

        print(f"\nAll plots saved to: {self.plots_dir}")
