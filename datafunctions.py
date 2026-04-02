import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import butter, filtfilt, welch

def convert_yes_no_to_binary(df):
    """Convert YES/NO strings to 1/0 integers in all columns"""
    columns_converted = []

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name in ['string', 'str']:
            non_nan = df[col].dropna()
            if len(non_nan) > 0:
                # Check if column contains YES/NO strings
                str_values = [str(x).upper() for x in non_nan if isinstance(x, str)]
                if any(val in ['YES', 'NO'] for val in str_values):
                    # Convert YES to 1 and NO to 0
                    df[col] = df[col].astype(str).str.upper().replace({'YES': 1, 'NO': 0})
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    columns_converted.append(col)

    if columns_converted:
        print(f"  Converted YES/NO to 1/0 in columns: {', '.join(columns_converted)}")

    return df

def sanitize_numeric_series(series):
    """Replace known integer sentinel values and infinities with NaN."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    numeric_series = numeric_series.replace([int64_min, int64_max, -np.inf, np.inf], np.nan)
    return numeric_series


def apply_channel_mappings(df, channel_mappings, source_type):
    """Apply channel name mappings to standardize column names"""
    # Apply channel mappings, but skip if target column already exists (prevents duplicates)
    if channel_mappings[source_type] is None:
        print(f"  No channel mappings defined for {source_type.upper()} data, skipping mapping")
        return df
    rename_dict = {dil_name: track_name for dil_name, track_name in channel_mappings[source_type].items() 
                    if dil_name in df.columns and track_name not in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"  Renamed {len(rename_dict)} channels in {source_type.upper()} data")
    return df

def apply_transformations(df, source_type, channel_transforms):
    """Apply transformations to specified channels"""
    
    transforms = channel_transforms[source_type]   
    if transforms is None:
        print(f"  No channel transformations defined for {source_type.upper()} data, skipping transformations")
        return df 
    for channel, transform_func in transforms.items():
        if channel.lower() == 'all':
            # Apply to all channels
            for col in df.columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = transform_func(df[col])
            print(f"  Applied transformation to all channels in {source_type.upper()} data")
        if channel in df.columns:
            # Convert to numeric first, coercing errors to NaN
            df[channel] = pd.to_numeric(df[channel], errors='coerce')
            df[channel] = transform_func(df[channel])
            print(f"  Applied transformation to {channel}")
        else:
            print(f"  Warning: Channel {channel} not found in {source_type.upper()} data, skipping transformation")
    
    return df

def apply_calculated_channels(df, source_type, calculated_channels):
    """Create new channels from existing columns using user-defined functions."""
    if calculated_channels is None:
        print(f"  No calculated channels defined for {source_type.upper()} data, skipping calculations")
        return df

    # Support a shared config for both datasets, while remaining compatible with
    # the previous {'dls': {...}, 'track': {...}} structure.
    if source_type in calculated_channels and isinstance(calculated_channels[source_type], dict):
        channel_calculations = calculated_channels[source_type]
    else:
        channel_calculations = calculated_channels

    if channel_calculations is None:
        print(f"  No calculated channels defined for {source_type.upper()} data, skipping calculations")
        return df

    for channel_name, calculation_func in channel_calculations.items():
        try:
            df[channel_name] = pd.to_numeric(calculation_func(df), errors='coerce')
            print(f"  Calculated channel {channel_name}")
        except KeyError as e:
            print(f"  Warning: Missing source channel {e} for calculated channel {channel_name} in {source_type.upper()} data, skipping")
        except Exception as e:
            print(f"  Warning: Failed to calculate channel {channel_name} in {source_type.upper()} data: {e}")

    return df

def apply_lowpass_filters(df, low_pass_filters, sample_rate, source_type):
    channels_filtered = []

    filter_all = False
    all_config = None
    channels_to_skip = []

    for channel, filter_config in low_pass_filters.items():
        if channel.lower() == 'all':
            filter_all = True
            all_config = filter_config
            continue  # Process 'all' filters after specific channel filters

        if channel not in df.columns:
            print(f"  Warning: Channel {channel} not found in {source_type.upper()} data, skipping filter")
            continue
        
        channels_to_skip.append(channel)

        # Convert to numeric first
        df[channel] = pd.to_numeric(df[channel], errors='coerce')
        
        # Check if filter_config has source-specific settings
        if source_type in filter_config:
            # Use source-specific filter config (e.g., different cutoffs for dil vs track)
            config = filter_config[source_type]
        elif 'cutoff' in filter_config:
            # Use common filter config for all sources
            config = filter_config
        else:
            print(f"  Warning: No valid filter configuration found for {channel} in {source_type.upper()} data, skipping filter")
            # No valid config found
            continue
        
        cutoff = config['cutoff']
        if cutoff == 0:
            print(f"  Info: Cutoff frequency is 0 Hz for {channel}, not filtering this channel")
            continue

        elif cutoff < 0:
            print(f"  Warning: Invalid cutoff frequency {cutoff} Hz for {channel}, skipping filter")
            continue
        order = config.get('order', 2)
        
        # Get data
        data = df[channel].values
        
        # Skip if not enough data or all NaN
        if len(data) <= order * 3 or np.all(np.isnan(data)):
            print(f"  Warning: Not enough valid data for {channel}, skipping filter")
            continue
        
        # Design Butterworth filter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        
        # Prevent cutoff frequency from being too high
        if normal_cutoff >= 1.0:
            print(f"  Warning: Cutoff {cutoff} Hz too high for sample rate {sample_rate} Hz, skipping {channel}")
            continue
        
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Handle NaN values
        nan_mask = np.isnan(data)
        
        if not np.any(nan_mask):
            # No NaN values - apply filter directly
            df[channel] = filtfilt(b, a, data)
        else:
            # Interpolate NaN values, filter, then restore NaN positions
            data_interp = pd.Series(data).interpolate(method='linear', limit_direction='both').values
            filtered_data = filtfilt(b, a, data_interp)
            filtered_data[nan_mask] = np.nan
            df[channel] = filtered_data
        
        channels_filtered.append(f"{channel}@{cutoff}Hz")
        
    
    if filter_all and all_config is not None:
        for col in df.columns:
            if col in channels_to_skip:
                continue  # Skip channels that were already filtered with specific configs
            
            # Convert to numeric first
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            cutoff = all_config['cutoff']
            order = all_config.get('order', 2)
            
            data = df[col].values
            
            if len(data) <= order * 3 or np.all(np.isnan(data)):
                print(f"  Warning: Not enough valid data for {col}, skipping filter")
                continue
            
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff / nyquist
            
            if normal_cutoff >= 1.0:
                print(f"  Warning: Cutoff {cutoff} Hz too high for sample rate {sample_rate} Hz, skipping {col}")
                continue
            
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            
            nan_mask = np.isnan(data)
            
            if not np.any(nan_mask):
                df[col] = filtfilt(b, a, data)
            else:
                data_interp = pd.Series(data).interpolate(method='linear', limit_direction='both').values
                filtered_data = filtfilt(b, a, data_interp)
                filtered_data[nan_mask] = np.nan
                df[col] = filtered_data
            
            channels_filtered.append(f"{col}@{cutoff}Hz")

    if channels_filtered:
        print(f"  Applied low-pass filters: {', '.join(channels_filtered)}")
    
    return df


def calculate_psd(signal, sample_rate, nperseg=256):
    """Calculate PSD using Welch's method for a 1D numeric signal."""
    signal = pd.to_numeric(pd.Series(signal), errors='coerce').dropna().values
    if len(signal) < 8:
        return None, None

    nperseg = min(nperseg, len(signal))
    if nperseg < 8:
        return None, None

    frequencies, power = welch(signal, fs=sample_rate, nperseg=nperseg)
    return frequencies, power

def find_best_text_position(ax):
    """
    Analyze data density in plot corners and return the best position for text.
    Returns (x_pos, y_pos, h_align, v_align) for the least crowded corner.
    """
    # Get all scatter plot data from the axis
    all_x = []
    all_y = []
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets) > 0:
            all_x.extend(offsets[:, 0])
            all_y.extend(offsets[:, 1])
    
    # If no data, default to top-right
    if len(all_x) == 0:
        return 0.95, 0.95, 'right', 'top'
    
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Normalize data to 0-1 range
    x_norm = [(x - xlim[0]) / (xlim[1] - xlim[0]) for x in all_x]
    y_norm = [(y - ylim[0]) / (ylim[1] - ylim[0]) for y in all_y]
    
    # Count points in each corner (using 40% threshold from each edge)
    corners = {
        'top_left': 0,
        'top_right': 0,
        'bottom_left': 0,
        'bottom_right': 0
    }
    
    threshold = 0.4  # Consider the outer 40% of each dimension as "corner"
    
    for x, y in zip(x_norm, y_norm):
        # Top corners (y > 0.6)
        if y > (1 - threshold):
            if x < threshold:
                corners['top_left'] += 1
            elif x > (1 - threshold):
                corners['top_right'] += 1
        # Bottom corners (y < 0.4)
        elif y < threshold:
            if x < threshold:
                corners['bottom_left'] += 1
            elif x > (1 - threshold):
                corners['bottom_right'] += 1
    
    # Find corner with fewest points
    best_corner = min(corners, key=corners.get)
    
    # Map corner to text position parameters
    positions = {
        'top_left': (0.05, 0.95, 'left', 'top'),
        'top_right': (0.95, 0.95, 'right', 'top'),
        'bottom_left': (0.05, 0.05, 'left', 'bottom'),
        'bottom_right': (0.95, 0.05, 'right', 'bottom')
    }
    
    return positions[best_corner]

def plot_scatter(ax, x_data, y_data, label, color, alpha, size, x_var='', y_var=''):
    """Plot scatter with optional fit line"""
    if len(x_data) == 0:
        print(f"  Warning: No data to plot for {label} ({x_var} vs {y_var}), skipping")
        return False, None, None
    
    # Plot scatter
    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color, 
                marker='o', label=label, edgecolors='none')
    
    return True, None, None

def plot_scatter_with_1fit(ax, x_data, y_data, label, color, alpha, size, x_var='', y_var='', FIT_LINE_X_LIMITS=None):
    """Plot scatter with 1st degree fit line and equation"""
    if len(x_data) == 0:
        print(f"  Warning: No data to plot for {label} ({x_var} vs {y_var}), skipping")
        return False, None, None, None, None

    # Plot scatter
    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color,
                marker='o', label=label, edgecolors='none')

    # Standard single fit line
    # Determine x-axis range for fit line
    if FIT_LINE_X_LIMITS is not None:
        x_min, x_max = FIT_LINE_X_LIMITS
    else:
        x_min, x_max = x_data.min(), x_data.max()
    x_range = np.linspace(x_min, x_max, 100)

    slope, intercept, r_value, _, _ = linregress(x_data, y_data)
    y_fit = slope * x_range + intercept

    ax.plot(x_range, y_fit, color="#000000", linewidth=1, alpha=1, linestyle='-', zorder=3)

    # Create equation text (without R^2)
    equation = f'y = {slope:.3f}x + {intercept:.3f}'

    return True, slope, intercept, equation, color

def plot_scatter_with_double_fit(ax, x_data, y_data, label, color, alpha, size, x_var='', y_var='', fit_split=None):
    """Plot scatter with dual fit lines and equations"""
    if len(x_data) == 0:
        print(f"  Warning: No data to plot for {label} ({x_var} vs {y_var}), skipping")
        return False, None, None

    # Plot scatter
    ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color,
                marker='o', label=label, edgecolors='none')

    if fit_split is not None:
        split_axis, split_value = fit_split
        if split_axis == 'x':
            mask_before = x_data < split_value
            mask_after = x_data >= split_value
            axis_name = x_var if x_var else 'x'
        elif split_axis == 'y':
            mask_before = y_data < split_value
            mask_after = y_data >= split_value
            axis_name = y_var if y_var else 'y'
        else:
            print(f"  Warning: Unsupported fit split axis '{split_axis}' for {label} ({x_var} vs {y_var})")
            return plot_scatter_with_1fit(ax, x_data, y_data, label, color, alpha, size, x_var, y_var)

        equation_text = ""
        slope_before = intercept_before = slope_after = intercept_after = None

        if mask_before.sum() > 1:
            x_before = x_data[mask_before]
            y_before = y_data[mask_before]
            slope_before, intercept_before, r_before, _, _ = linregress(x_before, y_before)

            x_range_before = np.linspace(x_before.min(), x_before.max(), 50)
            y_fit_before = slope_before * x_range_before + intercept_before

            ax.plot(x_range_before, y_fit_before, color="#000000", linewidth=1, alpha=1, linestyle='--', zorder=3)
            equation_text += f'{label} ({axis_name} < {split_value}): y = {slope_before:.3f}x + {intercept_before:.3f}\n'

        if mask_after.sum() > 1:
            x_after = x_data[mask_after]
            y_after = y_data[mask_after]
            slope_after, intercept_after, r_after, _, _ = linregress(x_after, y_after)

            x_range_after = np.linspace(x_after.min(), x_after.max(), 50)
            y_fit_after = slope_after * x_range_after + intercept_after
            ax.plot(x_range_after, y_fit_after, color="#000000", linewidth=1, alpha=1, linestyle='-.', zorder=3)
            equation_text += f'({axis_name} >= {split_value}): y = {slope_after:.3f}x + {intercept_after:.3f}'

        return True, (slope_before, slope_after), (intercept_before, intercept_after), equation_text.rstrip(), color

    else:
        print(f"  Warning: fit_split not set, reverting to single fit for {label} ({x_var} vs {y_var})")
        return plot_scatter_with_1fit(ax, x_data, y_data, label, color, alpha, size, x_var, y_var)

def add_units_to_label(var_name, units_map):
    """Add appropriate units to variable names for axis labels"""
    var_lower = var_name.lower()

    for key in units_map:
        if key.lower() == var_lower:
            return f'{var_name} ({units_map[key]})'
    return var_name
