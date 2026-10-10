import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.signal import savgol_filter
    from scipy.integrate import cumulative_trapezoid
    scipy_available = True
except ImportError:
    scipy_available = False
    print("Warning: Unable to import scipy.signal or scipy.integrate. Filtering and integration may be affected.")

# --- Configuration ---
file_path = 'PD_Ex12.txt'
time_interval_ps = 80
header_skip_keyword = 'di/dt'
samples_per_experiment = 375000
points_to_skip = 181250
start_time_us_after_skip = -0.5
experiment_to_display = 12

PERCENT_THRESHOLD_INNER = 50
PERCENT_THRESHOLD_OUTER = 70

SG_WINDOW_PULSE_SEGMENT = 11
SG_WINDOW_FLANK_SEGMENTS = 1001
SG_POLYORDER_SEGMENT = 3

MAX_POSITIVE_DEVIATION_PERCENT = 40

# --- Plot Axis Configuration ---
PLOT_XLIM_CONFIG = (-0.5, 6.0)

PLOT_YLIM_AX0_RAW          = (-3.5, 3.5)
PLOT_YLIM_AX1_PULSE_DETECT = (-3.5, 3.5) 
PLOT_YLIM_AX2_MULTI_FILTER = (-3.5, 3.5) 
PLOT_YLIM_AX3_INTEGRATED   = (114, 120)  

# --- 1. Read Data from File ---
all_data_values = []
try:
    with open(file_path, 'r') as f:
        start_reading_data = False
        for line in f:
            line_stripped = line.strip()
            if not start_reading_data:
                if header_skip_keyword in line_stripped: start_reading_data = True
                continue
            if start_reading_data and line_stripped:
                try: all_data_values.append(float(line_stripped))
                except ValueError: print(f"Warning: Cannot convert '{line_stripped}' to number. Skipping line.")
except FileNotFoundError: print(f"Error: File '{file_path}' not found."); exit()
except Exception as e: print(f"Error reading file: {e}"); exit()
if not all_data_values: print("Error: No numeric data found or file empty after header."); exit()

# --- 2. Split Data into Experiments ---
num_total_samples = len(all_data_values)
num_experiments_available = num_total_samples // samples_per_experiment
if num_experiments_available == 0: print(f"Error: Insufficient data (need {samples_per_experiment}, have {num_total_samples})."); exit()
print(f"Found {num_total_samples} samples. Can split into {num_experiments_available} experiments.")
if num_total_samples % samples_per_experiment != 0: print(f"Warning: Last {num_total_samples % samples_per_experiment} samples will be ignored.")

# --- 3. Select Desired Experiment Data ---
experiment_index = experiment_to_display - 1
if experiment_index < 0 or experiment_index >= num_experiments_available: print(f"Error: Experiment {experiment_to_display} does not exist."); exit()
start_data_index = experiment_index * samples_per_experiment
end_data_index = start_data_index + samples_per_experiment
full_data = all_data_values[start_data_index:end_data_index]
print(f"Experiment {experiment_to_display} has {len(full_data)} samples.")

# --- 4. Trim Data ---
if len(full_data) <= points_to_skip: print(f"Error: Data samples ({len(full_data)}) <= points to skip ({points_to_skip})."); exit()
trimmed_data = np.array(full_data[points_to_skip:])
print(f"After skipping first {points_to_skip} points, {len(trimmed_data)} samples remain.")

# --- 5. Create Time Axis ---
time_interval_s = time_interval_ps * 1e-12; time_interval_us = time_interval_s * 1e6
num_remaining_samples = len(trimmed_data)
time_axis_us_trimmed = np.arange(start_time_us_after_skip, start_time_us_after_skip + num_remaining_samples * time_interval_us, time_interval_us)
if len(time_axis_us_trimmed) != num_remaining_samples:
     time_axis_us_trimmed = np.linspace(start_time_us_after_skip, start_time_us_after_skip + (num_remaining_samples -1) * time_interval_us, num_remaining_samples)
time_unit_display = 'µs'
print(f"Time axis spans from {time_axis_us_trimmed[0]:.2f} {time_unit_display} to {time_axis_us_trimmed[-1]:.2f} {time_unit_display}.")

# --- 6. Process Data ---
# Variable definitions and explanations:
smoothed_data = None      # Savitzky-Golay smoothed data for pulse detection
trend_line_params = None  # Tuple of (slope, intercept) for the base trend line (2-4 µs)
pulse_start_time = None   # Start time of the detected pulse
pulse_end_time = None     # End time of the detected pulse
pulse_duration = None     # Duration of the detected pulse
t_analysis = None         # Time points within the analysis window (2-4 µs)
thresh_H = None           # High parallel threshold array based on base trend line
thresh_L = None           # Low parallel threshold array based on base trend line
filtered_data = None      # Multi-segment filtered data (separate parameters for pulse and flanks)
t_clip_plot = None        # Time points for plotting the clipping limit band
clip_upper_plot = None    # Upper bound of the clipping limit band for plotting
clip_lower_plot = None    # Lower bound of the clipping limit band for plotting
integrated_data = None    # Integrated data using cumulative trapezoidal rule

sg_window_initial_smooth = 51; sg_polyorder_initial_smooth = 3
attempt_savgol_processing = scipy_available
valid_initial_savgol_params = False
if attempt_savgol_processing:
    if sg_window_initial_smooth > sg_polyorder_initial_smooth and sg_window_initial_smooth % 2 != 0 : valid_initial_savgol_params = True
    else: print(f"Warning: Invalid initial Savitzky-Golay parameters (W: {sg_window_initial_smooth}, P: {sg_polyorder_initial_smooth})")
    if valid_initial_savgol_params:
        try:
            smoothed_data = savgol_filter(trimmed_data, sg_window_initial_smooth, sg_polyorder_initial_smooth, deriv=0, delta=time_interval_us, mode='nearest')
            pulse_mask = (time_axis_us_trimmed >= 2.0) & (time_axis_us_trimmed <= 4.0)
            if smoothed_data is not None and np.any(pulse_mask):
                t_analysis = time_axis_us_trimmed[pulse_mask]
                data_for_analysis = smoothed_data[pulse_mask]
                if data_for_analysis.size >= 2:
                    slope, intercept = np.polyfit(t_analysis, data_for_analysis, 1)
                    trend_line_params = (slope, intercept)
                    base_line = slope * t_analysis + intercept
                    avg_base_line = np.mean(base_line)
                    dec_thresh_in = PERCENT_THRESHOLD_INNER / 100.0; dec_thresh_out = PERCENT_THRESHOLD_OUTER / 100.0
                    off_in = dec_thresh_in * abs(avg_base_line); off_out = dec_thresh_out * abs(avg_base_line)
                    thresh_H = base_line - off_in
                    thresh_L = base_line - off_out
                    state = "IDLE"; temp_t_start_pulse = None
                    for i in range(len(data_for_analysis)):
                        current_time = t_analysis[i]; current_data_point = data_for_analysis[i]
                        current_Th_H_val = thresh_H[i]; current_Th_L_val = thresh_L[i]
                        if state == "IDLE":
                            if current_data_point < current_Th_H_val: temp_t_start_pulse = current_time; state = "ARMED_FALL"
                        elif state == "ARMED_FALL":
                            if current_data_point < current_Th_L_val: pulse_start_time = temp_t_start_pulse; state = "IN_PULSE"
                            elif current_data_point >= current_Th_H_val: state = "IDLE"
                        elif state == "IN_PULSE":
                            if current_data_point > current_Th_L_val: state = "ARMED_RISE"
                        elif state == "ARMED_RISE":
                            if current_data_point > current_Th_H_val: pulse_end_time = current_time; break
                            elif current_data_point <= current_Th_L_val: state = "IN_PULSE"
                    if pulse_start_time is not None and pulse_end_time is not None: pulse_duration = pulse_end_time - pulse_start_time; print(f"First negative pulse detected: Start: {pulse_start_time:.3f}, End: {pulse_end_time:.3f}, Duration: {pulse_duration:.3f} {time_unit_display}")
                    elif pulse_start_time is not None: print(f"Negative pulse start detected at {pulse_start_time:.3f} {time_unit_display} but end point not found in range.")
                    else: print("No negative pulse found meeting conditions in 2-4 µs range.")
                else: print("Insufficient initial smoothed data in 2-4 µs range to calculate trend line and analyze pulse.")
            else: print("2-4 µs time range is out of bounds for initial smoothed data time axis, or data is None.")
        except Exception as e: print(f"Error processing data (initial smooth & pulse detection): {e}"); smoothed_data = None; trend_line_params = None;
    if scipy_available: # Multi-segment filtering
        valid_sg_pulse_params = (SG_WINDOW_PULSE_SEGMENT > SG_POLYORDER_SEGMENT and SG_WINDOW_PULSE_SEGMENT % 2 != 0)
        valid_sg_flank_params = (SG_WINDOW_FLANK_SEGMENTS > SG_POLYORDER_SEGMENT and SG_WINDOW_FLANK_SEGMENTS % 2 != 0)
        filtered_segments = []
        if pulse_start_time is not None and pulse_end_time is not None:
            idx_start_pulse = np.searchsorted(time_axis_us_trimmed, pulse_start_time); idx_end_pulse = np.searchsorted(time_axis_us_trimmed, pulse_end_time)
            idx_start_pulse = max(0, min(idx_start_pulse, len(trimmed_data))); idx_end_pulse = max(idx_start_pulse, min(idx_end_pulse, len(trimmed_data)))
            current_idx = 0
            if idx_start_pulse > current_idx:
                seg_data = trimmed_data[current_idx:idx_start_pulse]
                if valid_sg_flank_params and len(seg_data) >= SG_WINDOW_FLANK_SEGMENTS: 
                    try: filtered_segments.append(savgol_filter(seg_data, SG_WINDOW_FLANK_SEGMENTS, SG_POLYORDER_SEGMENT, mode='interp')) 
                    except ValueError: filtered_segments.append(seg_data)
                else: filtered_segments.append(seg_data)
            current_idx = idx_start_pulse
            if idx_end_pulse > current_idx:
                seg_data = trimmed_data[current_idx:idx_end_pulse]
                if valid_sg_pulse_params and len(seg_data) >= SG_WINDOW_PULSE_SEGMENT: 
                    try: filtered_segments.append(savgol_filter(seg_data, SG_WINDOW_PULSE_SEGMENT, SG_POLYORDER_SEGMENT, mode='interp')) 
                    except ValueError: filtered_segments.append(seg_data)
                else: filtered_segments.append(seg_data)
            current_idx = idx_end_pulse
            if current_idx < len(trimmed_data):
                seg_data = trimmed_data[current_idx:]
                if valid_sg_flank_params and len(seg_data) >= SG_WINDOW_FLANK_SEGMENTS: 
                    try: filtered_segments.append(savgol_filter(seg_data, SG_WINDOW_FLANK_SEGMENTS, SG_POLYORDER_SEGMENT, mode='interp')) 
                    except ValueError: filtered_segments.append(seg_data)
                else: filtered_segments.append(seg_data)
            if filtered_segments: filtered_data = np.concatenate(filtered_segments)
            if filtered_data is not None and len(filtered_data) != len(trimmed_data): filtered_data = None
        if filtered_data is None:
            if valid_sg_flank_params and len(trimmed_data) >= SG_WINDOW_FLANK_SEGMENTS : 
                try: filtered_data = savgol_filter(trimmed_data, SG_WINDOW_FLANK_SEGMENTS, SG_POLYORDER_SEGMENT, mode='interp') 
                except ValueError: filtered_data = np.copy(trimmed_data)
            else: filtered_data = np.copy(trimmed_data)
    else: filtered_data = np.copy(trimmed_data)
    if filtered_data is not None and trend_line_params is not None: # Clipping
        slope_ref, intercept_ref = trend_line_params
        if np.any(pulse_mask):
            data_in_window_to_clip = filtered_data[pulse_mask]
            t_clip = time_axis_us_trimmed[pulse_mask]
            base_clip = slope_ref * t_clip + intercept_ref
            avg_base_clip = np.mean(base_clip)
            decimal_max_pos_dev = MAX_POSITIVE_DEVIATION_PERCENT / 100.0
            constant_positive_offset = decimal_max_pos_dev * abs(avg_base_clip)
            clip_upper_parallel = base_clip + constant_positive_offset
            clipped_segment = np.minimum(data_in_window_to_clip, clip_upper_parallel)
            if filtered_data is trimmed_data or (hasattr(filtered_data, 'base') and filtered_data.base is trimmed_data): filtered_data = np.copy(filtered_data)
            filtered_data[pulse_mask] = clipped_segment
            print(f"Clipping (only exceeding +{MAX_POSITIVE_DEVIATION_PERCENT}%) multi-segment filtered data in 2-4 µs range.")
            clip_lower_parallel = base_clip - constant_positive_offset
            t_clip_plot = t_clip
            clip_upper_plot = clip_upper_parallel
            clip_lower_plot = clip_lower_parallel
    if filtered_data is not None and scipy_available: # Integration
        print("Integrating filtered_data...")
        try:
            if len(filtered_data) == len(time_axis_us_trimmed):
                integrated_data = cumulative_trapezoid(filtered_data, time_axis_us_trimmed, initial=0)
                print("Integration successful.")
            else: integrated_data = None; print(f"Data size mismatch ({len(filtered_data)}) vs ({len(time_axis_us_trimmed)})")
        except Exception as e: integrated_data = None; print(f"Error during integration: {e}")
    elif not scipy_available: print("Scipy (scipy.integrate) unavailable, cannot integrate data.")
    elif filtered_data is None: print("filtered_data not found for integration.")

# --- 7. Plot Graphs ---
fig, axes = plt.subplots(4, 1, figsize=(12, 22), sharex=True)

# Subplot 1: Raw Data
axes[0].plot(time_axis_us_trimmed, trimmed_data, label=r'$V_{rogowski\ coil}$', color='blue')
axes[0].set_ylabel(r'$V_{rogowski\ coil}\ (V)$');
axes[0].legend(loc='upper right')
if PLOT_XLIM_CONFIG is not None: axes[0].set_xlim(PLOT_XLIM_CONFIG)
if PLOT_YLIM_AX0_RAW is not None: axes[0].set_ylim(PLOT_YLIM_AX0_RAW)

# Subplot 2: Smoothed Data (for pulse detection), Trend Line, Parallel Thresholds, and Pulse
if smoothed_data is not None:
    axes[1].plot(time_axis_us_trimmed, smoothed_data, label='Smoothed Data (for pulse find)', color='green', alpha=0.6)
    axes[1].set_ylabel(r'$Smoothed\ data$'); 
    if trend_line_params is not None:
        slope, intercept = trend_line_params; x_trend_plot = np.array([2.0, 4.0]); y_trend_plot = slope * x_trend_plot + intercept
        axes[1].plot(x_trend_plot, y_trend_plot, color='orange', linestyle='-', linewidth=2, label='Base Line (Trend 2-4µs)')
        if thresh_H is not None and thresh_L is not None and t_analysis is not None:
            axes[1].plot(t_analysis, thresh_H, color='purple', linestyle='--', linewidth=1.5, label=f'Parallel Th -{PERCENT_THRESHOLD_INNER}% (H)')
            axes[1].plot(t_analysis, thresh_L, color='darkviolet', linestyle='--', linewidth=1.5, label=f'Parallel Th -{PERCENT_THRESHOLD_OUTER}% (L)')
        if pulse_start_time is not None and pulse_end_time is not None and pulse_duration is not None:
            axes[1].axvspan(pulse_start_time, pulse_end_time, color='yellow', alpha=0.4, label=f'Detected Pulse ({pulse_duration:.2f} {time_unit_display})')
else:
    axes[1].plot(time_axis_us_trimmed, trimmed_data, label='Original Data (No Analysis)', color='gray', linestyle=':'); axes[1].set_ylabel('Value (di/dt)'); 
axes[1].legend(loc='upper right', fontsize='small')
if PLOT_XLIM_CONFIG is not None: axes[1].set_xlim(PLOT_XLIM_CONFIG) 
if PLOT_YLIM_AX1_PULSE_DETECT is not None: axes[1].set_ylim(PLOT_YLIM_AX1_PULSE_DETECT)

# Subplot 3: Multi-Segment Filtered Data (and Clipped)
if filtered_data is not None:
    axes[2].plot(time_axis_us_trimmed, filtered_data, label='Multi-Segment Filtered Data', color='teal')
    if t_clip_plot is not None and clip_upper_plot is not None and clip_lower_plot is not None:
        axes[2].plot(t_clip_plot, clip_upper_plot, color='purple', linestyle='-', linewidth=1.5, label=f'Limit Band (±{MAX_POSITIVE_DEVIATION_PERCENT}%)')
        axes[2].plot(t_clip_plot, clip_lower_plot, color='purple', linestyle='-', linewidth=1.5)
    axes[2].set_ylabel(r'$Filtered\ &\ Upper\ Clipped\ (V)$')
else:
    axes[2].text(0.5, 0.5, 'Multi-Segment Filtered Data not available', ha='center', va='center', transform=axes[2].transAxes); 
axes[2].legend(loc='upper right', fontsize='small')
if PLOT_XLIM_CONFIG is not None: axes[2].set_xlim(PLOT_XLIM_CONFIG) 
if PLOT_YLIM_AX2_MULTI_FILTER is not None: axes[2].set_ylim(PLOT_YLIM_AX2_MULTI_FILTER)

# Subplot 4: Integrated Data
integrated_data = integrated_data*28846/1000 
if integrated_data is not None:
    axes[3].plot(time_axis_us_trimmed, integrated_data, label='Integrated Data', color='brown') 
    axes[3].set_ylabel(r'$Current\ (kA)$')
else:
    axes[3].text(0.5, 0.5, 'Integrated Data not available\n(Requires Scipy and valid prior processing)',
                 horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)
    axes[3].set_title('Integrated Data (Not Available)')

axes[3].legend(loc='upper right')
axes[3].set_xlabel(f'$Time ({time_unit_display})$')
if PLOT_XLIM_CONFIG is not None: axes[3].set_xlim(PLOT_XLIM_CONFIG) 
if PLOT_YLIM_AX3_INTEGRATED is not None: axes[3].set_ylim(PLOT_YLIM_AX3_INTEGRATED)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()