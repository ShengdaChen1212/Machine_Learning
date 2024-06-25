import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from functions import denoise

def find_features_v2(ecg_signal, sampling_rate=800):

    # Preprocess the ECG signal
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

    # Find R-peaks
    r_peaks, info_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

    # Extract R-peak indices from the DataFrame and convert to a dictionary
    r_peak_indices = r_peaks['ECG_R_Peaks'][r_peaks['ECG_R_Peaks'] == 1].index.to_list()
    r_peaks_dict = {'ECG_R_Peaks': r_peak_indices}

    # Segment the ECG signal into PQRST complexes
    signals, info = nk.ecg_delineate(ecg_cleaned, r_peaks_dict, sampling_rate=sampling_rate, method='dwt')

    # Function to safely get values from delineation info
    def safe_get(info, key, idx, default):
        try:
            value = info[key][idx]
            if np.isnan(value):
                return default
            return int(value)
        except (IndexError, KeyError):
            return default

    # Initialize lists for storing PQRST peaks
    p_peaks = []
    q_peaks = []
    r_peaks = []
    s_peaks = []
    t_peaks = []

    # Initialize dictionary for storing L', S', T', and P'
    extra_peaks = {
        "L'": [],
        "S'": [],
        "T'": [],
        "P'": []
    }

    features = {
        'P': p_peaks,
        'Q': q_peaks,
        'R': r_peaks,
        'S': s_peaks,
        'T': t_peaks,
        "L'": extra_peaks["L'"],
        "S'": extra_peaks["S'"],
        "T'": extra_peaks["T'"],
        "P'": extra_peaks["P'"]
    }

    annotations = []

    for i, r_peak in enumerate(r_peak_indices):
        # Skip the last segment if it doesn't have a valid next P peak
        if i + 1 >= len(r_peak_indices):
            break

        prev_t_peak = safe_get(info, 'ECG_T_Peaks', i - 1, 0) if i > 0 else 0

        # Restrict the search for P_peak to the region immediately before Q_peak
        search_window_before_q = 50  # Define a search window before Q peak
        q_peak = safe_get(info, 'ECG_Q_Peaks', i, r_peak)
        p_wave_segment = ecg_cleaned[max(q_peak - search_window_before_q, 0):q_peak]
        if len(p_wave_segment) > 0:
            p_peak_rel = np.argmax(p_wave_segment)
            p_peak = max(q_peak - search_window_before_q, 0) + p_peak_rel
        else:
            p_peak = prev_t_peak

        r_wave = ecg_cleaned[r_peak]

        # Ensure S_peak is the lowest point within 150 samples after R peak
        s_search_start = r_peak
        s_search_end = r_peak + 150 if r_peak + 150 < len(ecg_cleaned) else len(ecg_cleaned) - 1
        s_wave_segment = ecg_cleaned[s_search_start:s_search_end]
        if len(s_wave_segment) > 0:
            s_peak_rel = np.argmin(s_wave_segment)
            s_peak = s_search_start + s_peak_rel
        else:
            s_peak = s_search_end

        # Ensure T_peak is the highest peak within 200 samples after S peak
        t_search_start = s_peak
        t_search_end = s_peak + 200 if s_peak + 200 < len(ecg_cleaned) else len(ecg_cleaned) - 1
        t_wave_segment = ecg_cleaned[t_search_start:t_search_end]
        if len(t_wave_segment) > 0:
            t_peak_rel = np.argmax(t_wave_segment)
            t_peak = t_search_start + t_peak_rel
        else:
            t_peak = t_search_end

        # If T peak is too close to R peak, set it to NaN
        if t_peak - r_peak <= 10:
            t_peak = np.nan

        p_peaks.append(p_peak)
        q_peaks.append(q_peak)
        r_peaks.append(r_peak)
        s_peaks.append(s_peak)
        t_peaks.append(t_peak)

        annotations.append((p_peak, ecg_cleaned[p_peak], 'P'))
        annotations.append((q_peak, ecg_cleaned[q_peak], 'Q'))
        annotations.append((r_peak, r_wave, 'R'))
        annotations.append((s_peak, ecg_cleaned[s_peak], 'S'))
        if not np.isnan(t_peak):
            annotations.append((t_peak, ecg_cleaned[t_peak], 'T'))

        # Calculate L', S', T', P'
        for name, point_range, comparison in [
            ("S'", np.arange(t_peak, s_peak, -1), lambda j: ecg_cleaned[j] * ecg_cleaned[j - 1] < 0),
            ("T'", np.arange(t_peak, t_peak + 200), lambda j: ecg_cleaned[j] * ecg_cleaned[j + 1] < 0),
            ("L'", np.arange(p_peak, p_peak - 200, -1), lambda j: ecg_cleaned[j] * ecg_cleaned[j - 1] < 0),
            ("P'", np.arange(p_peak, q_peak), lambda j: ecg_cleaned[j] * ecg_cleaned[j + 1] < 0)
        ]:
            found = False
            min_slope_index = -1
            min_slope = float('inf')
            for j in point_range:
                if comparison(j):
                    extra_peaks[name].append(j)
                    found = True
                    break
                # Calculate slope
                slope = abs(ecg_cleaned[j] - ecg_cleaned[j - 1])
                if slope < min_slope:
                    min_slope = slope
                    min_slope_index = j
            if not found and min_slope_index != -1:
                extra_peaks[name].append(min_slope_index)
            if not found and min_slope_index == -1:
                default_point = (t_peak + s_peak) // 2 if name == "S'" else \
                                t_peak + 100 if name == "T'" else \
                                p_peak - 200 if name == "L'" else \
                                (p_peak + q_peak) // 2
                extra_peaks[name].append(default_point)

    # Plot the ECG signal with delineated PQRST complexes and extra peaks

    #plt.figure(figsize=(15, 8))

    # Plot the cleaned ECG signal
    #plt.plot(ecg_cleaned, label='ECG Signal', color='black')

    # Add annotations for PQRST peaks
    #for (x, y, label) in annotations:
    #    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    # Add annotations for L', S', T', P'
    #for name, points in extra_peaks.items():
    #    for point in points:
    #        plt.annotate(name, (point, ecg_cleaned[point]), textcoords="offset points", xytext=(0, -15), ha='center', color='blue')

    # Add labels and legend
    '''
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Delineated PQRST Complexes and Extra Peaks')
    plt.legend()
    plt.show()
    '''
    return features

# Example usage
'''
data = np.load('ML_Train.npy', mmap_mode='r')
ecg_signal = data[10110, 0, :]
ecg_signal = denoise(ecg_signal)
features = find_features_v2(ecg_signal)
print(features)
'''
