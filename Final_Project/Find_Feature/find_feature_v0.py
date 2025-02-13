import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
#%%
def plot_feature(df, feature):
    plt.figure(figsize = (20, 20), dpi=200)
    plt.scatter(list(range(0, 12209)), df.loc[:, feature])
    plt.xlabel('Data', fontsize = 40)
    plt.ylabel(feature, fontsize = 40)
    plt.title(f'{feature} vs. Data', fontsize = 40)
    #plt.legend(fontsize = 18)
    plt.show()

def denoise(data, order=4, lowcut=25, fs=500):
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
    sig_denoised = sig.filtfilt(b, a, data)
    return sig_denoised

def find_and_filter_r_peaks(data, distance=150, height=None, threshold=0.5):
    peaks, _ = sig.find_peaks(data, distance=distance, height=height)
    if len(peaks) == 0:
        return peaks

    pmax = max(data[peaks])
    peaks = [peak for peak in peaks if data[peak] >= threshold * pmax]
    filtered_peaks = []
    for i in range(len(peaks) - 1):
        if peaks[i + 1] - peaks[i] >= distance:
            filtered_peaks.append(peaks[i])
    filtered_peaks.append(peaks[-1])

    return np.array(filtered_peaks)

def find_pqrst_peaks(data, r_peaks):
    features = {
        'P': [], 'Q': [], 'R': r_peaks, 'S': [], 'T': [],
        "S'": [], "T'": [], "P'": [], "L'": []
    }

    for i in range(len(r_peaks) - 1):
        # 找T峰
        t_range = np.arange(r_peaks[i] + 30, (r_peaks[i] + r_peaks[i + 1]) // 2)
        t_peak = t_range[0] + np.argmax(data[t_range])
        features['T'].append(t_peak)
        
        # 找P峰
        p_range = np.arange((r_peaks[i] + r_peaks[i + 1]) // 2 + 30, r_peaks[i + 1] - 30)
        p_peak = p_range[0] + np.argmax(data[p_range])
        features['P'].append(p_peak)
        
        # 找S峰
        s_range = np.arange(r_peaks[i], t_peak)
        s_peak = s_range[0] + np.argmin(data[s_range])
        features['S'].append(s_peak)
        
        # 找Q峰
        q_range = np.arange(p_peak, r_peaks[i + 1])
        q_peak = q_range[0] + np.argmin(data[q_range])
        features['Q'].append(q_peak)
        
        # 找转折点和斜率突变点
        for name, point_range, comparison in [
            ("S'", np.arange(t_peak, s_peak, -1), lambda j: data[j] * data[j - 1] < 0),
            ("T'", np.arange(t_peak, t_peak + 200), lambda j: data[j] * data[j + 1] < 0),
            ("L'", np.arange(p_peak, p_peak - 200, -1), lambda j: data[j] * data[j - 1] < 0),
            ("P'", np.arange(p_peak, q_peak), lambda j: data[j] * data[j + 1] < 0)
        ]:
            found = False
            min_slope_index = -1
            min_slope = float('inf')
            for j in point_range:
                if comparison(j):
                    features[name].append(j)
                    found = True
                    break
                # 计算斜率
                slope = abs(data[j] - data[j - 1])
                if slope < min_slope:
                    min_slope = slope
                    min_slope_index = j
            if not found and min_slope_index != -1:
                features[name].append(min_slope_index)
            if not found and min_slope_index == -1:
                default_point = (t_peak + s_peak) // 2 if name == "S'" else \
                                t_peak + 100 if name == "T'" else \
                                p_peak - 200 if name == "L'" else \
                                (p_peak + q_peak) // 2
                features[name].append(default_point)
        
        # 找S'点左侧和T'点右侧的斜率突变点
        features["S'_left"] = []
        features["T'_right"] = []
        features["L'_left"] = []
        features["P'_right"] = []
        if "S'" in features:
            for i in range(len(features["S'"])):
                s_prime = features["S'"][i]
                s_prime_slope = [(abs(data[j] - data[j - 1]), j) for j in range(s_prime - 12, s_prime + 15)]
                min_slope_index_s_prime = min(s_prime_slope, key=lambda x: x[0])[1]
                features["S'_left"].append(min_slope_index_s_prime)
        
        if "T'" in features:
            for i in range(len(features["T'"])):
                t_prime = features["T'"][i]
                t_prime_slope = [(abs(data[j + 1] - data[j]), j) for j in range(t_prime - 10, t_prime + 7)]
                min_slope_index_t_prime = min(t_prime_slope, key=lambda x: x[0])[1]
                features["T'_right"].append(min_slope_index_t_prime)

        if "L'" in features:
            for i in range(len(features["L'"])):
                l_prime = features["L'"][i]
                l_prime_slope = [(abs(data[j] - data[j - 1]), j) for j in range(l_prime - 3, l_prime)]
                min_slope_index_l_prime = min(l_prime_slope, key=lambda x: x[0])[1]
                features["L'_left"].append(min_slope_index_l_prime)
        
        if "P'" in features:
            for i in range(len(features["P'"])):
                p_prime = features["P'"][i]
                p_prime_slope = [(abs(data[j + 1] - data[j]), j) for j in range(p_prime, p_prime + 3)]
                min_slope_index_p_prime = min(p_prime_slope, key=lambda x: x[0])[1]
                features["P'_right"].append(min_slope_index_p_prime)

        # 修正S' T'之點位置
        features["S'"] = features["S'_left"]
        features["T'"] = features["T'_right"]
        features["L'"] = features["L'_left"]
        features["P'"] = features["P'_right"]
        
        # 移除重複 key
        del features["S'_left"]
        del features["T'_right"]
        del features["L'_left"]
        del features["P'_right"]
    return features
#%%
def plot_9_pt(person, lead):
    data = np.load('ML_Train.npy', mmap_mode='r')
    ecg_data = data[person, lead, :]
    sig_denoised = denoise(ecg_data)
    
    r_peaks = find_and_filter_r_peaks(sig_denoised, distance=150)
    features = find_pqrst_peaks(sig_denoised, r_peaks)
    
    plt.figure(figsize=(15, 7))
    plt.plot(sig_denoised, linewidth=0.5, label="Denoised ECG")
    plt.scatter(features['P'],  sig_denoised[features['P']], color='green', label='P peaks')
    plt.scatter(features['Q'],  sig_denoised[features['Q']], color='red', label='Q peaks')
    plt.scatter(features['R'],  sig_denoised[features['R']], color='blue', label='R peaks')
    plt.scatter(features['S'],  sig_denoised[features['S']], color='purple', label='S peaks')
    plt.scatter(features['T'],  sig_denoised[features['T']], color='orange', label='T peaks')
    plt.scatter(features["S'"], sig_denoised[features["S'"]], color='pink', label="S' points", marker='o')
    plt.scatter(features["T'"], sig_denoised[features["T'"]], color='cyan', label="T' points", marker='o')
    plt.scatter(features["P'"], sig_denoised[features["P'"]], color='brown', label="P' points", marker='o')
    plt.scatter(features["L'"], sig_denoised[features["L'"]], color='yellow', label="L' points", marker='o')
    
    # should be comment
    if "S'_left" in features:
        plt.scatter(features["S'_left"], sig_denoised[features["S'_left"]], color='black', label="S' left slope point", marker='x')
    if "T'_right" in features:
        plt.scatter(features["T'_right"], sig_denoised[features["T'_right"]], color='black', label="T' right slope point", marker='x')
    
    plt.legend()
    plt.title('Denoised ECG with PQRST Points and Slope Change Points')
    plt.show()
