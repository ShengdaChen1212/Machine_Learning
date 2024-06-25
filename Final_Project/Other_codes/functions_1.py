# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:20:00 2024

@author: ShengdaChen
"""

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import neurokit2 as nk


def denoise(data):
    fs     = 500 # 取樣率為500Hz
    lowcut = 100 # 低通濾波器截止頻率為100Hz
    order  = 4   # 濾波器階數
    # 設計低通濾波器
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
    # 使用低通濾波器對訊號降噪
    sig_denoised = sig.filtfilt(b, a, data)
    return sig_denoised

# ===============================
# plot_ecg : 
# plot the ecg signal, denoise inside the function
# INPUTS : 
# data   = original ecg signal
# client = client number
# lead   = 哪種導程
# ===============================

def plot_ecg(data, client, lead):
    sig_denoised = denoise(data)
    plt.figure(figsize = (20,8), dpi = 250)
    plt.plot(sig_denoised, 'b', label = 'Denoised ECG')
    plt.title(f"Client {client}'s Denoised ECG, Lead {lead}")

# 找同一個 client 的不同 leads
def get_leads_ecg(data, client, last_lead):
    for i in range(last_lead):
        ecg_data = data[client, i, :]
        plot_ecg(ecg_data, client, i)

# 找不同 Clients 的同一個 lead
def get_clients_ecg(data, last_client, lead):
    for i in range(last_client):
        ecg_data = data[i, lead, :]
        plot_ecg(ecg_data, i, lead)

def extract_ecg_features(ecg_lead, sampling_rate):
    """
    Processes an ECG lead, identifies, and returns the indices of P, Q, R, S, T points.
    
    Parameters:
    ecg_lead (numpy array): The ECG lead data.
    sampling_rate (int): The sampling rate of the ECG data.
    
    Returns:
    dict: A dictionary containing lists of indices for P, Q, R, S, T waves.
    """
    # Process the ECG signal
    processed_ecg = nk.ecg_process(ecg_lead, sampling_rate=sampling_rate)
    _, waves_peak = nk.ecg_delineate(processed_ecg[0], processed_ecg[1], sampling_rate=sampling_rate, method="peak")
    _, waves_contour = nk.ecg_delineate(processed_ecg[0], processed_ecg[1], sampling_rate=sampling_rate, method="dwt")
    # Extract peaks using a safe method for potentially empty arrays
    p_peaks = np.atleast_1d(waves_peak.get('ECG_P_Peaks', np.array([])))
    q_peaks = np.atleast_1d(waves_peak.get('ECG_Q_Peaks', np.array([])))
    r_peaks = np.atleast_1d(waves_contour.get('ECG_R_Peaks', np.array([])))  # Using R offsets if R peaks are not directly available
    s_peaks = np.atleast_1d(waves_peak.get('ECG_S_Peaks', np.array([])))
    t_peaks = np.atleast_1d(waves_peak.get('ECG_T_Peaks', np.array([])))
    # Storing the indices
    p_indices = np.where(p_peaks == 1)[0]
    q_indices = np.where(q_peaks == 1)[0]
    r_indices = np.where(r_peaks == 1)[0]  # Check if this mapping is adequate for your needs
    s_indices = np.where(s_peaks == 1)[0]
    t_indices = np.where(t_peaks == 1)[0]
    # Plot the ECG with annotations
    '''
    plt.figure(figsize=(15, 5))
    nk.ecg_plot(processed_ecg[0])
    '''
    # Highlight the waves
    '''
    for wave_indices, color, label in zip([p_indices, q_indices, r_indices, s_indices, t_indices], 
                                          ['green', 'red', 'blue', 'purple', 'orange'], 
                                          ['P', 'Q', 'R', 'S', 'T']):
        if wave_indices.size > 0:
            plt.scatter(wave_indices, ecg_lead[wave_indices], color=color, label=f"{label} peaks", zorder=3)
    plt.legend()
    plt.title('ECG Signal with PQRST Points')
    plt.show()
    '''
    return {
        "P": processed_ecg[1]['ECG_P_Peaks'],
        "Q": processed_ecg[1]['ECG_Q_Peaks'],
        "R": processed_ecg[1]['ECG_R_Peaks'],
        "S": processed_ecg[1]['ECG_S_Peaks'],
        "T": processed_ecg[1]['ECG_T_Peaks']
    }

# Example usage
# You need to provide the 'ecg_lead' as a numpy array and 'sampling_rate' as an integer
# ecg_features = extract_ecg_features(your_ecg_lead_data, your_sampling_rate)

def remove_nans(peaks):
    """
    Removes NaNs from the list of peak indices.
    
    Parameters:
    peaks (list): List of peak indices.
    
    Returns:
    list: List of peak indices with NaNs removed.
    """
    return [int(peak) for peak in peaks if not np.isnan(peak)]