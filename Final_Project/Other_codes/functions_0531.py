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

file = np.load("ML_Train.npy", mmap_mode='r')

def denoise(data, order=4, lowcut=80, fs=1000):
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
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
        


get_leads_ecg(file, 0, 1)
#get_leads_ecg(file, 12208, 4)



















