# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:20:00 2024

@author: ShengdaChen
"""

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt


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