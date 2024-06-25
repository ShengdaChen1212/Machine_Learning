# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:26:30 2023

@author: Administrator
"""

import numpy as np
import scipy.signal as sig
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

#%% Load the data
data = np.load('./ML_Train.npy')
ecg_data = data[0,0,:]

#%% Spectrum Analysis
fs = 500 # ECG訊號取樣率為500Hz

# 計算ECG訊號的功率譜密度（PSD）
f, psd = sig.welch(ecg_data, fs=fs)


def derivative(signal):
    '''
    Derivative Filter 
    :param signal: input signal
    :return: prcoessed signal
    
    Methodology/Explaination:
    The derivative of the input signal is taken to obtain the
    information of the slope of the signal. Thus, the rate of change
    of input is obtain in this step of the algorithm.
    
    The derivative filter has the recursive equation:
      y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)
    '''
    
    # Initialize result
    result = signal.copy()
    
    # Apply the derivative filter using the equation given
    for index in range(len(signal)):
        result[index] = 0
    
    if (index >= 1):
        result[index] -= 2*signal[index-1]
    
    if (index >= 2):
        result[index] -= signal[index-2]
    
    if (index >= 2 and index <= len(signal)-2):
        result[index] += 2*signal[index+1]
    
    if (index >= 2 and index <= len(signal)-3):
       result[index] += signal[index+2]
    
    result[index] = (result[index]*annotation.fs)/8
    
    return result
#%% FFT
fs = 500
T = 1/fs
L = ecg_data.shape[-1]

Y = fft(ecg_data)
freq = fftfreq(ecg_data.shape[-1])
P2 = abs(Y/L)
P1 = P2[0:L//2]
P1[1:] = 2*P1[1:]
freq = freq[:L//2]*fs

#%%
plt.figure(figsize = (10,10), dpi = 250)
plt.subplot(211)
plt.plot(f, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2 / Hz)')
plt.title('Estimate Power Spectral Density (PSD) Using Welch’s Method',fontsize=16, fontweight='bold')
plt.grid()

plt.subplot(212)
plt.plot(freq, P1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('|P1(f)|')
plt.title('Single-Sided Amplitude Spectrum of ECG-Signal Using FFT',fontsize=16, fontweight='bold')
plt.grid()
plt.show()
