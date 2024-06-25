#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 02:22:45 2023

@author: wendyyu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as sig

measures = {}

def data_process(person,lead,ecg_data):
    # Denoise ECG signal
    fs = 500 # 取樣率為500Hz
    lowcut = 100 # 低通濾波器截止頻率為100Hz，會將高於100Hz的部分濾掉
    order = 4 # 濾波器階數
    
    # 設計低通濾波器
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
    # 使用低通濾波器對訊號降噪
    sig_denoised = sig.filtfilt(b, a, ecg_data)
    return sig_denoised

def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def rolmean(dataset, hrw, fs):
    mov_avg = dataset['hart'].rolling(int(hrw*fs)).mean()#計算移動平均
    avg_hr = (np.mean(dataset.hart))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    #math.isnan(x) 輸入的x是 NaN 時，回傳 True，否則回傳 False
    #這裡：是NaN，則用平均值
    mov_avg = [x*1.2 for x in mov_avg]
    #將平均值提高 20% 以防止繼發性心臟收縮受到干擾
    dataset['hart_rollingmean'] = mov_avg

#標記ROI並找出最高點
def detect_peaks(dataset):
    window = [] #記錄每一輪ROI範圍
    peaklist = [] #紀錄最高點位置
    listpos = 0 #用一個計數器來移動不同的數據列
    for datapoint in dataset.hart:
        rollingmean = dataset.hart_rollingmean[listpos]
        if (datapoint < rollingmean) and (len(window) < 1):
		#未檢測到R-complex activity(因為len(window) < 1，所以目前沒有要檢查的ROI)
            listpos += 1
        elif (datapoint > rollingmean):#信號在平均之上，標記為ROI
            window.append(datapoint)
            listpos += 1
        else:#當信號將要掉到平均之下且等於平均的那一刻，回頭去找ROI範圍中最高的一點
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(maximum))
	    #標記peak的位置
            peaklist.append(beatposition)
            window = []#重置window
            listpos += 1
    
    peak_x, peak_y = filter_r_peak(peaklist,dataset)
            
    measures['peaklist'] = peak_x
    measures['ybeat'] = peak_y
    return peak_x
    #peaklist只是最高點在x軸上的位置，還要找出最高點在y軸上的值

def filter_r_peak(peaklist, dataset): # 4/30 簡化程式、加上x軸距離的判斷
    ybeat = [dataset.hart[x] for x in peaklist]
    peak_large = sorted(ybeat, reverse=True)[0:30][2:5]
    peak_mean = np.mean(peak_large)
    peak_criteria = peak_mean*0.4
    peak_x, peak_y =[],[]
    for i,y in enumerate(ybeat):
        if abs(peak_mean - y) < peak_criteria:
            peak_x.append(peaklist[i])
            peak_y.append(y)
            
    diff_x = [peak_x[i+1]-peak_x[i] for i in range(len(peak_x)-1)] #取最大的5個取平均
    avg_x = np.mean(sorted(diff_x, reverse=True)[:5])

    for i,x in enumerate(peak_x):
        if i < (len(peak_x)-1) and abs(peak_x[i+1]-peak_x[i]) < (avg_x)*0.7:
            if peak_y[i+1] > peak_y[i]:
                peak_x.pop(i)
                peak_y.pop(i)
            else:
                peak_x.pop(i+1)
                peak_y.pop(i+1)

    return (peak_x, peak_y)

def process(dataset, hrw, fs, person, lead):
    rolmean(dataset, hrw, fs)
    detect_peaks(dataset)

