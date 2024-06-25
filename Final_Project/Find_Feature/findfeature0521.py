import numpy as np
import scipy.signal as sig

def denoise(data):
    order = 4
    lowcut = 100
    fs = 500
    # 設計低通濾波器
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)

    # 使用低通濾波器對訊號降噪
    sig_denoised = sig.filtfilt(b, a, data)

    return sig_denoised

def findR(data):
    try:
        p_data = np.zeros_like(data, dtype=np.int32)
        count = data.shape[0]
        arr_rowsum = []
        for k in range(1, count // 2 + 1):
            row_sum = 0
        
            for i in range(k, count - k):
                if data[i] > data[i - k] and data[i] > data[i + k]:
                    row_sum = row_sum - 1
            arr_rowsum.append(row_sum)
        
        min_index = np.argmin(arr_rowsum)
        max_window_length = min_index
        
        for kk in range(1, max_window_length + 1):
            for ii in range(kk, count - kk):
                if data[ii] > data[ii - kk] and data[ii] > data[ii + kk]:
                    p_data[ii] = p_data[ii] + 1
                    
        px = np.where(p_data == max_window_length)[0]
        return px
    except:
        px = 0
        return px

def filtR(px, data):
    try:
        pmax = max(data[px])
        pxlist = px.tolist()
        newpx = pxlist.copy()
        newpx.remove(pxlist[np.argmax(data[pxlist])])
        pmax = max(data[newpx])# 找次高項 避免極值(ex 369)
        
        for i in range(0, len(px)):
            if data[px[i]] < 0.7*pmax:
                pxlist.remove(px[i])
                
        Rx = pxlist.copy()
        for i in range(0, len(pxlist)-1):
            if pxlist[i+1]-pxlist[i] < 150:
                Rx.remove(pxlist[i+1])
            
        return Rx
    except:
        Rx = pxlist
        return Rx

def findPQST(data, Rx):
    try:
        Px, Tx , Qx, Sx = [], [], [], []
        L_x, P_x, S_x, T_x = [], [], [], []
        stop = 0
        for i in range(0, len(Rx)-1):
            #找P、T
            Rx1 = np.arange(Rx[i]+30, (Rx[i+1]+Rx[i])//2, 1)
            peak = np.argmax(data[Rx1])
            Tpeak = Rx1[0] + peak
            Tx.append(Tpeak)
            
            Rx2 = np.arange((Rx[i+1]+Rx[i])//2+30, Rx[i+1]-30, 1)
            peak = np.argmax(data[Rx2])
            Ppeak = Rx2[0] + peak
            Px.append(Ppeak)
            
            #找QS
            Srange = np.arange(Rx[i], Tx[i])
            Sx.append(np.argmin(data[Srange]) + Srange[0])
            
            Qrange = np.arange(Px[i], Rx[i+1])
            Qx.append(np.argmin(data[Qrange]) + Qrange[0])
            
            
            #找轉折點
            foundS_ = False
            for j in range(Tx[i], Sx[i], -1):
                if data[j]*data[j-1] < 0:
                    S_x.append(j)
                    foundS_ = True
                    break
                if not foundS_:
                    continue
                
            foundT_ = False
            for j in range(Tx[i], Tx[i]+200, 1):
                if data[j]*data[j+1] < 0:
                    T_x.append(j)
                    foundT_ = True
                    break
                if not foundT_:
                    continue
                
            foundL_ = False
            for j in range(Px[i], Px[i]-200, -1):
                if data[j]*data[j-1] < 0:
                    L_x.append(j)
                    foundL_ = True
                    break
                if not foundL_:
                    continue
                
            foundP_ = False
            for j in range(Px[i], Qx[i], 1):
                if data[j]*data[j+1] < 0:
                    P_x.append(j)
                    foundP_ = True
                    break
                if not foundP_:
                    continue
            
            if not foundS_:
                S_x.append((Tx[i] + Sx[i])//2)
            if not foundT_:
                T_x.append(Tx[i]+100)
            if not foundL_:
                L_x.append(Px[i]-200)
            if not foundP_:
                P_x.append((Px[i] + Qx[i])//2)
    
        return Px, Tx, Qx, Sx, S_x, T_x, P_x, L_x
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0
