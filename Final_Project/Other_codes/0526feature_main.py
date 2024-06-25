import numpy as np
import pandas as pd
import heartbeat as hb
import findfeature0521 as ff
#%%
origindata = np.load('ML_Train.npy', mmap_mode='r')
# origindata = np.load('./ML_Test.npy')
#%%
feature_list = []
#%%
for personn in range(0, 5):
    person = personn
    lead = 1
    data = origindata[person, lead, :]
    data = ff.denoise(data)
    stop = 0
    try:
        px = ff.findR(data)
        Rx = ff.filtR(px, data)
        
        if len(Rx)>30 or len(Rx)<=5:
            dataset = pd.DataFrame(data, columns=['hart'])
            hb.process(dataset, 0.03, 500, person, lead)
            Rx, Ry = hb.detect_peaks(dataset)
        
        if len(Rx)>30 or len(Rx)<=5:
            stop = 1
            
        Px, Tx, Qx, Sx, S_x, T_x, P_x, L_x = ff.findPQST(data, Rx)
        
        
        for p in [Rx, Px, Tx, Qx, Sx]:
            ystd = np.std(data[p])
            if ystd>0.15:
                stop = 1
                
        if stop == 1:
            feature_list.append([0, 0, 0, 0, 0, 0, 0])
            print(person, ' x')
            continue
        
        #找距離
        dRQx , dRS_x, dS_T_x = [], [] ,[]
        dRQy, dRSy, dTSy, dTT_y = [], [], [], []
    
        for i in range(0, len(Rx)-2):
            dRQx.append(abs(Rx[i+1] - Qx[i]))
            dRS_x.append(abs(Rx[i+1] - S_x[i+1]))
            dS_T_x.append(abs(S_x[i+1] - T_x[i+1]))
            
            dRQy.append(abs(data[Qx[i]] - data[Rx[i+1]]))
            dRSy.append(abs(data[Rx[i+1]] - data[Sx[i+1]]))
            dTSy.append(abs(data[Sx[i+1]] - data[Tx[i+1]]))
            dTT_y.append(abs(data[T_x[i+1]] - data[Tx[i+1]]))
        print(len(dRQx), len(dRS_x))
        dRQx = np.mean(dRQx)
        dRS_x = np.mean(dRS_x)
        dS_T_x = np.mean(dS_T_x)
        
        dRQy = np.mean(dRQy)
        dRSy = np.mean(dRSy)
        dTSy = np.mean(dTSy)
        dTT_y = np.mean(dTT_y)
            
        feature_list.append([dRQx, dRS_x, dS_T_x, dRQy, dRSy, dTSy, dTT_y])
        print(person)
    except:
        feature_list.append([0, 0, 0, 0, 0, 0, 0])
        # xxx = feature_list[-1]
        # xxx.pop(0)
        # xxx.insert(0, person)
        # feature_list.append(xxx)
        print(person, ' x')
    
    
#%%

feature = pd.DataFrame(feature_list, columns=['dRQx', 'dRS_x', 'dS_T_x', 'dRQy', 'dRSy', 'dTSy', 'dTT_y'])
feature.to_csv(f'feature_train{lead}.csv', index=False)
