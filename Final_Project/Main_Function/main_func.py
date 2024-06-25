import numpy as np
import pandas as pd
import find_feature as ff

#%%
trainset = np.load('ML_Train.npy', mmap_mode='r')
#testset  = np.load('ML_Test.npy',  mmap_mode='r')

Fiducial_21 = ["dRP_x", "dRQ_x", "dRS_x", "dRT_x", "dRL'_x",
               "dRP'_x", "dRS'_x", "dRT'_x", "dL'P'_x", "dS'T'_x",
               "dST_x", "dPQ_x", "dPT_x", "dL'Q_x", "dST'_x",
               "dPL'_y", "dPQ_y", "dRQ_y", "dRS_y", "dTS_y", "dTT'_y"]

#%%

def get_21(origindata, lead, Fiducial_21):
    feature_list = []
    stop_cnt  = 0
# 
    for person in range(0, len(origindata)):
        data = origindata[person, lead, :]
        data = ff.denoise(data)
        stop = 0
        try:
            
            Rx  = ff.find_and_filter_r_peaks(data, distance=150)
            points = ff.find_pqrst_peaks(data, Rx)
            
            Px  = points['P']
            Qx  = points['Q']
            Sx  = points['S']
            Tx  = points['T']
            L_x = points["L'"]
            P_x = points["P'"]
            S_x = points["S'"]
            T_x = points["T'"]
            
            if len(Rx)>30 or len(Rx)<=5:
                stop = 1
            
            for p in [Rx, Px, Tx, Qx, Sx]:
                ystd = np.std(data[p])
                if ystd>0.15:
                    stop = 1
                    
            if stop == 1:
                feature_list.append([0] * 21)
                print(person, ' x')
                stop_cnt += 1
                continue
            
            #找距離
            features = {key: [] for key in Fiducial_21}
            
            for i in range(0, len(Rx)-2):
                # 找x軸 (時間)
                features["dRP_x"].append(abs(data[Rx[i+1]] - data[Px[i]]))
                features["dRQ_x"].append(abs(data[Rx[i+1]] - data[Qx[i]]))
                features["dRS_x"].append(abs(data[Rx[i+1]] - data[Sx[i+1]]))
                features["dRT_x"].append(abs(data[Rx[i+1]] - data[Tx[i+1]]))
                
                features["dRL'_x"].append(abs(data[Rx[i+1]] - data[L_x[i]]))
                features["dRP'_x"].append(abs(data[Rx[i+1]] - data[P_x[i]]))
                features["dRS'_x"].append(abs(data[Rx[i+1]] - data[S_x[i+1]]))
                features["dRT'_x"].append(abs(data[Rx[i+1]] - data[T_x[i+1]]))
                
                features["dL'P'_x"].append(abs(data[L_x[i]] - data[P_x[i]]))
                features["dS'T'_x"].append(abs(data[S_x[i+1]] - data[T_x[i+1]]))
                features["dST_x"].append(abs(data[Sx[i+1]] - data[Tx[i+1]]))
                features["dPQ_x"].append(abs(data[Px[i]] - data[Qx[i]]))
                
                features["dPT_x"].append(abs(data[Px[i]] - data[Tx[i+1]]))
                features["dL'Q_x"].append(abs(data[L_x[i]] - data[Qx[i]]))
                features["dST'_x"].append(abs(data[Sx[i+1]] - data[T_x[i+1]]))
                
                # 找y軸 (電位)
                features["dPL'_y"].append(abs(Px[i] - L_x[i]))
                features["dPQ_y"].append(abs(Px[i] - Qx[i]))
                features["dRQ_y"].append(abs(Qx[i] - Rx[i+1]))
                features["dRS_y"].append(abs(Rx[i+1] - Sx[i+1]))
                features["dTS_y"].append(abs(Sx[i+1] - Tx[i+1]))
                features["dTT'_y"].append(abs(T_x[i+1] - Tx[i+1]))
                
            #print(features)
            list_21 = []
            for index in Fiducial_21:
                list_21.append(np.mean(features[index]))
            feature_list.append(list_21)
            print(person)
        except:
            feature_list.append([0] * 21)
            print(person, ' x')
            stop_cnt += 1
    print(f'Stop count: {stop_cnt}')
    return feature_list
#%%
lead = 0
feature_list = get_21(trainset, lead, Fiducial_21)
#feature_list = get_21(testset, lead, Fiducial_21)
feature = pd.DataFrame(feature_list, columns=Fiducial_21)
print(feature.isna().sum(axis=0))
feature.to_csv(f'training_lead{lead}_features.csv', index=False)


'''
feature_list = get_21(testset, lead, Fiducial_21)
feature = pd.DataFrame(feature_list, columns=Fiducial_21)
print(feature.isna().sum(axis=0))
feature.to_csv(f'testing_lead{lead}_features.csv', index=False)
'''
