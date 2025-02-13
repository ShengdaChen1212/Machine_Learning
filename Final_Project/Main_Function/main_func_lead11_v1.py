import numpy as np
import pandas as pd
#import find_feature_v0 as ff
import find_feature_v1 as ff

#%%

Fiducial_21 = ["dQS_x", "dRS_x", "dRT_x", "dRS'_x", "dRT'_x", "dL'P'_x", "dS'T'_x",
               "dST_x", "dST'_x", "dRQ_y", "dRS_y", "dTS_y", "dTT'_y"]

#%% get features 

def get_21(origindata, lead, Fiducial_21):
    feature_list = []
    stop_cnt  = 0
    feature_num = len(Fiducial_21)
    
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
                feature_list.append([0] * feature_num)
                print(person, ' x')
                stop_cnt += 1
                continue
            
            #找距離
            features = {key: [] for key in Fiducial_21}
            
            for i in range(0, len(Rx)-2):
                # 找x軸 (時間)
                features["dQS_x"].append(abs(Sx[i] - Qx[i]))
                features["dRS_x"].append(abs(Rx[i+1] - Sx[i+1]))
                features["dRT_x"].append(abs(Rx[i+1] - Tx[i+1]))
                features["dRS'_x"].append(abs(Rx[i+1] - S_x[i+1]))
                features["dRT'_x"].append(abs(Rx[i+1] - T_x[i+1]))
                features["dL'P'_x"].append(abs(L_x[i] - P_x[i]))
                features["dS'T'_x"].append(abs(S_x[i+1] - T_x[i+1]))
                features["dST_x"].append(abs(Sx[i+1] - Tx[i+1]))
                features["dST'_x"].append(abs(Sx[i+1] - T_x[i+1]))
                
                # 找y軸 (電位)
                features["dRQ_y"].append(abs(data[Qx[i]] - data[Rx[i+1]]))
                features["dRS_y"].append(abs(data[Rx[i+1]] - data[Sx[i+1]]))
                features["dTS_y"].append(abs(data[Sx[i+1]] - data[Tx[i+1]]))
                features["dTT'_y"].append(abs(data[T_x[i+1]] - data[Tx[i+1]]))
                
            list_21 = []
            for index in Fiducial_21:
                list_21.append(np.mean(features[index]))
            feature_list.append(list_21)
            print(person)
        except:
            feature_list.append([0] * feature_num)
            print(person, ' x')
            stop_cnt += 1
    print(f'Stop count: {stop_cnt}')
    return feature_list
#%% feature generation
def feature_gen(Fiducial_21, lead, option=True):
    if option:
        print(f'Generating Lead {lead} Training Features: ')
        print('='*30)
        dataset = np.load('ML_Train.npy', mmap_mode='r')
    else:
        print(f'Generating Lead {lead} Testing Features: ')
        print('='*30)
        dataset = np.load('ML_Test.npy',  mmap_mode='r')
        
    feature_list = get_21(dataset, lead, Fiducial_21)
    feature = pd.DataFrame(feature_list, columns=Fiducial_21)
    print(feature.isna().sum(axis=0))
    
    if option:
        print('='*30)
        print(f'Generating Lead {lead} Training Features DONE!')
        feature.to_csv(f'training_lead{lead}_features.csv', index=False)
    else:
        print('='*30)
        print(f'Generating Lead {lead} Testing Features DONE!')
        feature.to_csv(f'testing_lead{lead}_features.csv', index=False)
    return feature

#%%

lead = 11
feature = feature_gen(Fiducial_21, lead)
# option=False
print(f'Total feature: {len(Fiducial_21)}')
# for f in Fiducial_21:
#     ff.plot_feature(feature, f)
    