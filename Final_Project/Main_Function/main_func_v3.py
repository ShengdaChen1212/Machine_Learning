import numpy as np
import pandas as pd
#import find_feature_v0 as ff
import find_feature_v1 as ff
# import multiprocessing
# from joblib import Parallel, delayed

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
                features["dRP_y"].append(abs(data[Rx[i+1]] - data[Px[i]]))
                features["dRQ_y"].append(abs(data[Rx[i+1]] - data[Qx[i]]))
                features["dRS_y"].append(abs(data[Rx[i+1]] - data[Sx[i+1]]))
                features["dRT_y"].append(abs(data[Rx[i+1]] - data[Tx[i+1]]))
                
                features["dRL'_y"].append(abs(data[Rx[i+1]] - data[L_x[i]]))
                features["dRP'_y"].append(abs(data[Rx[i+1]] - data[P_x[i]]))
                features["dRS'_y"].append(abs(data[Rx[i+1]] - data[S_x[i+1]]))
                features["dRT'_y"].append(abs(data[Rx[i+1]] - data[T_x[i+1]]))
                
                features["dL'P'_y"].append(abs(data[L_x[i]] - data[P_x[i]]))
                features["dS'T'_y"].append(abs(data[S_x[i+1]] - data[T_x[i+1]]))
                features["dST_y"].append(abs(data[Sx[i+1]] - data[Tx[i+1]]))
                features["dPQ_y"].append(abs(data[Px[i]] - data[Qx[i]]))
                
                features["dPT_y"].append(abs(data[Px[i]] - data[Tx[i+1]]))
                features["dL'Q_y"].append(abs(data[L_x[i]] - data[Qx[i]]))
                features["dST'_y"].append(abs(data[Sx[i+1]] - data[T_x[i+1]]))
                
                features["dPL'_x"].append(abs(Px[i] - L_x[i]))
                features["dPQ_x"].append(abs(Px[i] - Qx[i]))
                features["dRQ_x"].append(abs(Qx[i] - Rx[i+1]))
                features["dRS_x"].append(abs(Rx[i+1] - Sx[i+1]))
                features["dTS_x"].append(abs(Sx[i+1] - Tx[i+1]))
                features["dTT'_x"].append(abs(T_x[i+1] - Tx[i+1]))
                
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
    
    
    
    # replace 0 to mean 
    feature = pd.DataFrame(feature_list, columns=Fiducial_21)
    print(feature.isna().sum(axis=0))
    Sums = feature.sum()
    Means = Sums / (len(feature)-stop_cnt)
    
    for data in range(len(origindata)):
        if 0 in feature.iloc[data, :].values:
            feature.iloc[data,:] = Means
    
    return feature

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
        
    feature = get_21(dataset, lead, Fiducial_21)
    return feature

# write features to csv
def to_csv(feature, option=True):
    if option:
        print('='*30)
        feature.to_csv('training_features_1.csv', index=False)
        print('Generating Training CSV DONE!')
    else:
        print('='*30)
        feature.to_csv('testing_features_1.csv', index=False)
        print('Generating Testing CSV DONE!')
    return None
#%%

# Fiducial_21 = ["dRP_x", "dRQ_x", "dRS_x", "dRT_x", "dRL'_x",
#                 "dRP'_x", "dRS'_x", "dRT'_x", "dL'P'_x", "dS'T'_x",
#                 "dST_x", "dPQ_x", "dPT_x", "dL'Q_x", "dST'_x",
#                 "dPL'_y", "dPQ_y", "dRQ_y", "dRS_y", "dTS_y", "dTT'_y"]

Fiducial_21 = ["dRP_y", "dRQ_y", "dRS_y", "dRT_y", "dRL'_y",
                "dRP'_y", "dRS'_y", "dRT'_y", "dL'P'_y", "dS'T'_y",
                "dST_y", "dPQ_y", "dPT_y", "dL'Q_y", "dST'_y",
                "dPL'_x", "dPQ_x", "dRQ_x", "dRS_x", "dTS_x", "dTT'_x"]

# Fiducial_21 = ["dRP_y", "dRS_y", "dRT_y", "dRL'_y",
#                 "dRP'_y", "dRT'_y", "dL'P'_y", "dS'T'_y",
#                 "dST_y", "dPQ_y", "dPT_y", "dL'Q_y", "dST'_y",
#                 "dPL'_x", "dTT'_x",
#                 "dRP_x", "dRS_x", "dRT_x", "dRL'_x",
#                 "dRP'_x", "dRT'_x", "dL'P'_x", "dS'T'_x",
#                 "dST_x", "dPQ_x", "dPT_x", "dL'Q_x", "dST'_x",
#                 "dPL'_y", "dTT'_y"]

# Fiducial_21 = ["dRP_y", "dRQ_y", "dRS_y", "dRT_y", "dST_y", "dPQ_y", "dPT_y", "dPQ_x", "dRQ_x", "dRS_x", "dTS_x"]

def gen_all(option=True):
    feature = pd.DataFrame()
    leads = [0, 1, 4, 10, 11]
    for lead in leads:
        feature_lead = feature_gen(Fiducial_21, lead, option)
        #option=False
        feature_lead = feature_lead.add_suffix(f'_{lead}')
        feature = pd.concat([feature, feature_lead], axis = 1)
    
    to_csv(feature, option)
    return feature
    
feature = gen_all()
#feature = gen_all(option=False)
# option=False
#print(f'Total feature: {len(Fiducial_21)}')

#%%

# features["dRP_x"].append(abs(Rx[i+1] - Px[i]))
# features["dRQ_x"].append(abs(Rx[i+1] - Qx[i]))
# features["dRS_x"].append(abs(Rx[i+1] - Sx[i+1]))
# features["dRT_x"].append(abs(Rx[i+1] - Tx[i+1]))

# features["dRL'_x"].append(abs(Rx[i+1] - L_x[i]))
# features["dRP'_x"].append(abs(Rx[i+1] - P_x[i]))
# features["dRS'_x"].append(abs(Rx[i+1] - S_x[i+1]))
# features["dRT'_x"].append(abs(Rx[i+1] - T_x[i+1]))

# features["dL'P'_x"].append(abs(L_x[i] - P_x[i]))
# features["dS'T'_x"].append(abs(S_x[i+1] - T_x[i+1]))
# features["dST_x"].append(abs(Sx[i+1] - Tx[i+1]))
# features["dPQ_x"].append(abs(Px[i] - Qx[i]))

# features["dPT_x"].append(abs(Px[i] - Tx[i+1]))
# features["dL'Q_x"].append(abs(L_x[i] - Qx[i]))
# features["dST'_x"].append(abs(Sx[i+1] - T_x[i+1]))

# # 找y軸 (電位)
# features["dPL'_y"].append(abs(data[Px[i]] - data[L_x[i]]))
# features["dPQ_y"].append(abs(data[Px[i]] - data[Qx[i]]))
# features["dRQ_y"].append(abs(data[Qx[i]] - data[Rx[i+1]]))
# features["dRS_y"].append(abs(data[Rx[i+1]] - data[Sx[i+1]]))
# features["dTS_y"].append(abs(data[Sx[i+1]] - data[Tx[i+1]]))
# features["dTT'_y"].append(abs(data[T_x[i+1]] - data[Tx[i+1]]))



#%%
# a = pd.read_csv('./training_features.csv')  

# select_feature = ["dS'T'_y_1", "dS'T'_y_4", 'dPQ_y_4', 'dRS_x_10', "dS'T'_y_10", 'dPQ_x_4', 'dPQ_y_1', 'dTS_x_4', 'dRQ_x_0', 'dRQ_x_4', 'dRS_x_11', 'dRS_x_4', "dS'T'_y_0", 'dRS_x_1', 'dTS_x_10', "dS'T'_y_11", 'dPQ_x_1', 'dPQ_x_0', 'dRQ_x_10', 'dRQ_x_11', 'dRS_x_0', 'dPQ_y_0', "dST'_y_1", 'dTS_x_0', 'dPT_y_4', 'dTS_x_11', 'dPQ_x_10', 'dTS_x_1', 'dRQ_x_1', "dST'_y_0", 'dPQ_x_11']

# for f in select_feature:
#     ff.plot_feature(a, f)