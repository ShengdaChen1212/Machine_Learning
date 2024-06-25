import numpy as np
import pandas as pd
from functions import denoise
import find_feature_v2 as ff
from multiprocessing import Pool, cpu_count

def process_person(args):
    data, Fiducial_21, person = args
    feature_num = len(Fiducial_21)
    features = {key: [] for key in Fiducial_21}
    stop = 0

    try:
        points = ff.find_features_v2(data, 1000)
        Px, Qx, Rx, Sx, Tx = points['P'], points['Q'], points['R'], points['S'], points['T']
        L_x, P_x, S_x, T_x = points["L'"], points["P'"], points["S'"], points["T'"]

        if len(Rx) > 30 or len(Rx) <= 5:
            stop = 1

        for p in [Rx, Px, Tx, Qx, Sx]:
            if len(p) == 0 or max(p) >= len(data):
                stop = 1
            else:
                ystd = np.std(data[p])
                if ystd > 0.15:
                    stop = 1

        if stop == 1:
            return [0] * feature_num, person, 1

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
                #features["dTS_x"].append(abs(Sx[i+1] - Tx[i+1]))
                features["dTT'_x"].append(abs(T_x[i+1] - Tx[i+1]))
                
                features["dRP_x"].append(abs(Rx[i+1] - Px[i]))
                #features["dRQ_x"].append(abs(Rx[i+1] - Qx[i]))
                #features["dRS_x"].append(abs(Rx[i+1] - Sx[i+1]))
                features["dRT_x"].append(abs(Rx[i+1] - Tx[i+1]))
                
                features["dRL'_x"].append(abs(Rx[i+1] - L_x[i]))
                features["dRP'_x"].append(abs(Rx[i+1] - P_x[i]))
                features["dRS'_x"].append(abs(Rx[i+1] - S_x[i+1]))
                features["dRT'_x"].append(abs(Rx[i+1] - T_x[i+1]))
                
                features["dL'P'_x"].append(abs(L_x[i] - P_x[i]))
                features["dS'T'_x"].append(abs(S_x[i+1] - T_x[i+1]))
                features["dST_x"].append(abs(Sx[i+1] - Tx[i+1]))
                #features["dPQ_x"].append(abs(Px[i] - Qx[i]))
                
                features["dPT_x"].append(abs(Px[i] - Tx[i+1]))
                features["dL'Q_x"].append(abs(L_x[i] - Qx[i]))
                features["dST'_x"].append(abs(Sx[i+1] - T_x[i+1]))
                
                features["dPL'_y"].append(abs(data[Px[i]] - data[L_x[i]]))
                #features["dPQ_y"].append(abs(data[Px[i]] - data[Qx[i]]))
                #features["dRQ_y"].append(abs(data[Qx[i]] - data[Rx[i+1]]))
                #features["dRS_y"].append(abs(data[Rx[i+1]] - data[Sx[i+1]]))
                #features["dTS_y"].append(abs(data[Sx[i+1]] - data[Tx[i+1]]))
                features["dTT'_y"].append(abs(data[T_x[i+1]] - data[Tx[i+1]]))

        list_21 = [np.mean(features[index]) for index in Fiducial_21]
        return list_21, person, 0

    except:
        return [0] * feature_num, person, 1

def get_21(origindata, lead, Fiducial_21):
    feature_list = []
    stop_cnt = 0
    feature_num = len(Fiducial_21)
    
    with Pool(processes=cpu_count()) as pool:
        results = []
        for person in range(len(origindata)):
            data = origindata[person, lead, :]
            data = denoise(data)
            results.append(pool.apply_async(process_person, ((data, Fiducial_21, person),)))

        for result in results:
            list_21, person, stop = result.get()
            feature_list.append(list_21)
            stop_cnt += stop
            print(person, ' x' if stop else '')

    # replace 0 to mean 
    feature = pd.DataFrame(feature_list, columns=Fiducial_21)
    print(feature.isna().sum(axis=0))
    Sums = feature.sum()
    Means = Sums / (len(feature)-stop_cnt)
    
    for data in range(0,6000):
        if 0 in feature.iloc[data, :].values:
            feature.iloc[data,:] = Means
            
    
    return feature

def feature_gen(Fiducial_21, lead, option=True):
    if option:
        print(f'Generating Lead {lead} Training Features: ')
        print('='*30)
        dataset = np.load('ML_Train.npy', mmap_mode='r')
    else:
        print(f'Generating Lead {lead} Testing Features: ')
        print('='*30)
        dataset = np.load('ML_Test.npy', mmap_mode='r')
        
    feature = get_21(dataset, lead, Fiducial_21)
    return feature

def to_csv(feature, option=True):
    if option:
        print('='*30)
        feature.to_csv('training_1000_80.csv', index=False)
        print('Generating Training CSV DONE!')
    else:
        print('='*30)
        feature.to_csv('testing_1000_80.csv', index=False)
        print('Generating Testing CSV DONE!')
    return None

Fiducial_21 = ["dRP_y", "dRQ_y", "dRS_y", "dRT_y", "dRL'_y",
                "dRP'_y", "dRS'_y", "dRT'_y", "dL'P'_y", "dS'T'_y",
                "dST_y", "dPQ_y", "dPT_y", "dL'Q_y", "dST'_y",
                "dPL'_x", "dTT'_x",
                "dRP_x", "dRQ_x", "dRS_x", "dRT_x", "dRL'_x",
                "dRP'_x", "dRS'_x", "dRT'_x", "dL'P'_x", "dS'T'_x",
                "dST_x", "dPQ_x", "dPT_x", "dL'Q_x", "dST'_x",
                "dPL'_y", "dTT'_y"]

def gen_all(option=True):
    feature = pd.DataFrame()
    leads = [0, 1, 4, 5, 9, 10, 11]
    for lead in leads:
        feature_lead = feature_gen(Fiducial_21, lead, option)
        feature_lead = feature_lead.add_suffix(f'_{lead}')
        feature = pd.concat([feature, feature_lead], axis=1)
    
    to_csv(feature, option)
    return feature

if __name__ == '__main__':
    feature = gen_all()
    # option=False
#%%
'''
import numpy as np
import pandas as pd
from functions import denoise
import find_feature_v2 as ff
from multiprocessing import Pool, cpu_count

data1 = pd.read_csv("training_features_0531.csv")
data2 = pd.read_csv("training_features_1.csv")
df = pd.read_csv("ML_Train.csv")

df['class'] = ''
df.loc[df.loc[:,'Label'] == 'NORM', 'class'] = 0
df.loc[df.loc[:,'Label'] == 'STTC', 'class'] = 1
df.loc[df.loc[:,'Label'] == 'CD',   'class'] = 2
df.loc[df.loc[:,'Label'] == 'MI',   'class'] = 3
data2=pd.concat([data2, df['class']], axis=1)
data2.to_csv('training_features_data2.csv', index=False)
#%%
import numpy as np
import pandas as pd
from functions import denoise
import find_feature_v2 as ff
from multiprocessing import Pool, cpu_count

data1 = pd.read_csv("training_features_1.csv")
data2 = pd.read_csv("training_features_data2.csv")
df = pd.read_csv("ML_Train.csv")

for i in range(12209):
    if 0 in data1.iloc[i, :].values :
        data2 = data2.drop(index=[i])
data2.to_csv('training_features_data2.csv', index=False)
#%%
import numpy as np
import pandas as pd
from functions import denoise
import find_feature_v2 as ff
from multiprocessing import Pool, cpu_count

data2 = pd.read_csv("training_features_data2.csv")

np.random.seed(524773)
train_num = np.sort(np.random.choice(np.arange(6651), 1500, replace=False))

train1 = data2.iloc[train_num,:]
train2 = data2.iloc[6651:,:]
result=pd.concat([train1, train2], axis=0)
result.to_csv('training_features_result.csv', index=False)                  
'''











