import numpy as np
import pandas as pd

#%%
def read_data(feature_file_path, label_file_path, option=True): # 讀feature並加上label
    # data_feature = pd.read_csv(feature_file_path)       # read feature
    # data_label   = pd.read_csv(label_file_path)         # read label
    
    a = pd.read_csv(feature_file_path)       # read feature
    data_label   = pd.read_csv(label_file_path)         # read label
    # # for lead 0:
    # select_feature = ["dRS'_x", "dS'T'_x", "dTT'_y", "dTS_y", "dST'_x"]
    
    # for lead 11:
    select_feature = ['dRP_x_4', "dRP'_x_4", "dRL'_x_4", "dST'_x_1", 'dTS_y_1', 'dPT_x_7', 'dPQ_x_7', 'dRS_x_4', 'dST_x_1']
    
    data_feature = a[select_feature]
    
    data_feature = data_feature.reset_index(drop=True)
    data_label   = data_label.reset_index(drop=True)
    
    data = pd.DataFrame()
    data['SubjectId'] = data_label['SubjectId']
    if option: # changing label
        data['Label'] = data_label['Label'].replace({'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3})
    data['pred'] = None
    data = pd.concat([data, data_feature], axis=1)
    
    # chaos = {}
    # for i in range(len(data)):
    #     if (data.iloc[i, -3] ==0.0):
    #         #print(data.loc[i, "Label"])
    #         if data.loc[i, "Label"] in chaos:
    #             chaos[data.loc[i, "Label"]] += 1
    #         else:
    #             chaos[data.loc[i, "Label"]] = 1
                
    if option:# 濾掉 0, Nan 的數據
        data = data.drop(data[(data.iloc[:, -3] ==0.0)].index)
        data = data.drop(data[pd.isna(data.iloc[:, -3])].index)
        data = data.iloc[1500:].reset_index(drop=True)
    data = data.reset_index(drop=True)
    return data

def expand_data(data): # 增加疾病的數據
    data_norm = data[data['Label'] == 0].copy()
    new_data = data_norm.copy()
    
    for label in range(1,4):
        class_data = data[data['Label'] == label].copy()
        new_class_data = class_data.copy()
        add_dict = {1: range(4), 2: range(2), 3: range(3)} # 不同label增加不同倍
        add = add_dict.get(label)
        
        for i in add: #加上~倍的數據
            noise = np.random.normal(0, 0.005, size=(class_data.shape[0], class_data.shape[1]-3))
            noisy_data = class_data.copy()
            noisy_data.iloc[:, 3:] += noise
            new_class_data = pd.concat([new_class_data, noisy_data], axis=0)
            new_class_data.reset_index(drop=True, inplace=True)
    
        new_data = pd.concat([new_data, new_class_data], axis=0)
        new_data.reset_index(drop=True, inplace=True)
    return new_data

def split_train_test(data): # 分train test dataset
    data = data.reset_index(drop=True)
    np.random.seed(40)
    rand_seq = np.random.permutation(len(data))
    
    val_size = round(len(data) * 0.2)
    val_index = rand_seq[:val_size]
    
    data_val = data.iloc[val_index]
    data_train = data.drop(val_index)
    
    data_val = data_val.reset_index(drop=True)
    data_train = data_train.reset_index(drop=True)
    return data_train, data_val

#%% 計算機率
class_labels = [0, 1, 2, 3]

# 算每一個class的機率參數
def prob_parameter(data): 
    prob_para = pd.DataFrame()
    for label in class_labels: # 算每一個class要算各自的mean和cov
        x     = data.loc[data['Label'] == label]
        x     = x.iloc[:, 3:] 
        prior = len(x)/len(data)
        mean  = x.mean(axis=0).to_numpy().reshape(-1, 1)
        cov   = x.cov() 
        prob_para = prob_para.append({'mean': mean, 'con': cov, 'prior': prior}, ignore_index=True)
        # print(prob_para)
    return prob_para

# 算 normal distribution 的 pdf
def norm_pdf(x, mean, cov): 
    # exponent = np.exp(-1/2*(x-mean).T.dot(np.linalg.inv(cov)).dot(x-mean))
    # pdf = 1/np.sqrt((2 * np.pi)**len(x) * np.linalg.det(cov)) * exponent
    d = mean.shape[1]
    #print(mean.shape[0], mean.shape[1])
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    pdf = (1/np.sqrt((2*np.pi)**d*cov_det)) * (np.exp((-0.5) * ((x-mean).T) @ cov_inv @ (x-mean)))
    # print(pdf)
    return pdf

# 輸入x，算4個class的機率
def prob_class_v0(x, prob_para): 
    prob = []
    for i in class_labels:
        pdf = norm_pdf(x, prob_para.iloc[i,0], prob_para.iloc[i,1])*(prob_para.iloc[i,2])
        prob.append(float(pdf))
        #print(prob)
    label = prob.index(max(prob)) # 找prob最大的index
    return class_labels[label]

'''
def prob_class(x, lamda): 
    prob = []
    for i in class_labels:
        pdf = norm_pdf(x, prob_para.iloc[i,0],prob_para.iloc[i,1])*(prob_para.iloc[i,2])
        prob.append(float(pdf))
    lam_0, lam_1, lam_2, lam_3 = lamda[0], lamda[1], lamda[2], lamda[3]
    lamda = np.array([[0, lam_0, lam_0, lam_0],[lam_1, 0, lam_1, lam_1],[lam_2, lam_2, 0, lam_2], [lam_3, lam_3, lam_3, 0]])
    risk = (lamda @ prob).tolist()
    label = risk.index(min(risk)) # 找risk最小的index
    # label = prob.index(max(prob)) # 找prob最大的index
    return class_labels[label]
'''

def pred_statistics(data, option=True):
    if option:
        print('\nValidation result')
        result = pd.DataFrame(index=class_labels, columns=['label', 'pred', 'correct', 'wrong'])
        for label in class_labels:
            label_prop = (data['Label'] == label).sum() / len(data)
            pred_prop  = (data['pred']  == label).sum() / len(data)
            correct    = ((data['Label'] == label) & (data['pred'] == label)).sum() / len(data[data['pred'] == label])
            wrong      = ((data['Label'] != label) & (data['pred'] == label)).sum() / len(data[data['pred'] == label])
            result.loc[label] = [round(label_prop,2), round(pred_prop,2), round(correct,2),round(wrong,2)]
    else:
        print('\nTest result')
        result = pd.DataFrame(index=class_labels, columns=['pred'])
        for label in class_labels:
            pred_prop = (data['pred'] == label).sum() / len(data)
            result.loc[label] = [round(pred_prop,2)]
    
    print('='*30); print(result); print('='*30)
    return result

# 計算 accuracy
def Accuracy(data):
    accuracy = (data['Label'] == data['pred']).sum()/len(data)
    print(f'\nAccuracy : {accuracy:.2f}')
    return None

# 調 lambda 參數
def toss_lamda(data, lamda): 
    diff_list = []
    for label in class_labels:
        label_prop = (data['Label'] == label).sum() / len(data)
        pred_prop = (data['pred'] == label).sum() / len(data)
        diff = round(pred_prop - label_prop,3)
        diff_list.append(diff)
    return diff_list

#%% training validation
def predict_valid(option=True):
    feature_file_path_train = './training_features.csv' # train dataset 的.csv檔路徑
    
    
    data = read_data(feature_file_path_train, 'ML_Train.csv') # 讀feature並加上label
    print(f'Before appending data: {len(data)}')
    
    if option: # include noise
        data = expand_data(data) # 增加有疾病的
        print(f'After appending data: {len(data)}')
    
    # split train test
    data_train, data_val = split_train_test(data) 
    
    # 預測機率
    prob_para = prob_parameter(data_train)
    
    '''lamda = [0.645, 0.115, 0.12, 0.119] #已經調好的參數（for lead1的9個特徵）'''
    
    for i in range(len(data_val)): # do the testing data times
        x = data_val.iloc[i, 3:]   # test data
        x = np.array(x.values, dtype=float).reshape(-1, 1)
        data_val.loc[i, 'pred'] = prob_class_v0(x, prob_para)
    
    Accuracy(data_val)
    pred_statistics(data_val)
    return data, prob_para

#%% testing
def predict_test(prob_para):
    feature_file_path_test  = './testing_features.csv'  # test dataset 的.csv檔路徑
    pred_file_path          = './Team_1.csv' # 存 test dataset 預測結果的檔案路徑
    data_test = read_data(feature_file_path_test, 'ML_Test.csv', option = False)
    # prob_para = prob_parameter(data)
    
    for i in range(len(data_test)):
        x = data_test.iloc[i, 2:]
        x = np.array(x.values, dtype=float).reshape(-1, 1)
        if data_test.iloc[i, -3] == 0:
            data_test.loc[i, 'pred'] = 2
        else:
            data_test.loc[i, 'pred'] = prob_class_v0(x, prob_para)
    pred_statistics(data_test, option = False)
    
    data_test_result = data_test[['SubjectId', 'pred']].copy()
    data_test_result['Label'] = data_test_result['pred'].astype(int)
    data_test_result.drop(columns='pred', inplace=True)
    
    data_test_result.to_csv(pred_file_path, index=False)
    
#%%

data, prob_para = predict_valid()
# option=False
predict_test(prob_para)

#%%
# 下方這段是用while loop，找適合的lambda參數，讓驗證集預測的比例接近實際的
# 對不同的特徵組合，需要試不同的initial guess才能找到適合的lambda參數
## =============================================================================
# lamda =  [0.5, 0.1, 0.14, 0.1] # initial guess
# loss = 1
# threshold = 0.10
# lr = 0.001 # learning_rate
# lr_d = 0.0001  # learning_rate_decreasing
# 
# while loss > threshold:
#     diff_list = toss_lamda(data_val, lamda)
#     loss = sum( [abs(i) for i in diff_list])
#     print(f'loss : {loss:.2f}')
#     print(f'lamda : {lamda}')
#     
#     # index = max(range(len(diff_list)), key=lambda i: abs(diff_list[i]))
#     sorted_indices = sorted(range(len(diff_list)), key=lambda i: abs(diff_list[i]), reverse=True)
#     max_indices = sorted_indices[:2]
#     for i in max_indices:
#         lamda[i] += lr if diff_list[i] > 0 else -lr
#     lr -= lr_d
# 
#     lamda = [round(i,3) for i in lamda]
#     
#     for i in range(len(data_val)):
#         x = data_val.iloc[i, 3:]
#         x = np.array(x.values, dtype=float).reshape(-1, 1)
#         data_val.loc[i, 'pred'] = prob_class(x, lamda)
#     if loss < threshold or any(value < 0.01 or value > 0.99 for value in lamda):
#         break
# 
## =============================================================================