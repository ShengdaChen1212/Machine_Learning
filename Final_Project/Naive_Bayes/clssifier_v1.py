import numpy as np
import pandas as pd
import prepare_train_test as ptt

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
        #print(prob_para)
    return prob_para

# 算 normal distribution 的 pdf
def norm_pdf(x, mean, cov): 
    # exponent = np.exp(-1/2*(x-mean).T.dot(np.linalg.inv(cov)).dot(x-mean))
    # pdf = 1/np.sqrt((2 * np.pi)**len(x) * np.linalg.det(cov)) * exponent
    d = mean.shape[1]
    #print(mean.shape[0], mean.shape[1])
    cov_det = np.linalg.det(cov)
    #print(cov_det)
    cov_inv = np.linalg.inv(cov)
    pdf = (1/np.sqrt((2*np.pi)**d*cov_det)) * (np.exp((-0.5) * ((x-mean).T) @ cov_inv @ (x-mean)))
    #print(pdf)
    return pdf

# 輸入x，算4個class的機率
def prob_class_v0(x, prob_para): 
    prob = []
    for i in class_labels:
        pdf = norm_pdf(x, prob_para.iloc[i,0], prob_para.iloc[i,1])#*(prob_para.iloc[i,2])
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
# def predict_valid(select_feature, option=True):
#     feature_file_path_train = './training_features.csv' # train dataset 的.csv檔路徑
#     data = ptt.read_data(feature_file_path_train, 'ML_Train.csv', select_feature) # 讀feature並加上label
#     print(f'Before appending data: {len(data)}')
    
#     if option: # include noise
#         data = ptt.expand_data(data) # 增加有疾病的
#         print(f'After appending data: {len(data)}')
    
#     # split train test
#     data_train, data_val = ptt.split_train_test(data) 
    
#     # 預測機率
#     prob_para = prob_parameter(data_train)
    
#     '''lamda = [0.645, 0.115, 0.12, 0.119] #已經調好的參數（for lead1的9個特徵）'''
    
#     for i in range(len(data_val)): # do the testing data times
#         x = data_val.iloc[i, 3:]   # test data
#         x = np.array(x.values, dtype=float).reshape(-1, 1)
#         data_val.loc[i, 'pred'] = prob_class_v0(x, prob_para)
    
#     Accuracy(data_val)
#     pred_statistics(data_val)
#     return data_train, data_val, prob_para

def predict_valid(select_feature):
    train = ptt.Training(select_feature)
    data_train, data_val = train.split_train_test()
    prob_para = prob_parameter(data_train)
    
    for i in range(len(data_val)): # do the testing data times
        x = data_val.iloc[i, 3:]   # test data
        x = np.array(x.values, dtype=float).reshape(-1, 1)
        data_val.loc[i, 'pred'] = prob_class_v0(x, prob_para)
    
    Accuracy(data_val)
    pred_statistics(data_val)
    return data_train, data_val, prob_para

#%% testing
'''
def predict_test(prob_para, select_feature):
    feature_file_path_test  = './testing_features.csv'  # test dataset 的.csv檔路徑
    pred_file_path          = './Team_1.csv' # 存 test dataset 預測結果的檔案路徑
    data_test = ptt.read_data(feature_file_path_test, 'ML_Test.csv', select_feature, option = False)
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
'''

def predict_test(prob_para, select_feature):
    test = ptt.Testing(select_feature)
    data_test = test.read_data()
    
    for i in range(len(data_test)):
        x = data_test.iloc[i, 2:]
        x = np.array(x.values, dtype=float).reshape(-1, 1)
        data_test.loc[i, 'pred'] = prob_class_v0(x, prob_para)
            
    pred_statistics(data_test, option = False)
    pred_file_path          = './Team_1.csv' # 存 test dataset 預測結果的檔案路徑
    data_test_result = data_test[['SubjectId', 'pred']].copy()
    data_test_result['Label'] = data_test_result['pred'].astype(int)
    data_test_result.drop(columns='pred', inplace=True)
    
    data_test_result.to_csv(pred_file_path, index=False)
    
#%%
#0.488 select_feature = ["dTT'_x_11", "dTT'_x_10", 'dPT_y_0', "dTT'_x_1", 'dPT_y_11', 'dPT_y_10', 'dST_y_1', "dL'Q_y_4", 'dPT_y_1', "dST'_y_11", 'dTS_x_1', 'dPQ_x_1', 'dPQ_x_0']
#0.498 select_feature = ['dTS_x_1', 'dST_y_1', "dST'_y_11", 'dPT_y_11', 'dPT_y_0', 'dPT_y_10', "dL'Q_y_1", "dST'_y_10", 'dPT_y_1', "dST'_y_0", "dPL'_x_11", "dS'T'_y_0"]
#0.501 select_feature = ['dTS_x_1', 'dST_y_1', "dST'_y_11", 'dPT_y_11', 'dPT_y_0', 'dPT_y_10', "dL'Q_y_1", "dST'_y_10", 'dPT_y_1', "dST'_y_0", "dPL'_x_11", "dPL'_x_10"]
#0.502 select_feature = ['dTS_x_1', 'dST_y_1', "dST'_y_11", 'dPT_y_11', 'dPT_y_0', 'dPT_y_10', "dL'Q_y_1", "dST'_y_10", 'dPT_y_1', "dST'_y_0", "dPL'_x_11"]
#0.502 select_feature = ['dTS_x_1', 'dST_y_1', "dST'_y_11", 'dPT_y_11', 'dPT_y_0', 'dPT_y_10', "dL'Q_y_1", "dST'_y_10", 'dPT_y_1', "dST'_y_0", "dPL'_x_10"]
#select_feature = ["dRS'_x_10", "dS'T'_x_10", "dL'Q_x_0", 'dRS_x_10', "dPL'_x_11", "dTT'_x_1", 'dST_x_10', 'dRT_x_10', "dPL'_x_0", "dL'P'_x_11", "dTT'_x_10", 'dRS_x_11', 'dPQ_x_1', 'dPQ_x_0']
#select_feature = ["dS'T'_y_0", "dST'_y_0", "dST'_y_1", "dST'_y_11", "dST'_y_10", "dS'T'_y_11", 'dST_y_0', 'dST_y_1', 'dST_y_10', "dL'Q_y_1", 'dPT_y_0']
#select_feature = ["dS'T'_y_0", "dST'_y_0", "dST'_y_1", "dST'_y_11", "dST'_y_10", "dS'T'_y_11", 'dST_y_0', 'dST_y_1', 'dST_y_10', "dL'Q_y_1", 'dPT_y_0', "dS'T'_y_10", 'dPT_y_1', 'dRQ_x_0', 'dST_y_11', "dL'P'_y_1", 'dPQ_y_1', "dPL'_x_1", "dS'T'_y_1", 'dRQ_x_1', "dL'Q_y_0", 'dPT_y_10', 'dRT_y_10', 'dRT_y_11', 'dPQ_y_10']
select_feature = ["dST'_y_0", "dS'T'_y_0", "dST'_y_10", 'dST_y_0', "dST'_y_4", "dST'_y_1", "dST'_y_11", 'dPQ_y_0', 'dPQ_y_1', 'dPQ_y_4', "dL'Q_y_1", "dL'Q_y_4", "dS'T'_y_11", 'dRT_y_0', "dS'T'_y_10", 'dST_y_1', "dL'Q_y_0", 'dTS_x_0', 'dRQ_x_1', 'dST_y_4']
#%%
data_train, data_val, prob_para = predict_valid(select_feature)
predict_test(prob_para, select_feature)

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