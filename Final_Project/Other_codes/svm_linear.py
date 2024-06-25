import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def accuracy_score(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / total


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    
    return cm

class LinearSVMUsingSoftMargin:
    def __init__(self, C=0.1):
        self._support_vectors = None
        self.C = C
        self.beta = None
        self.b = None
        self.X = None
        self.y = None
        self.classes = None
        self.classifiers = None

    def __decision_function(self, X):
        return X.dot(self.beta) + self.b

    def __cost(self, margin):
        return (1 / 2) * self.beta.dot(self.beta) + self.C * np.sum(np.maximum(0, 1 - margin))

    def __margin(self, X, y):
        return y * self.__decision_function(X)

    def fit(self, X, y, lr=0.0001, epochs=1000):
        self.classes = np.unique(y) #找出數據中的不同類別
        self.classifiers = {}

        for class_label in self.classes:
            y_binary = np.where(y == class_label, 1, -1) #是該類別為1、不是為-1
            svm = LinearSVMUsingSoftMargin(C=self.C)
            svm._fit_binary(X, y_binary, lr, epochs)
            self.classifiers[class_label] = svm

    def _fit_binary(self, X, y_binary, lr, epochs):
        self.n, self.d = X.shape #n是樣本數量、d是特徵數量
        self.beta = np.random.randn(self.d) #隨機生成的初始beta
        self.b = 0

        for _ in range(epochs):
            margin = self.__margin(X, y_binary)
            misclassified_pts_idx = np.where(margin < 1)[0] #錯誤的點(距離小於1)

            d_beta = self.beta - self.C * y_binary[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - lr * d_beta

            d_b = -self.C * np.sum(y_binary[misclassified_pts_idx])
            self.b = self.b - lr * d_b

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classes)))

        for i, class_label in enumerate(self.classes):
            svm = self.classifiers[class_label]
            predictions[:, i] = svm.__decision_function(X)

        return self.classes[np.argmax(predictions, axis=1)]

#%%
def read_data(feature_file_path, label_file_path, option=True): # 讀feature並加上label
    data_feature = pd.read_csv(feature_file_path)       # read feature
    data_label   = pd.read_csv(label_file_path)         # read label

    # 確保沒有重複的標籤
    data_feature = data_feature.drop_duplicates()
    data_label = data_label.drop_duplicates()

    # 重置索引
    data_feature = data_feature.reset_index(drop=True)
    data_label   = data_label.reset_index(drop=True)

    data = pd.DataFrame()
    data['SubjectId'] = data_label['SubjectId']
    if option: # changing label
        data['Label'] = data_label['Label'].replace({'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3})
    data['pred'] = None
    data = pd.concat([data, data_feature], axis=1)
    
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

# 分train test dataset
def split_train_test(data):
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

#%%
'''
predictions = [] 
for k in range(1):
    print('lead ',k)
    feature_file_path_train = f'training_lead{k}_features.csv' # train file path
    feature_file_path_test = f'testing_lead{k}_features.csv' # test file path
    
    data = read_data(feature_file_path_train, 'ML_Train.csv')
    data_test = read_data(feature_file_path_test, 'ML_Test.csv', option = False)
    data = expand_data(data) # 增加有疾病的
    data_train, data_val = split_train_test(data) 
    select_feature = ["dRS_x", "dRS'_x", "dL'P'_x", "dS'T'_x", "dST_x", "dTT'_y", "dTS_y"]
    
    X_train = data_train[select_feature] #[['0', '1', '2', '3', '4', '5', '6', '7']]  # 特徵    
    #X_train = data_train[['0', '1', '2', '3', '4', '5', '6', '7','8','9']]  # 特徵
    #X_train = data_train[['dRQx','dRS_x','dS_T_x','dRQy','dRSy','dTSy','dTT_y']]  # 特徵    
    y_train = data_train['Label']  # 類別
    
    X_test = data_val[select_feature] #[['0', '1', '2', '3', '4', '5', '6', '7','8','9']]
    #X_test = data_val[['0', '1', '2', '3', '4', '5', '6', '7']]
    #X_test = data_val[['dRQx','dRS_x','dS_T_x','dRQy','dRSy','dTSy','dTT_y']]
    y_test = data_val['Label']
           
    # 特徵標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    svm = LinearSVMUsingSoftMargin(C=1.0)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test) 
    predictions.append(y_pred)  
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
      
    #計算每個類別的準確率
    class_accuracy = {}
    for i in range(4):
        class_accuracy[i] = cm[i, i] / np.sum(cm[i, :])
    
    print("準確率：", accuracy)
    print("每個類別的準確率：", class_accuracy)

predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.transpose()

#column_names = [f'lead{k}' for k in range(12)]
#predictions_df.columns = column_names
'''

#Test
def run_SVM(lead):
    predictions = [] 
    kk = lead
    for k in [kk]:
        print('lead ',k)
        feature_file_path_train = f'training_lead{k}_features.csv' # train file path
        feature_file_path_test = f'testing_lead{k}_features.csv' # test file path
        
        data = read_data(feature_file_path_train, 'ML_Train.csv')
        data_test = read_data(feature_file_path_test, 'ML_Test.csv', option = False)
        data = expand_data(data) # 增加有疾病的
        data_train, data_val = data, data_test
        select_feature = ["dRS_x", "dRS'_x", "dL'P'_x", "dS'T'_x", "dST_x", "dTT'_y", "dTS_y"]
        
        X_train = data_train[select_feature] #[['0', '1', '2', '3', '4', '5', '6', '7']]  # 特徵    
        #X_train = data_train[['0', '1', '2', '3', '4', '5', '6', '7','8','9']]  # 特徵
        #X_train = data_train[['dRQx','dRS_x','dS_T_x','dRQy','dRSy','dTSy','dTT_y']]  # 特徵    
        y_train = data_train['Label']  # 類別
        
        X_test = data_val[select_feature] #[['0', '1', '2', '3', '4', '5', '6', '7','8','9']]
        #X_test = data_val[['0', '1', '2', '3', '4', '5', '6', '7']]
        #X_test = data_val[['dRQx','dRS_x','dS_T_x','dRQy','dRSy','dTSy','dTT_y']]
        #y_test = data_val['Label']
            
        # 特徵標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        svm = LinearSVMUsingSoftMargin(C=1.0)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test) 
        predictions.append(y_pred)  

    for i in range(len(data_test)):
        x = data_test.iloc[i, 2:]
        x = np.array(x.values, dtype=float).reshape(-1, 1)
        data_test.loc[i, 'pred'] = predictions[0][i]

    data_test_result = data_test[['SubjectId', 'pred']].copy()
    data_test_result['Label'] = data_test_result['pred'].astype(int)
    data_test_result.drop(columns='pred', inplace=True)

    data_test_result.to_csv(f'SVM_Predict{kk}.csv', index=False)
'''
predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.transpose()
predictions_df.to_csv('SVM_Predict.csv', index=False)
'''
#column_names = [f'lead{k}' for k in range(12)]
#predictions_df.columns = column_names