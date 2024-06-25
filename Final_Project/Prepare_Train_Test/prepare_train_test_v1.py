import numpy as np
import pandas as pd


#%%
class DATA_format:
    def __init__(self, select_feature):
        self._feature_file_path = ''
        self._label_file_path   = ''
        self._select_feature    = select_feature
        self._option            = True
        
    def read_data(self):
        data_feature = pd.read_csv(self._feature_file_path) # 從 training/testing_feature 讀 feature
        data_label   = pd.read_csv(self._label_file_path)   # 從 ML_train/test 讀 label
        data_feature = data_feature[self._select_feature]   # 從 input 選 select feature
        
        data_feature = data_feature.reset_index(drop=True)  # 修改排序避免錯誤
        data_label   = data_label.reset_index(drop=True)
        
        data = pd.DataFrame()
        data['SubjectId'] = data_label['SubjectId']
        if self._option:                                    # 當模式為 train 需要轉換 label 為數字
            data['Label'] = data_label['Label'].replace({'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3})
        data['pred'] = None                                 # 設定空 prediction
        data = pd.concat([data, data_feature], axis=1)      # 將 subject ID、label、prediction 接在一起
        if self._option: # 濾掉 0, Nan 的數據
            data = data.drop(data[(data.iloc[:, -3] ==0.0)].index)
        print('='*30); print('Finish reading data.')
        return data

class Training(DATA_format):
    def __init__(self, select_feature):
        super().__init__(select_feature)
        self._feature_file_path = './training_features_1000_50.csv'#'./train_0531.csv'#'./training_features_total406.csv'
        self._label_file_path   = './ML_Train.csv'
        
    def _pick_data(self):
        df = self.read_data()
        
        # 先篩選出某個 label 值的資料，例如 'label1'
        label_value = 0
        subset_df = df[df['Label'] == label_value]
        
        # 從該 label 的資料中隨機選取 1500 行
        sampled_df = subset_df.sample(n=1500, random_state=42)
        
        # 篩選出其他 label 的資料
        remaining_df = df[df['Label'] != label_value]
        
        # 將選取的 1500 行資料與其他 label 的資料合併
        result_df = pd.concat([sampled_df, remaining_df])
        return result_df
        
    def split_train_test(self, random_num=40): # 分train/test dataset
        data = self._pick_data()
        data['pred'] = None    
        data = data.reset_index(drop=True)          # 修改排序避免錯誤
        np.random.seed(random_num)                  # 產生 random seed
        rand_seq = np.random.permutation(len(data)) # 產生隨機序列
        
        val_size  = round(len(data) * 0.2)          # train / test = 8:2
        val_index = rand_seq[:val_size]             # index 取用 random sequence
        
        data_validation = data.iloc[val_index]      # 選取 valid index
        data_train      = data.drop(val_index)      # 扣掉 valid index
        
        data_validation = data_validation.reset_index(drop=True) # 修改排序避免錯誤
        data_train      = data_train.reset_index(drop=True)
        
        print('='*30); print('Finish splitting data.')
        return data_train, data_validation
    
    def feature_label(self):
        data_train, data_validation = self.split_train_test()
        X_train = np.array(data_train.loc[:, self._select_feature].values, dtype=float)
        Y_train = data_train.loc[:, 'Label'].values
        X_valid = np.array(data_validation.loc[:, self._select_feature].values, dtype=float)
        Y_valid = data_validation.loc[:, 'Label'].values
        return X_train, Y_train, X_valid, Y_valid
        
class Testing(DATA_format):
    def __init__(self, select_feature):
        super().__init__(select_feature)
        self._feature_file_path = './testing_features_1000_50.csv'#'./testing_0531.csv' #'./testing_features_total406.csv'
        self._label_file_path   = './ML_Test.csv' 
        self._option            = False

        
        
    