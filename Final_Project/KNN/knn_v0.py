import numpy as np
import prepare_train_test as ptt
    
#%%
class KNNClassifier:
    def __init__(self, k=85):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []
        count = 0
        for x in X_test:
            predictions.append(self._predict(x))
            count += 1
            print(count)
        #predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        
        # 計數標籤出現的次數
        label_count = {}
        for label in k_nearest_labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        # 選擇最多次的標籤
        most_common_label = max(label_count, key=label_count.get)
        return most_common_label
    
#%%
select_feature = ["dST'_y_0", "dS'T'_y_0", "dST'_y_10", 'dST_y_0', "dST'_y_4", "dST'_y_1", "dST'_y_11", 'dPQ_y_0', 'dPQ_y_1', 'dPQ_y_4', "dL'Q_y_1", "dL'Q_y_4", "dS'T'_y_11", 'dRT_y_0', "dS'T'_y_10", 'dST_y_1', "dL'Q_y_0", 'dTS_x_0', 'dRQ_x_1', 'dST_y_4']

train = ptt.Training(select_feature)
X_train, Y_train, X_valid, Y_valid = train.feature_label()

# feature_file_path_train = './training_features.csv'
# data = ptt.read_data(feature_file_path_train, 'ML_Train.csv', select_feature)
# data = ptt.expand_data(data)
# data_train, data_val = ptt.split_train_test(data) 

# X_train = np.array(data_train.loc[:, select_feature].values, dtype=float)
# Y_train = data_train.iloc[:, 1].values
# X_valid = np.array(data_val.loc[:, select_feature].values, dtype=float)
# Y_valid = data_val.iloc[:, 1].values

knn = KNNClassifier()
knn.fit(X_train,Y_train)
#%% validation
predictions = knn.predict(X_valid)
Y_valid = list(Y_valid)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_valid,predictions))

#%% test
# feature_file_path_test  = './testing_features.csv'  # test dataset 的.csv檔路徑
# testset = ptt.read_data(feature_file_path_test, 'ML_Test.csv', select_feature, option=False)
# X_test = testset.loc[:, select_feature]
# X_test = np.array(X_test.values, dtype=float)

test = ptt.Testing(select_feature)
testset = test.read_data()
X_test = testset.loc[:, select_feature]
X_test = np.array(X_test.values, dtype=float)
predictions = knn.predict(X_test)
#%%
pred_file_path          = './Team_1.csv' # 存 test dataset 預測結果的檔案路徑
testset.loc[:, 'pred'] = predictions
data_test_result = testset[['SubjectId', 'pred']].copy()
data_test_result['Label'] = data_test_result['pred']
data_test_result.drop(columns='pred', inplace=True)
data_test_result.to_csv(pred_file_path, index=False)


