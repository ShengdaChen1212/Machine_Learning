import numpy as np
from joblib import Parallel, delayed
import prepare_train_test as ptt
from sklearn.metrics import accuracy_score
import multiprocessing

class KNNClassifier:
    def __init__(self, k=79):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        # Using joblib for parallel processing
        num_cores = multiprocessing.cpu_count()
        predictions = Parallel(n_jobs=num_cores, backend="multiprocessing")(delayed(self._predict)(x) for x in X_test)
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        
        # Count labels
        label_count = {}
        for label in k_nearest_labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        # Select the most common label
        most_common_label = max(label_count, key=label_count.get)
        return most_common_label

if __name__ == "__main__":
    # Data preparation
    select_feature = ["dS'T'_y_0", "dST'_y_0", "dST'_y_1", "dST'_y_11", "dST'_y_10", "dS'T'_y_11", 'dST_y_0', 'dST_y_1', 'dST_y_10', "dL'Q_y_1", 'dPT_y_0', "dS'T'_y_10", 'dPT_y_1', 'dRQ_x_0', 'dST_y_11', "dL'P'_y_1", 'dPQ_y_1', "dPL'_x_1", "dS'T'_y_1", 'dRQ_x_1', "dL'Q_y_0", 'dPT_y_10', 'dRT_y_10', 'dRT_y_11', 'dPQ_y_10', 'dRT_y_1', "dPL'_x_0", 'dTS_x_10', "dRT'_y_10", 'dRS_x_10', 'dRS_y_0', 'dPQ_x_0', 'dRQ_x_11']
    feature_file_path_train = './training_features.csv'
    data = ptt.read_data(feature_file_path_train, 'ML_Train.csv', select_feature)
    data = ptt.expand_data(data)
    data_train, data_val = ptt.split_train_test(data)

    X_train = np.array(data_train.loc[:, select_feature].values, dtype=float)
    Y_train = data_train.iloc[:, 1].values
    X_valid = np.array(data_val.loc[:, select_feature].values, dtype=float)
    Y_valid = data_val.iloc[:, 1].values

    # Train and validate the model
    knn = KNNClassifier()
    knn.fit(X_train, Y_train)

    # predictions = knn.predict(X_valid)
    # print(accuracy_score(Y_valid, predictions))

    # Test data prediction
    feature_file_path_test = './testing_features.csv'
    pred_file_path = './Team_1.csv'
    testset = ptt.read_data(feature_file_path_test, 'ML_Test.csv', select_feature, option=False)
    X_test = testset.loc[:, select_feature]
    X_test = np.array(X_test.values, dtype=float)
    predictions = knn.predict(X_test)

    # Save test predictions
    testset.loc[:, 'pred'] = predictions
    data_test_result = testset[['SubjectId', 'pred']].copy()
    data_test_result['Label'] = data_test_result['pred']
    data_test_result.drop(columns='pred', inplace=True)

    data_test_result.to_csv(pred_file_path, index=False)
