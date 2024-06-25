import numpy as np
import pandas as pd
#import prepare_train_test as ptt
import prepare_train_test_2 as ptt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 指定要選擇的特徵
# 0.619 select_feature = ['dPT_y_11', 'dPT_y_10', 'dST_y_11', 'Power_Ratio_0', 'Spectral_Entropy_0', "dRS'_x_11", "dRS'_x_0", 'dPT_y_9', 'Power_Ratio_11', 'Power_Ratio_4', 'Power_Ratio_1', "dST'_y_11", 'Spectral_Entropy_1', 'dST_y_10', "dRS'_x_10", "dRS'_x_1", 'Power_Ratio_5', 'Spectral_Entropy_9', 'Spectral_Entropy_11', 'Spectral_Entropy_4', "dS'T'_x_10", 'dST_y_1', 'Spectral_Entropy_10', "dS'T'_x_11", 'flourish_4', 'dRT_y_10', 'dRT_y_11', "dRS'_x_9", 'Power_Ratio_10', 'dRT_y_1', 'dST_y_0', "dS'T'_x_0", "dS'T'_x_9", "dST'_y_10", 'Peak_Frequency_9', 'dRP_y_1', 'dRS_y_1', 'Spectral_Entropy_5', 'Power_Ratio_9', 'dRT_y_0', "dS'T'_x_1", 'flourish_5', 'dRS_x_11', 'bpm_10', 'dRP_y_11', 'dRS_y_4', 'dRS_x_0', 'sdnn_4', "dL'P'_x_11", 'bpm_11']
# 0.619 select_feature = ['dPT_y_11', "dTT'_y_1", 'dPT_y_10', "dTT'_y_0", 'dST_y_11', 'Power_Ratio_0', 'Spectral_Entropy_0', 'dPT_y_0', "dRS'_x_11", "dRS'_x_0", 'dPT_y_9', 'Power_Ratio_11', 'Power_Ratio_4', 'Power_Ratio_1', "dST'_y_11", 'dPT_y_1', 'Spectral_Entropy_1', 'dST_y_10', "dRS'_x_10", "dRS'_x_1", 'Power_Ratio_5', 'Spectral_Entropy_9', 'Spectral_Entropy_11', "dS'T'_y_0", 'Spectral_Entropy_4', "dS'T'_x_10", 'dST_y_1', 'Spectral_Entropy_10', "dS'T'_x_11", 'flourish_4', 'dRT_y_10', 'dRT_y_11', "dRS'_x_9", 'Power_Ratio_10', 'dRT_y_1', "dTT'_y_5", "dST'_y_0", 'dST_y_0', "dS'T'_x_0", 'pnn20_10', "dS'T'_x_9", "dST'_y_10", 'pnn20_9', 'Peak_Frequency_9', 'dRP_y_1']
# select_feature = ['dPT_y_11', 'dPT_y_10', 'dST_y_11', 'Power_Ratio_0', 'Spectral_Entropy_0', "dRS'_x_11", "dRS'_x_0", 'dPT_y_9', 'Power_Ratio_11', 'Power_Ratio_4', 'Power_Ratio_1', "dST'_y_11", 'Spectral_Entropy_1', 'dST_y_10', "dRS'_x_10", "dRS'_x_1", 'Power_Ratio_5', 'Spectral_Entropy_9', 'Spectral_Entropy_11', 'Spectral_Entropy_4', "dS'T'_x_10", 'dST_y_1', 'Spectral_Entropy_10', "dS'T'_x_11", 'flourish_4', 'dRT_y_10', 'dRT_y_11', "dRS'_x_9", 'Power_Ratio_10', 'dRT_y_1', 'dST_y_0', "dS'T'_x_0", "dS'T'_x_9", "dST'_y_10", 'Peak_Frequency_9', 'dRP_y_1', 'dRS_y_1', 'Spectral_Entropy_5', 'Power_Ratio_9', 'dRT_y_0']
# select_feature = ['dPT_y_11', "dTT'_y_1", 'dPT_y_10', "dTT'_y_0", 'dST_y_11', 'Power_Ratio_0', 'Spectral_Entropy_0', 'dPT_y_0', "dRS'_x_11", "dRS'_x_0", 'dPT_y_9', 'Power_Ratio_11', 'Power_Ratio_4', 'Power_Ratio_1', "dST'_y_11", 'dPT_y_1', 'Spectral_Entropy_1', 'dST_y_10', "dRS'_x_10", "dRS'_x_1", 'Power_Ratio_5', 'Spectral_Entropy_9', 'Spectral_Entropy_11', "dS'T'_y_0", 'Spectral_Entropy_4', "dS'T'_x_10", 'dST_y_1', 'Spectral_Entropy_10', "dS'T'_x_11", 'flourish_4', 'dRT_y_10', 'dRT_y_11', "dRS'_x_9", 'Power_Ratio_10', 'dRT_y_1']
# 2. 0.643 200/35 
# select_feature =["dTT'_y_10", 'Power_Ratio_0', 'dRQ_x_1', "dTT'_y_11", "dTT'_y_1", "dST'_y_11", "dRT'_y_10", 'Spectral_Entropy_0', 'Low_Frequency_Power_5', "dRS'_x_0", "dRT'_y_1", 'Power_Ratio_9', "dRS'_x_1", "dST'_y_0", 'dRS_x_11', 'Low_Frequency_Power_11', "dTT'_y_9", "dTT'_y_0", "dRS'_x_9", "dRS'_x_11", 'Power_Ratio_1', 'Spectral_Entropy_9', "dST'_y_1", 'Power_Ratio_5', 'pnn20_10', 'Power_Ratio_4', 'Low_Frequency_Power_10', 'Low_Frequency_Power_9', 'dST_y_11', 'flourish_4', "dRL'_x_1", 'dST_y_0', "dRT'_y_11", "dS'T'_x_0", 'High_Frequency_Power_1', 'Spectral_Entropy_1', "dST'_y_10", 'flourish_9', 'dRS_x_5', 'Spectral_Entropy_5', 'dRS_x_10', 'dRQ_x_11', 'Low_Frequency_Power_4', 'dPT_y_0', "dST'_y_9", 'pnn20_9', 'pnn50_9', 'pnn20_11', "dL'P'_x_0", "dL'P'_x_11"]

#select_feature = ["dTT'_x_5", "dPL'_x_0", "dPL'_x_11", "dS'T'_x_1", "dRS'_x_1", "dS'T'_x_4", "dST'_y_11", "dTT'_x_0", "dST'_y_0", "dPL'_y_5", "dTT'_x_1", "dS'T'_x_5", "dRS'_x_9", "dS'T'_x_9", 'dRQ_x_10', "dPL'_x_9", "dPL'_x_5", "dPL'_x_10", "dTT'_x_11", 'dRS_x_11', "dRS'_x_0", "dL'P'_x_11", "dST'_y_10", "dTT'_x_9", "dTT'_x_4", 'flourish_5', "dPL'_y_0", 'dRS_x_1', "dL'P'_x_10", 'dRS_x_5', "dRS'_x_10", 'dRQ_x_1', 'dRS_x_0', "dTT'_x_10", "dS'T'_x_10", 'dPQ_x_5', "dS'T'_x_0", 'dRQ_x_9', 'dRQ_x_11', "dST'_y_1", "dRS'_x_5", "dL'P'_x_1", "dPL'_y_10", 'dPQ_x_1', "dPL'_y_4", "dPL'_x_4", "dS'T'_y_0", "dTT'_y_4", "dPL'_y_11", 'dRQ_x_4']

#select_feature = ["dTT'_y_11", "dS'T'_y_0", 'Harmonic_Power_Ratio_9', "dTT'_y_1", "dTT'_y_10", 'dRQ_x_1', "dST'_y_11", "dTT'_y_0", 'dST_y_11', 'Power_Ratio_0', "dTT'_y_9", 'High_Frequency_Power_1', "dRT'_y_1", "dRS'_x_0", 'Spectral_Entropy_1', 'Low_Frequency_Power_4', "dRS'_x_1", "dRS'_x_11", 'Power_Ratio_4', 'Power_Ratio_1', 'Low_Frequency_Power_5', 'Spectral_Entropy_0', 'Spectral_Entropy_9', "dST'_y_1", 'dST_y_1', 'flourish_4', 'dPT_y_0', "dST'_y_10", 'Power_Ratio_9', "dRT'_y_11", "dST'_y_0", "dRS'_x_10", "dS'T'_x_10", "dRS'_x_9", "dS'T'_x_1", "dS'T'_x_0", 'Power_Ratio_11', 'dRS_x_1', 'Low_Frequency_Power_10', "dRT'_y_10", 'dPT_y_1', 'flourish_5', 'Low_Frequency_Power_11', 'dRS_x_5', 'dST_y_0', "dRL'_x_1", 'Power_Ratio_5', 'Peak_Frequency_9', 'dST_y_10', 'dRQ_x_11']

select_feature = ["dTT'_y_10", 'Power_Ratio_0', 'dRQ_x_1', "dTT'_y_11", "dTT'_y_1", "dST'_y_11", "dRT'_y_10", 'Spectral_Entropy_0', 'Low_Frequency_Power_5', "dRS'_x_0", "dRT'_y_1", 'Power_Ratio_9', "dRS'_x_1", "dST'_y_0", 'dRS_x_11', 'Low_Frequency_Power_11', "dTT'_y_9", "dTT'_y_0", "dRS'_x_9", "dRS'_x_11", 'Power_Ratio_1', 'Spectral_Entropy_9', "dST'_y_1", 'Power_Ratio_5', 'pnn20_10', 'Power_Ratio_4', 'Low_Frequency_Power_10', 'dST_y_11', 'flourish_4', "dRL'_x_1", "dS'T'_x_0", 'Spectral_Entropy_1', "dST'_y_10", 'dRS_x_5', 'Spectral_Entropy_5', 'Low_Frequency_Power_4', "dST'_y_9", 'pnn20_9', 'pnn50_9', "dL'P'_x_11"]

#select_feature = ["dTT'_y_10", 'Power_Ratio_0', "dTT'_y_1", "dTT'_y_11", "dST'_y_11", 'Spectral_Entropy_0', 'Low_Frequency_Power_11', "dRT'_y_1", "dST'_y_0", "dRS'_x_0", "dTT'_y_0", "dRS'_x_1", "dRS'_x_11", 'dRS_x_10', 'qrs_amplitude_9', 'Spectral_Entropy_9', 'Power_Ratio_4', 'dRS_x_11', "dTT'_y_9", 'Low_Frequency_Power_10', 'Power_Ratio_9', "dRS'_x_9", 'qrs_area_4', 'dPT_y_0', "dST'_y_1", 'dST_y_11', 'Power_Ratio_1', 'qrs_area_11', 'norm_ratio_value_11', 'Spectral_Entropy_5', 'dRS_x_5', 'norm_ratio_value_1', "dST'_y_10", "dST'_y_5", 'pnn20_9', 'norm_ratio_value_0', 'Spectral_Entropy_1', 'Power_Ratio_5', 'norm_ratio_value_9', 'Spectral_Entropy_4', "dRS'_x_10", 'flourish_5', 'dRS_x_1', "dL'P'_x_9", "dL'P'_x_11", 'pnn20_10', 'rmssd_9', "dS'T'_x_0", "dRS'_x_4", 'Power_Ratio_11', 'dST_y_0', 'rmssd_4', 'pnn50_11', 'High_Frequency_Power_1', "dS'T'_x_9", "dST'_y_9", 'Spectral_Entropy_11', 'rmssd_10', 'qrs_area_9', "dS'T'_x_11"]

pred_file_path = './tree.csv' # 存 test dataset 預測結果的檔案路徑

# 定義模型類別
class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        classes = list(set(y))
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        num_parent = [np.sum(y == c) for c in classes]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, sorted_classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(classes)
            num_right = num_parent.copy()
            for i in range(1, m):
                c = sorted_classes[i - 1]
                num_left[class_to_index[c]] += 1
                num_right[class_to_index[c]] -= 1
                gini_left = 1.0 - sum((num_left[k] / i) ** 2 for k in range(len(classes)))
                gini_right = 1.0 - sum((num_right[k] / (m - i)) ** 2 for k in range(len(classes)))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

class RandomForest:
    def __init__(self, n_trees, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.forest = []

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        args = [(X, y, self.max_depth) for _ in range(self.n_trees)]
        with Pool(cpu_count()) as pool:
            self.forest = list(tqdm(pool.imap(self._train_tree, args), total=self.n_trees))

    def _train_tree(self, args):
        X, y, max_depth = args
        indices = np.random.choice(len(X), len(X), replace=True)
        X_sample, y_sample = X[indices], y[indices]
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.forest])
        return np.squeeze(np.apply_along_axis(lambda x: np.bincount(x, minlength=self.n_classes_).argmax(), arr=tree_preds, axis=0))

if __name__ == "__main__":
    # 讀取和預處理訓練數據
    print("==============================")
    print("Start reading and processing data.")
    train = ptt.Training(select_feature)
    X_train, Y_train, X_valid, Y_valid = train.feature_label()
    print("==============================")
    print("Data processing finished.")

    # 讀取和預處理測試數據
    test = ptt.Testing(select_feature)
    testset = test.read_data()
    X_test = testset.loc[:, select_feature]
    X_test = np.array(X_test.values, dtype=float)

    rf = RandomForest(n_trees=100, max_depth=35)
    rf.fit(X_train, Y_train)

    # 預測測試數據
    y_test_pred = rf.predict(X_test)

    # 保存預測結果
    testset['pred'] = y_test_pred
    data_test_result = testset[['SubjectId', 'pred']].copy()
    data_test_result['Label'] = data_test_result['pred']
    data_test_result.drop(columns='pred', inplace=True)
    data_test_result.to_csv(pred_file_path, index=False)

    print("Predictions saved to", pred_file_path)
