import pandas as pd

# 讀取數據
training_features_df = pd.read_csv('./training_features.csv')
labels_df = pd.read_csv('./Labels.csv')

# Map labels to 0, 1, 2, 3
label_mapping = {'NORM': 0, 'STTC': 1, 'CD': 2, 'MI': 3}
labels_df['Label'] = labels_df['Label'].map(label_mapping)

# 指定要選擇的特徵
select_features = ["dS'T'_y_0", "dST'_y_0", "dST'_y_1", "dST'_y_11", "dST'_y_10", "dS'T'_y_11", 'dST_y_0', 'dST_y_1', 'dST_y_10', "dL'Q_y_1", 'dPT_y_0', "dS'T'_y_10", 'dPT_y_1', 'dRQ_x_0', 'dST_y_11', "dL'P'_y_1", 'dPQ_y_1', "dPL'_x_1", "dS'T'_y_1", 'dRQ_x_1', "dL'Q_y_0", 'dPT_y_10', 'dRT_y_10', 'dRT_y_11', 'dPQ_y_10']

# 選擇特徵
filtered_training_features_df = training_features_df[select_features]

# 結合特徵和標籤
combined_df = filtered_training_features_df.copy()
combined_df['Label'] = labels_df['Label']

# 分離特徵和標籤
X = combined_df[select_features].values
y = combined_df['Label'].values

# 打印檢查數據
print(combined_df.head())
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

#%%
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
                if (thresholds[i] == thresholds[i - 1]):
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

class HierarchicalRandomForest:
    def __init__(self, n_trees_binary, n_trees_multi, max_depth=None):
        self.n_trees_binary = n_trees_binary
        self.n_trees_multi = n_trees_multi
        self.max_depth = max_depth
        self.binary_forest = []
        self.multi_forest = []

    def fit(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        print(f"Data distribution: {dict(zip(unique, counts))}")

        X_norm = X[y == 0]
        y_norm = y[y == 0]
        X_non_norm = X[y != 0]
        y_non_norm = y[y != 0]

        # 確保數據平衡
        min_samples = min(len(X_norm), len(X_non_norm))
        X_norm = X_norm[:min_samples]
        y_norm = y_norm[:min_samples]
        X_non_norm = X_non_norm[:min_samples]
        y_non_norm = y_non_norm[:min_samples]

        for _ in range(self.n_trees_binary):
            X_sample = np.vstack((X_norm, X_non_norm))
            y_sample = np.hstack((y_norm, y_non_norm))
            indices = np.random.choice(len(X_sample), len(X_sample), replace=True)
            X_sample, y_sample = X_sample[indices], y_sample[indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.binary_forest.append(tree)

        for _ in range(self.n_trees_multi):
            indices = np.random.choice(len(X_non_norm), len(X_non_norm), replace=True)
            X_sample, y_sample = X_non_norm[indices], y_non_norm[indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.multi_forest.append(tree)

    def predict(self, X):
        binary_preds = np.array([tree.predict(X) for tree in self.binary_forest])
        binary_majority = np.squeeze(np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), arr=binary_preds, axis=0))

        print(f"binary_majority: {binary_majority}")

        multi_preds = np.zeros_like(binary_majority)
        non_norm_indices = binary_majority == 1
        if np.any(non_norm_indices):
            multi_preds_non_norm = np.array([tree.predict(X[non_norm_indices]) for tree in self.multi_forest])
            print(f"multi_preds_non_norm: {multi_preds_non_norm}")
            print(f"self.multi_forest[0].n_classes_: {self.multi_forest[0].n_classes_}")
            multi_majority = np.squeeze(np.apply_along_axis(lambda x: np.bincount(x, minlength=self.multi_forest[0].n_classes_).argmax(), arr=multi_preds_non_norm, axis=0))
            multi_preds[non_norm_indices] = multi_majority

        return multi_preds

# 訓練和預測
hr = HierarchicalRandomForest(n_trees_binary=200, n_trees_multi=200, max_depth=25)
hr.fit(X, y)
y_pred = hr.predict(X)

accuracy = np.mean(y_pred == y)
accuracy
