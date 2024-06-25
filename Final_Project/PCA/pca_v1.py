import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCAAnalysis:
    def __init__(self, variance_threshold):
        self.variance_threshold = variance_threshold
        self.mean = None
        self.std = None
        self.principal_components = None
        self.num_components = None
        self.explained_variances = None
        self.cumulative_explained_variance = None

    def fit(self, X):
        # 標準化資料
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = (X - self.mean) / self.std

        # 計算協方差矩陣
        covariance_matrix = np.cov(X_normalized, rowvar=False)

        # 檢查協方差矩陣是否為對稱矩陣
        if not np.allclose(covariance_matrix, covariance_matrix.T, atol=1e-8):
            raise ValueError("協方差矩陣不是對稱的。請檢查輸入資料。")

        # 計算特徵值和特徵向量
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"特徵值計算失敗：{e}")

        # 將特徵值從小到大排序，並同時對應排序特徵向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 計算累積解釋變異數比例
        explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        cumulative_explained_variance = np.cumsum(explained_variances)

        # 找到滿足累積解釋變異數比例的最小數量的特徵向量
        num_components = np.argmax(cumulative_explained_variance >= self.variance_threshold) + 1

        # 保存結果
        self.principal_components = sorted_eigenvectors
        self.num_components = num_components
        self.explained_variances = explained_variances
        self.cumulative_explained_variance = cumulative_explained_variance

        return self

    def transform(self, X):
        if self.principal_components is None:
            raise ValueError("PCAAnalysis is not fitted yet. Call 'fit' with appropriate data.")
        
        # 標準化資料
        X_normalized = (X - self.mean) / self.std

        # 投影原始資料到主成分空間
        X_pca = np.dot(X_normalized, self.principal_components)
        return X_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def plot_cumulative_explained_variance(self):
        if self.cumulative_explained_variance is None:
            raise ValueError("PCAAnalysis is not fitted yet. Call 'fit' with appropriate data.")

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.cumulative_explained_variance) + 1), self.cumulative_explained_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by Number of Principal Components')
        plt.grid(True)
        plt.show()

    def get_important_features(self, feature_names):
        if self.principal_components is None:
            raise ValueError("PCAAnalysis is not fitted yet. Call 'fit' with appropriate data.")

        feature_contributions = np.sum(np.abs(self.principal_components), axis=1)
        sorted_feature_indices = np.argsort(-feature_contributions)
        sorted_feature_names = [feature_names[i] for i in sorted_feature_indices]
        
        return sorted_feature_names[:self.num_components]

# 使用示例
file_path = 'training_features_1.csv'  # 將這裡的路徑換成你的數據文件的路徑
data = pd.read_csv(file_path)

# 刪除包含空值的列
data.dropna(inplace=True)

# 提取特徵名稱和資料
feature_names = data.columns.tolist()
X = data.values

# 創建PCAAnalysis實例並進行擬合和轉換
pca_analysis = PCAAnalysis(variance_threshold=0.8)
X_pca = pca_analysis.fit_transform(X)

# 繪製累積解釋變異數比例的圖
pca_analysis.plot_cumulative_explained_variance()

# 獲取最重要的特徵名稱
important_features = pca_analysis.get_important_features(feature_names)
print("最適特徵名稱（按重要性排序）：", important_features, len(important_features))
