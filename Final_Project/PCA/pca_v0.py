import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pca(X, variance_threshold=0.8):
    # 計算協方差矩陣
    covariance_matrix = np.cov(X, rowvar=False)
    #print(covariance_matrix)
    
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
    print(sorted_indices)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 計算累積解釋變異數比例
    explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variances)
    
    # 找到滿足累積解釋變異數比例的最小數量的特徵向量
    num_components = np.argmax(cumulative_explained_variance >= variance_threshold) + 1
    
    # 選擇前 num_components 個特徵向量
    principal_components = sorted_eigenvectors
    
    # 投影原始資料到主成分空間
    X_pca = np.dot(X, principal_components)
    
    return X_pca, principal_components, num_components, explained_variances, cumulative_explained_variance, eigenvalues, eigenvectors

# 讀取資料
file_path = 'training_features_total406.csv'  # 將這裡的路徑換成你的數據文件的路徑
data = pd.read_csv(file_path)

# 刪除包含空值的列
data.dropna(inplace=True)

# 提取特徵名稱和資料
feature_names = data.columns.tolist()
X = data.values

# 資料標準化（手動）
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# 檢查資料是否包含 NaN 或 無限大值
if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
    raise ValueError("資料包含 NaN 或無限大值，請檢查並處理這些值。")

# 執行PCA
X_pca, principal_components, num_components, explained_variance, cumulative_explained_variance, eigenvalues, eigenvectors = pca(X_normalized)

# 找到降維後的最適特徵名稱
# 先找到每個特徵在主成分上的貢獻度總和，然後排序
feature_contributions = np.sum(np.abs(principal_components), axis=1)
sorted_feature_indices = np.argsort(-feature_contributions)
sorted_feature_names = [feature_names[i] for i in sorted_feature_indices]

print("最適特徵名稱（按重要性排序）：", sorted_feature_names[:num_components], len(sorted_feature_names[:num_components]))

# 繪製累積解釋變異數比例的圖
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Number of Principal Components')
plt.grid(True)
plt.show()

