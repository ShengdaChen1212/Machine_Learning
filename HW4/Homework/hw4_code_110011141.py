import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#%% Problem 1: Dimensionality Reduction

def r_squared(y, y_pred):
    # 計算R平方值
    ss_residual = ((y - y_pred) ** 2).sum()
    ss_total = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared


'''讀取資料集'''

data = pd.read_csv("./auto-mpg.csv")

# 定義初始特徵集合
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']

model = LinearRegression()

# 創建特徵矩陣 x 和目標向量 y
x = data[features].copy()
y = pd.DataFrame(data['mpg'])
y = y.to_numpy()

# 填補缺失值
print('\nBefore filling Means:')
null = x.copy().isnull().sum()
print('Number of missing value')
print(null)

mean = np.around(np.mean(x['horsepower']),1)
x['horsepower'].fillna(mean,inplace = True)

null = x.copy().isnull().sum()
print('\nAfter filling Means:')
print('Number of missing value')
print(null, '\n')

#=================================================================================
'''High Correlation filter'''

print('================ High Correlation filter =================')

def high_correlation_filter(df, threshold=0.9):
    # 計算相關性矩陣
    corr_matrix = df.corr().abs()
    
    # 繪製相關性矩陣
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 24})  # 調整字體大小
    plt.title('Feature Correlation Matrix')
    plt.show()

    # 選擇上三角矩陣
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找到和其他特徵高度相關的特徵
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # 移除這些特徵
    df_filtered = df.drop(columns=to_drop)
    
    return df_filtered

selected_x = high_correlation_filter(x, threshold=0.9)
mse = mean_squared_error(y, model.fit(selected_x, y).predict(selected_x))

# 將原始目標變量（mpg）與降維後的特徵合併
target = data["mpg"]
final_data = pd.concat([selected_x, target], axis=1)

# 計算r_squared
HiCorr_r_squared = r_squared(y, model.fit(selected_x, y).predict(selected_x))

# 印出篩選後的特徵
print('\nOriginal features:', list(x.columns))
print('\nSelected features:', list(selected_x.columns))
print("\nFinal data (first 5 rows):")
print(final_data.head())
print('\nr_squared:', HiCorr_r_squared)
print('\nmse:', mse)
print('='*60, '\n')


#=================================================================================
'''Backward Selection'''

print('==================== Backward Selection ====================')

#設定初始比較標準
best_features = features.copy()
Round = 1

while len(best_features) > 4:
    best_mse = 100
    worst_feature = None
    print(f'\nRound {Round}:')
    for feature in best_features:
        # 暫時剔除一個特徵
        temp_features = [f for f in best_features if f != feature]
        temp_x = x[temp_features].copy()

        # 計算mse並比較效能
        mse = mean_squared_error(y, model.fit(temp_x, y).predict(temp_x))
        print('\ndrop:', feature)
        print('features remain:', temp_features)
        print('mse & best mse:', mse, best_mse)
        if mse < best_mse:
            best_mse = mse
            worst_feature = feature
    Round += 1
    # 如果找到更好的特徵組合，則更新最佳特徵
    if worst_feature:
        best_features.remove(worst_feature)
        x_new = x[best_features].copy()
        best_mse = mean_squared_error(y, model.fit(x, y).predict(x))
    else:
        break
    
# 將原始目標變量（mpg）與降維後的特徵合併
target = data["mpg"]
x_new = x[best_features].copy()
final_data = pd.concat([x_new, target], axis=1)

# 計算r_squared
SBS_r_squared = r_squared(y, model.fit(x_new, y).predict(x_new))

print("\nSelected features:", best_features)
print("\nFinal data (first 5 rows):")
print(final_data.head())
print('\nr_squared:', SBS_r_squared)
print('\nmse:', best_mse)
print('='*60, '\n')

#=================================================================================
'''PCA'''
print('=========================== PCA ============================')

# 標準化特徵
scaled_x = (x - x.mean()) / x.std()

# 計算協方差矩陣
covariance_matrix = np.cov(scaled_x.values.T)

# 計算特徵向量和特徵值
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# 排序特徵值和特徵向量
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# 計算解釋方差比例
explained_variance_ratio = [pair[0] / sum(eigenvalues) for pair in eigen_pairs]

# 找到所需的主成分數量(題目要求variance>95%)
total_variance = 0
n_components = 0
for i, ratio in enumerate(explained_variance_ratio):
    total_variance += ratio
    if total_variance >= 0.975:
        n_components = i + 1
        break

# 取得前n_components個特徵向量
selected_eigenvectors = np.array([pair[1] for pair in eigen_pairs[:n_components]])

# 將特徵向量應用於原始特徵資料
reduced_x = scaled_x.values.dot(selected_eigenvectors.T)

# 將降維後的特徵轉換為DataFrame
reduced_x = pd.DataFrame(reduced_x)

# 計算mse
model = LinearRegression()
mse = mean_squared_error(y, model.fit(reduced_x, y).predict(reduced_x))

# 將原始目標變量（mpg）與降維後的特徵合併
target = data["mpg"]
final_data = pd.concat([reduced_x, target], axis=1)

# 計算r_squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

PCA_r_squared = r_squared(y, model.fit(reduced_x, y).predict(reduced_x))

# 繪製累積解釋方差比例圖
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Principal Components')
plt.grid(True)
plt.show()

# 最終資料集包含降維後的特徵和原始目標變量
print("\nFinal data (first 5 rows):")
print(final_data.head())
print('\nr_squared:', PCA_r_squared)
print('\nmse:', mse)
print('='*60)

#%% Problem 2: clustering

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dist = ((x1-x2)**2+(y1-y2)**2)**0.5
    return dist

data = pd.read_csv('./Cluster_data.csv')
data = data[['x', 'y']]
data = data.to_numpy()
num_iterations = 100
Kx = np.arange(1, 8)
SSE = []
    
for K in Kx:
    # 隨機初始化 K 個重心
    centroids = data[np.random.choice(range(data.shape[0]), K, replace=False)]
    
    for i in range(num_iterations):
        # 歸類資料點到最近的重心
        labels = []
        for point in data[:]:
            distances = [distance(point, centroid) for centroid in centroids]
            label = np.argmin(distances)
            labels.append(label)
        labels = np.array(labels)
        
        # 更新重心
        new_centroids = []
        for k in range(K):
            mask = (labels == k)
            cluster_points = data[mask]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)
    
    
        # 檢查重心是否變化，沒變化則停止
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
    sse = 0
    for kk in range(K):
        mask = (labels == kk)
        cluster_points = data[mask]
        centroid = cluster_points.mean(axis=0)
        for cp in cluster_points:
            sse += distance(cp, centroid)
    SSE.append(sse)
        
    # 繪製散點圖
    # 指定不同簇的顏色
    colors = ['lightcoral', 'skyblue', 'yellowgreen', 'mediumpurple', 'sandybrown', 'cornflowerblue', 'plum']
    plt.figure(figsize=(8, 6), dpi=250)
    for i in range(K):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i], label=f'Cluster {i+1}', s=45)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Centroids')
    plt.title(f'K-means cluster with K = {K}', fontsize=30)
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 6), dpi=250)
plt.plot(Kx, SSE, linewidth=3)
plt.xlabel('No. of K', fontsize=15)
plt.ylabel('SSE', fontsize=15)
plt.title('SSE plot of elbow method', fontsize=20)
plt.grid(True)
    