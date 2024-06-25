# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:25:59 2024

@author: ess
"""

#%% import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

#%%
# =============================================================================
# preprocessing & EDA
# =============================================================================

# visualize on iris
iris = datasets.load_iris()
x = iris.data
y = iris.target
feature_names = iris.feature_names

n = 8 # Nan number
[xidx, yidx] = [np.random.choice(x.shape[0],n),np.random.choice(x.shape[1],n)] # Nan index
for i in range(n) :
    x[xidx[i],yidx[i]] = np.nan
df = pd.DataFrame(x, columns=feature_names)
df.head()
#%%
### missing value
null = df.isnull().sum()
print('number of missing value')
print(null)
# fill NaN with mean
for j in df.columns:
    mean = np.around(np.mean(df[j]),3)
    df[j].fillna(mean,inplace = True)
    print('fill Nan in %s with mean %.3f (cm)'%(j,mean))
df.head()
X = df.to_numpy()

#%% 
# =============================================================================
# PCA
# =============================================================================
from sklearn.decomposition import PCA

pca_iris = PCA().fit(X)
plt.figure()
plt.plot(np.arange(1,5), np.cumsum(pca_iris.explained_variance_ratio_), marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
#%%
# set n_component = 3
pca_iris3 = PCA(n_components = 3)
pca_iris3_feature = pca_iris3.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('PCA(component = 3)')
ax.scatter(pca_iris3_feature[:,0],
           pca_iris3_feature[:,1],
           pca_iris3_feature[:,2],
           c=iris.target)
plt.show()
#%%
# set n_component = 2
pca_iris2 = PCA(n_components = 2)
pca_iris2_feature = pca_iris2.fit_transform(X)
plt.figure()
plt.title('PCA(component = 2)')
plt.scatter(pca_iris2_feature[:,0], pca_iris2_feature[:,1], c=iris.target)
plt.show()

#%%
# =============================================================================
# LDA
# =============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda_iris = lda.fit(X, y)
print(lda.explained_variance_ratio_)

# set n_component = 2
lda2 = LinearDiscriminantAnalysis(n_components=2)
lda2_iris = lda2.fit_transform(X, y)
plt.figure()
plt.title('LDA(component = 2)')
plt.scatter(lda2_iris[:,0], lda2_iris[:,1], c=iris.target)
plt.show()
#%%
# set n_component = 1
lda1 = LinearDiscriminantAnalysis(n_components=1)
lda1_iris = lda1.fit_transform(X, y)
plt.figure()
plt.title('LDA(component = 1)')
plt.scatter(lda1_iris[:,0], np.zeros([1,150]), c=iris.target)
plt.show()
#%%
plt.figure(figsize = (15,9))
plt.title('LDA(component = 1)')
plt.scatter(np.arange(0,150), lda1_iris[:,0], c=iris.target)
plt.bar(np.arange(0,150), lda1_iris[:,0], width = 0.2, fc = 'k')
plt.show()

#%%
# =============================================================================
# K-means
# =============================================================================
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
   model = KMeans(n_clusters=i, n_init=10, init="k-means++")
   model.fit(X)
   wcss.append(model.inertia_)
plt.figure(figsize=(10,10))
plt.plot(range(1,11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

estimators = [('k_means_iris_4', KMeans(n_clusters=4, n_init=10, init="k-means++")),
              ('k_means_iris_3', KMeans(n_clusters=3, n_init=10, init="k-means++")),
              ('k_means_iris_2', KMeans(n_clusters=2, n_init=10, init="k-means++"))]

fignum = 1
titles = ['4 clusters', '3 clusters', '2 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(14,14))
    ax = fig.add_subplot(111, projection='3d')
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 2], X[:, 3], c=labels.astype(float), edgecolor='k', s=100)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel(feature_names[0],fontsize=15)
    ax.set_ylabel(feature_names[2],fontsize=15)
    ax.set_zlabel(feature_names[3],fontsize=15)
    ax.set_title(titles[fignum - 1],fontsize=30)
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(14,14))
ax = fig.add_subplot(111, projection='3d')

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 2].mean(),
              X[y == label, 3].mean(),
              name, horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'),fontsize=15)
# Reorder the labels to have colors matching the cluster results
yy = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 2], X[:, 3], c=yy, edgecolor='k', s=100)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_xlabel(feature_names[0],fontsize=15)
ax.set_ylabel(feature_names[2],fontsize=15)
ax.set_zlabel(feature_names[3],fontsize=15)
ax.set_title('Ground Truth',fontsize=30)
plt.show()