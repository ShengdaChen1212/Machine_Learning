#%% import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
#%%
df = pd.read_csv("auto-mpg.csv")
#%% 填補缺失值
mean = np.around(np.mean(df['horsepower']),1)
df['horsepower'].fillna(mean,inplace = True)
#%% 觀察 correlation
corrM = df.iloc[:,0:7].corr()
print(round(corrM,3))
plt.figure(figsize = (6,4), dpi=200)
plt.imshow(corrM)
plt.colorbar()
plt.clim([-1,1])
plt.xticks(np.arange(7), corrM.columns[:], rotation='vertical')
plt.yticks(np.arange(7), corrM.columns[:])
plt.show()
#%% 觀察 weighting factor
X = df[["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]]
y = df[["mpg"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # add Intercept term
model_time1 = sm.OLS(y_train, X_train)
results_time1 = model_time1.fit()
print(results_time1.summary())
#%% 
'''
從 correlation 觀察發現 "cylinders", "displacement", "horsepower", "weight" 
他們彼此間對 mpg 的 correlation 很相近，加上從取 6 個 feature 的 linear 
regression 觀察發現 "displacement", "horsepower" 最接近 0，因此我捨棄這兩個
feature 去選擇剩下 4 個 feature。
'''

X = df[["cylinders", "weight", "acceleration", "model year"]]
y = df[["mpg"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

#Test MSE
y_pred = reg_model.predict(X_test)
print(f"Test MSE: {mean_squared_error(y_test, y_pred)}")