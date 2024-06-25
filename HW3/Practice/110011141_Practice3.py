import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load data
pd.set_option("display.float_format", lambda x: "%.2f" % x)
df = pd.read_csv("advertising.csv")

X = df[["TV", "radio", "newspaper"]]
y = df[["sales"]]

# 工具包
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # add Intercept term
model_time1 = sm.OLS(y_train, X_train)
results_time1 = model_time1.fit()
print(results_time1.summary())

# My function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # add Intercept term
coefficients = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train) # y_hat = b0 + b1*TV
print("Intercept:", coefficients[0])
print(f"Coefficient for TV: beta1 = {coefficients[1]}, beta2 = {coefficients[2]}, beta3 = {coefficients[3]}")

'''
以上兩段 code 分別為使用套件與手刻函式去尋找 coefficient，
從 print 出來的結果可以看到兩者算出來的結果是相同的，誤差原因
是套件將小數後面的位數四捨五入，因此將手刻算出來的值四捨五入也
可以得到同樣的值，然後從執行的結果觀察發現 radio 的 beta 值
最大，我認為這代表在 TV, radio, newspaper 三者當中 radio 對
sales的影響最大，而相較之下 newspaper 的 beta 最小，也代表他對
sales的影響最小。
'''