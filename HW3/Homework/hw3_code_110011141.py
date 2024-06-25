#%%
# Implement 1
# Dataset Loading & splitting:
# randomly pick 60% training data、40% test data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./Real_estate.csv")
train_set = (df.sample(frac=0.6, random_state = 50)) # random_state用來固定同一組隨機數據
test_set  = df.drop(train_set.index)

#%%
# Implement 2
# Plot scatter figure including training data and testing data:
# Consider [X2 house age], [X3 distance to the nearest MRT station] and 
# [X4 number of convenience stores] corresponding to Y house price of unit area
# Blue color as training data, Red color as testing data

plt.figure(figsize = (15,5), dpi=200)
for i in range (1, 4):
    plt.subplot(1, 3, i)
    plt.scatter(train_set.iloc[:,i+1].values, train_set.iloc[:,7].values, s=10, c='b')
    plt.scatter(test_set.iloc[:,i+1].values, test_set.iloc[:,7].values, s=10, c='r')
    plt.xlabel(df.columns[i+1], fontsize=15)
    plt.ylabel(df.columns[7], fontsize=15)
    plt.title(f'X{i+1} vs Y')
    if (i==3):
        plt.legend(['Train', 'Test'], loc='center left',bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

#%%
# Implement 3
# Define loss function (Mean Square Error)
# Using basic mathematical operations in NumPy

def mse_loss(error):
    MSE = np.mean(np.square(error))
    return MSE

def error_func(predict, real):
    error = predict - real
    return error

def y_pred(X, beta):
    prediction = X.dot(beta)
    return prediction

#%%
# Implement 4

# select specify column
def column_sel(df, line):
    return (df.iloc[:, line].values).reshape(-1,1)

def beta_gen():
    beta = np.random.random_sample(size=(4,1))/1000
    #print(beta)
    return beta

def gradient_descent(X_train, X_test, y_train, y_test, lrate, iters):
    num_samples, num_features = X_train.shape
    beta = beta_gen()
    mse_history_train = []
    mse_history_test  = []
    saved_beta  = []
    sigma = np.zeros((num_features, 1))
    
    for iteration in range(iters):
        # gain Y predictions
        y_pred_train = y_pred(X_train, beta) # Predictions
        y_pred_test  = y_pred(X_test, beta)  # Predictions
        
        # Error Function: gain Y prediction - Y real
        error_train  = error_func(y_pred_train, y_train) # Error
        error_test   = error_func(y_pred_test, y_test)   # Error
        
        # Gradient function
        gradients = (2 / num_samples) * X_train.T.dot(error_train) 
        
        # sigma to prevent learning rate diverge
        sigma += gradients**2
        beta  -= lrate/sigma * gradients 
        
        # calculate loss : Mean Square Error
        mse_train = mse_loss(error_train)
        mse_test  = mse_loss(error_test)
        
        # Append MSE to list
        mse_history_train.append(mse_train)
        mse_history_test.append(mse_test)
        
        if iteration % 50 == 0:
            saved_beta.append(beta.copy())
            
    return saved_beta, mse_history_train, mse_history_test

iteration = 500
lrate     = 1000

for j in range(10):
    X1  = np.ones((len(train_set), 1))
    X2  = column_sel(train_set, 2)
    X3  = column_sel(train_set, 3)
    X4  = column_sel(train_set, 4)

    X_train = np.concatenate((np.ones((len(train_set), 1)), train_set.iloc[:, 2:5].values), axis=1)
    Y_train = (train_set.iloc[:,7].values).reshape(-1,1)
    X_test  = np.concatenate((np.ones((len(test_set), 1)), test_set.iloc[:, 2:5].values), axis=1)
    Y_test  = (test_set.iloc[:,7].values).reshape(-1,1)

    saved_beta, mse_history_train, mse_history_test = gradient_descent(X_train, X_test, Y_train, Y_test, lrate, iteration)

    print(f'Training loss {j+1}:')
    for i in range(iteration):
        if (i%50==0):
            print(f'{int(i/50)+1} epoch training loss: {mse_history_train[i]:.4f}')  
    print(f'\nTesting loss {j+1}:')
    for i in range(iteration):
        if (i%50==0):
            print(f'{int(i/50)+1} epoch testing loss: {mse_history_test[i]:.4f}')

    xxx = np.arange(0, len(mse_history_train))
    plt.figure()
    plt.plot((xxx),np.log(mse_history_train), color='red', label='training')
    plt.plot((xxx),np.log(mse_history_test), color='blue',  label='testing')
    plt.title(f'Train vs. Test loss {j+1}')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
#%%
# Implement 5
# Using least square method:
# calculate beta
def least_square(X, y):
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return betas

# calculate R^2
def R_square(y_pred, Y_test):
    mean_actual = np.mean(Y_test)
    ss_total    = np.sum((Y_test - mean_actual) ** 2)
    ss_residual = np.sum((Y_test - y_pred) ** 2)
    R2          = 1 - (ss_residual / ss_total)
    return R2

# gain beta, y_predict, R2 in one function
def beta_Ypred_R2(X, X_test, y, y_test):
    beta      = least_square(X, y)
    # train
    y_predict = y_pred(X, beta)
    error     = error_func(y_predict, y)
    MSE       = mse_loss(error)
    # test
    y_predict_1 = y_pred(X_test, beta)
    error_1     = error_func(y_predict_1, y_test)
    MSE_1       = mse_loss(error_1)
    
    R2        = R_square(y_predict, y)
    return beta, MSE, MSE_1, y_predict, R2


# prepare training data set for
# 1. Y = β0' + β1'*X2  + β2'*X3  + β3'*X4
# 2. Y = β4  + β5 *X2  + β6 *X22 + β7 *X3  + β8 *X4
# 3. Y = β9  + β10*X2  + β11*X 3 + β12*X33 + β13*X4

X1  = np.ones((len(train_set), 1))
X2  = column_sel(train_set, 2)
X3  = column_sel(train_set, 3)
X4  = column_sel(train_set, 4)
X22 = X2**2
X33 = X3**2

X_train1 = np.concatenate((X1, X2, X3, X4), axis=1)
X_train2 = np.concatenate((X1, X2, X22, X3, X4), axis=1)
X_train3 = np.concatenate((X1, X2, X3, X33, X4), axis=1)

#print(f'{X_train1.shape}, {X_train2.shape}, {X_train3.shape}')

X1_test  = np.ones((len(test_set), 1))
X2_test  = column_sel(test_set, 2)
X3_test  = column_sel(test_set, 3)
X4_test  = column_sel(test_set, 4)
X22_test = X2_test**2
X33_test = X3_test**2

X_test1 = np.concatenate((X1_test, X2_test, X3_test, X4_test), axis=1)
X_test2 = np.concatenate((X1_test, X2_test, X22_test, X3_test, X4_test), axis=1)
X_test3 = np.concatenate((X1_test, X2_test, X3_test, X33_test, X4_test), axis=1)

#print(f'{X_train1.shape}, {X_train2.shape}, {X_train3.shape}')

beta_1, Train_loss1, Test_loss1, y_pred_1, R2_1 = beta_Ypred_R2(X_train1, X_test1, Y_train, Y_test)
beta_2, Train_loss2, Test_loss2, y_pred_2, R2_2 = beta_Ypred_R2(X_train2, X_test2, Y_train, Y_test)
beta_3, Train_loss3, Test_loss3, y_pred_3, R2_3 = beta_Ypred_R2(X_train3, X_test3, Y_train, Y_test)

#y_pred_1 = y_pred(X_train1, beta_1)

print('Least Square Method\n')
print(f'Model 1:\nbetas: {beta_1.T},\nTrain loss: {Train_loss1:.4f}, Test loss: {Test_loss1:.4f}, R Square: {R2_1:.4f}\n')
print(f'Model 2:\nbetas: {beta_2.T},\nTrain loss: {Train_loss2:.4f}, Test loss: {Test_loss2:.4f}, R Square: {R2_2:.4f}\n')
print(f'Model 3:\nbetas: {beta_3.T},\nTrain loss: {Train_loss3:.4f}, Test loss: {Test_loss3:.4f}, R Square: {R2_3:.4f}')

def get_xxx(X_test, X_type):
    xxx = np.arange(0.8*min(X_test[:,X_type]), max(1.15*X_test[:,X_type]))
    #print(xxx.shape)
    return xxx

def fitting_line(xxx, yyy, X_type, X_train, Y_train, X_test, Y_test):
    plt.scatter(X_train[:,X_type], Y_train, s=10, c='b', marker='o', label='training data')
    plt.scatter(X_test[:,X_type], Y_test, s=10, c='r', marker='o', label='testing data')
    plt.plot(xxx, yyy, color='green',  label='model')
    plt.title(f'X{X_type+1} vs. Y')
    plt.xlabel(df.columns[X_type+1])
    plt.ylabel(df.columns[7])
    plt.legend()
    
mu = []
for i in range(4):
    mu_temp = np.mean(X_train[:,i])
    mu.append(mu_temp)
print(mu)

def y_func_1(xxx, X_type, beta):
    if (X_type==1):
        yyy = beta[0] + beta[1]*xxx + beta[2]*mu[2] + beta[3]*mu[3]
    elif (X_type==2):
        yyy = beta[0] + beta[1]*mu[1] + beta[2]*xxx + beta[3]*mu[3]
    elif (X_type==3):
        yyy = beta[0] + beta[1]*mu[1] + beta[2]*mu[2] + beta[3]*xxx
    return yyy

def y_func_2(xxx, X_type, beta):
    if (X_type==1):
        yyy = beta[0] + beta[1]*xxx + beta[2]*(xxx**2) + beta[3]*mu[2] + beta[4]*mu[3]
    elif (X_type==2):
        yyy = beta[0] + beta[1]*mu[1] + beta[2]*(mu[1]**2) + beta[3]*xxx + beta[4]*mu[3]
    elif (X_type==3):
        yyy = beta[0] + beta[1]*mu[1] + beta[2]*(mu[1]**2) + beta[3]*mu[2] + beta[4]*xxx
    return yyy

def y_func_3(xxx, X_type, beta):
    if (X_type==1):
        yyy = beta[0] + beta[1]*xxx + beta[2]*mu[2] + beta[3]*(mu[2]**2) + beta[4]*mu[3]
    elif (X_type==2):
        yyy = beta[0] + beta[1]*mu[1] + beta[2]*xxx + beta[3]*(xxx**2) + beta[4]*mu[3]
    elif (X_type==3):
        yyy = beta[0] + beta[1]*mu[1] + beta[2]*mu[2] + beta[3]*(mu[1]**2) + beta[4]*xxx
    return yyy

plt.figure()
xxx = get_xxx(X_test, 1)
yyy = y_func_1(xxx, 1, beta_1)
fitting_line(xxx, yyy, 1, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 2)
yyy = y_func_1(xxx, 2, beta_1)
fitting_line(xxx, yyy, 2, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 3)
yyy = y_func_1(xxx, 3, beta_1)
fitting_line(xxx, yyy, 3, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 1)
yyy = y_func_2(xxx, 1, beta_2)
fitting_line(xxx, yyy, 1, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 2)
yyy = y_func_2(xxx, 2, beta_2)
fitting_line(xxx, yyy, 2, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 3)
yyy = y_func_2(xxx, 3, beta_2)
fitting_line(xxx, yyy, 3, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 1)
yyy = y_func_3(xxx, 1, beta_3)
fitting_line(xxx, yyy, 1, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 2)
yyy = y_func_3(xxx, 2, beta_3)
fitting_line(xxx, yyy, 2, X_train, Y_train, X_test, Y_test)
plt.show()

plt.figure()
xxx = get_xxx(X_test, 3)
yyy = y_func_3(xxx, 3, beta_3)
fitting_line(xxx, yyy, 3, X_train, Y_train, X_test, Y_test)
plt.show()