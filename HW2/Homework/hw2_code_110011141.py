# Problem 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv("hw2_data1.csv")

class shape:
    def __init__(self):    # constructor
        self.right  = 0
        self.left   = 0
        self.top    = 0
        self.bottom = 0

# 畫出分出兩種class的scatter plot
plt.figure()
plt.scatter(data1.iloc[(data1.iloc[:,2] == 0).values, 0],
            data1.iloc[(data1.iloc[:,2] == 0).values, 1], 
            c='b', marker='o', label='class0') # outer
plt.scatter(data1.iloc[(data1.iloc[:,2] == 1).values, 0],
            data1.iloc[(data1.iloc[:,2] == 1).values, 1], 
            c='r', marker='^', label='class1') # inner
plt.xlabel('feature 1')
plt.ylabel('feature 2')

# get class 0 & class 1
class0 = data1.iloc[(data1.iloc[:,2] == 0).values, :] 
class1 = data1.iloc[(data1.iloc[:,2] == 1).values, :] 

# feature 1 : x-axis 0
# feature 2 : y-axis 1

S  = shape()
G  = shape()
G1 = shape()
G2 = shape()
C  = shape()

# get class1 top / bottom / right / left
S.right  = max(class1.iloc[:,0])
S.left   = min(class1.iloc[:,0])
S.top    = max(class1.iloc[:,1])
S.bottom = min(class1.iloc[:,1])

# plot specific hypothesis
S_x = [S.left, S.right, S.right, S.left, S.left]
S_y = [S.top, S.top, S.bottom, S.bottom, S.top]
plt.plot(S_x, S_y, "r")

# get G1
rights, lefts, tops, bottoms = [], [], [], []

# 固定左右找上下
for index, row in class0.iterrows():
    if S.left <= row["feature 1"] <= S.right:
        if row["feature 2"] > S.top:
            tops.append(row["feature 2"])
        elif row["feature 2"] < S.bottom:
            bottoms.append(row["feature 2"])

G1.top    = min(tops)
G1.bottom = max(bottoms)

# 找到上下再找左右
for index, row in class0.iterrows():
    if G1.bottom <= row["feature 2"] <= G1.top:
        if row["feature 1"] > S.right:
            rights.append(row["feature 1"])
        elif row["feature 1"] < S.left:
            lefts.append(row["feature 1"])

G1.right = min(rights)
G1.left  = max(lefts)
    
# get G2
rights, lefts, tops, bottoms = [], [], [], []

# 固定上下找左右
for index, row in class0.iterrows():
    if S.bottom <= row["feature 2"] <= S.top:
        if row["feature 1"] > S.right:
            rights.append(row["feature 1"])
        elif row["feature 1"] < S.left:
            lefts.append(row["feature 1"])

G2.right = min(rights)
G2.left  = max(lefts)

# 找到左右再找上下
for index, row in class0.iterrows():
    if G2.left <= row["feature 1"] <= G2.right:
        if row["feature 2"] > S.top:
            tops.append(row["feature 2"])
        elif row["feature 2"] < S.bottom:
            bottoms.append(row["feature 2"])

G2.top    = min(tops)
G2.bottom = max(bottoms)

# 判斷面積大小
Area1 = (G1.right - G1.left) * (G1.top - G1.bottom)
Area2 = (G2.right - G2.left) * (G2.top - G2.bottom)

# takes the larger area one
if Area1 > Area2:
    G = G1
else:
    G = G2

# plot general hypothesis
G_x = [G.left, G.right, G.right, G.left, G.left]
G_y = [G.top, G.top, G.bottom, G.bottom, G.top]
plt.plot(G_x, G_y, "g")

# 取平均
C.right  = (G.right + S.right)/2
C.left   = (G.left + S.left)/2
C.top    = (G.top + S.top)/2
C.bottom = (G.bottom + S.bottom)/2

# 畫出平均的框
C_x = [C.left, C.right, C.right, C.left, C.left]
C_y = [C.top, C.top, C.bottom, C.bottom, C.top]
plt.plot(C_x, C_y)
plt.legend()

# print("The most specific hypothesis, S:")
# print(f"Right  : {S.right}\nLeft   : {S.left}\nTop    : {S.top}\nBottom : {S.bottom}\n")
# print("The most general hypothesis, G:")
# print(f"Right  : {G.right}\nLeft   : {G.left}\nTop    : {G.top}\nBottom : {G.bottom}")

print("(a) (a < feature1 < b) AND (c < feature2 < d) :")
print(f"a = {C.left}\nb = {C.right}\nc = {C.bottom}\nd = {C.top}\n")
# (b)

a = (S.right + S.left)/2
b = (S.top + S.bottom)/2
r = (((S.right - S.left)**2 + (S.top - S.bottom)**2)**(0.5))/2

# 畫出分出兩種class的scatter plot
plt.figure()
plt.scatter(data1.iloc[(data1.iloc[:,2] == 0).values, 0],
            data1.iloc[(data1.iloc[:,2] == 0).values, 1], 
            c='b', marker='o', label='class0') # outer
plt.scatter(data1.iloc[(data1.iloc[:,2] == 1).values, 0],
            data1.iloc[(data1.iloc[:,2] == 1).values, 1], 
            c='r', marker='^', label='class1') # inner
plt.xlabel('feature 1')
plt.ylabel('feature 2')

xc = np.arange(a-r, a+r, 0.0001)
yc = (abs(r**2 - (xc-a)**2))**0.5 + b
yc2 = -yc + 2*b

plt.plot(xc,yc,color='g')  # top half 
plt.plot(xc,yc2,color='g') # bottom half
plt.legend()

print("(b) (feature1 - a)^2 + (feature2 - b)^2 = r^2 :")
print(f"a = {a}\nb = {b}\nr = {r}")


#%%
# Problem 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data2 = pd.read_csv("hw2_data2.csv")

plt.figure()
plt.scatter(data2.iloc[:,0], data2.iloc[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')

# Define MultiNormal Distribution PDF
def dmultinorm(x, mu, cov):
    exponent = np.exp(-1/2*(x-mu).dot(np.linalg.inv(cov)).dot((x-mu).T))
    D = 1/np.sqrt((2 * np.pi)**len(x) * np.linalg.det(cov)) * exponent
    return D

mu_1  = np.array([1, 1])  # class 1 , feature's mean
mu_2  = np.array([3, 2])  # class 2 , feature's mean
cov_1 = np.array([[1, 0.3], [0.3, 1]])
cov_2 = np.array([[1, 0.15], [0.15, 1]])

L1    = np.array([[0, 1], [1, 0]])
L2    = np.array([[0, 0.1], [0.9, 0]])
L3    = np.array([[0, 0.02], [0.98, 0]])
L4    = np.array([[0, 0.9], [0.1, 0]])
L5    = np.array([[0, 0.98], [0.02, 0]])

threshold = 0.001
select_feature = ["x1", "x2"]

# create empty dataframe with column names & index numbers
risks = pd.DataFrame(columns=['risk1','risk2','risk3','risk4','risk5'], index=range(0,100))
probability = pd.DataFrame()
probability_1, probability_2 = [], []

# total in index shape[0] = 100, from 0 to 99
for index in range(risks.shape[0]):
    # x = [x1, x2]
    x = np.array([data2.iloc[index,0], data2.iloc[index,1]])
    
    # get p(x|Ci)
    pr_1 = dmultinorm(x, mu_1, cov_1)
    pr_2 = dmultinorm(x, mu_2, cov_2)
    
    # get value from array using .item()
    pr_all = np.array([[pr_1.item(), pr_2.item()]])
    probability_1.append(pr_1.item())
    probability_2.append(pr_2.item())
    
    # using axis=0 makes the array won't concat to one array
    risk = L1.dot(pr_all.T).T
    risk = np.append(risk, L2.dot(pr_all.T).T, axis=0)
    risk = np.append(risk, L3.dot(pr_all.T).T, axis=0)
    risk = np.append(risk, L4.dot(pr_all.T).T, axis=0)
    risk = np.append(risk, L5.dot(pr_all.T).T, axis=0)

    # determined which class
    for column in range(0, 5):
        if (risk[column, 0].item() > threshold) & (risk[column, 1].item() > threshold):
            risks.iloc[index, column] = "undetermined"
        elif (risk[column, 0].item() > risk[column, 1].item()):
            risks.iloc[index, column] = "class2"
        elif (risk[column, 0].item() < risk[column, 1].item()):
            risks.iloc[index, column] = "class1"

probability.loc[:, "probability 1"] = probability_1
probability.loc[:, "probability 2"] = probability_2

dataset = pd.concat([data2, probability, risks], axis=1)

plt.figure(figsize = (20, 20), dpi=200)
for pic in range(4, 9):
    plt.subplot(3, 2, pic-3)
    plt.scatter(dataset.iloc[(dataset.iloc[:, pic] == 'undetermined').values, 0], 
                dataset.iloc[(dataset.iloc[:, pic] == 'undetermined').values, 1],
                s=55, c='g', marker='o', label='undetermined')
    plt.scatter(dataset.iloc[(dataset.iloc[:, pic] == 'class1').values, 0], 
                dataset.iloc[(dataset.iloc[:, pic] == 'class1').values, 1],
                s=55, c='c', marker='s', label='class1')
    plt.scatter(dataset.iloc[(dataset.iloc[:, pic] == 'class2').values, 0], 
                dataset.iloc[(dataset.iloc[:, pic] == 'class2').values, 1],
                s=55, c='r', marker='^', label='class2')
    plt.xlabel('x1', fontsize = 18)
    plt.ylabel('x2', fontsize = 18)
    plt.title(f'Lambda{pic-3}', fontsize = 18)
    plt.legend(fontsize = 12)


#%% 
# Problem 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

college  = pd.read_csv('College.csv')
college2 = pd.read_csv('College.csv', index_col=0)
college3 = college.rename ({'Unnamed: 0': 'College'}, axis=1)
college3 = college3.set_index('College')
college  = college3

# check for different data types
college.dtypes

# (a)
college.describe()

# (b)
# Visualization - Scatter Plot
scatter_mat = college.loc[:, ["Top10perc", "Apps", "Enroll"]]

plt.figure(figsize = (12,10), dpi=200)
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,i*3+j+1)
        if i!=j:
            plt.scatter(scatter_mat.iloc[:,j], scatter_mat.iloc[:,i], s=5)
        if i==j:
            x_loc = np.mean([np.min(scatter_mat.iloc[:,j]), np.max(scatter_mat.iloc[:,j])])
            y_loc = np.mean([np.min(scatter_mat.iloc[:,i]), np.max(scatter_mat.iloc[:,i])])
            plt.scatter(scatter_mat.iloc[:,j], scatter_mat.iloc[:,i], s=0)
            plt.text(y_loc, x_loc, scatter_mat.columns[i], horizontalalignment='center', verticalalignment='center',fontsize=11, fontweight='bold')
        if j!=0:
            plt.yticks([])
        if i!=2:
            plt.xticks([])
plt.tight_layout()
plt.show()

# (c)

# Add new label
college['Elite'] = ''
#display(college.head(n=10))
college.loc[college.loc[:,'Top10perc']>=50, 'Elite'] = '1'
college.loc[~(college.loc[:,'Top10perc']>=50), 'Elite'] = '0'
#display(college.head(n=10))

elites = np.sum((college['Elite'] == '1'))
print(f'Total elite university : {elites}')

# Box Plot
# Outstate vs. Elite

plt.figure(figsize = (10,6), dpi=200)
plt.boxplot(college.iloc[(college.iloc[:, 18] == '0').values, 8], positions=[0])
plt.boxplot(college.iloc[(college.iloc[:, 18] == '1').values, 8], positions=[1])
plt.xticks(range(2),['0', '1'])
plt.title(college.columns[8], fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#(d)
# plot Histogram
plt.figure(figsize = (10,15), dpi=200)
for i in range(17):
    plt.subplot(5,4,i+1)
    plt.hist(college.iloc[(college.iloc[:, 18] == '0').values, i], 
             bins=20, rwidth=1, edgecolor='black', linewidth=0.4)
    plt.hist(college.iloc[(college.iloc[:, 18] == '1').values, i], 
             bins=20, rwidth=1, edgecolor='black', linewidth=0.4)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title(college.columns[i], fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# (e)
# randomly pick 70% training data、30% test data
train = college.sample(frac=0.7, random_state = 30) # random_state用來固定其為同一組隨機數據
test  = college.drop(train.index)
#display(train)
#display(test)

for i in range(1, 3):
    plt.subplot(1, 2, i)
    if (i==1):
        plt.hist(train.iloc[(train.iloc[:, 18] == '0').values, 18])
        plt.hist(train.iloc[(train.iloc[:, 18] == '1').values, 18])
        plt.title('Training Set', fontsize=14, fontweight='bold')
    elif (i==2):
        plt.hist(test.iloc[(test.iloc[:, 18] == '0').values, 18])
        plt.hist(test.iloc[(test.iloc[:, 18] == '1').values, 18])
        plt.title('Test Set', fontsize=14, fontweight='bold')
    plt.xticks(range(2),['0', '1'])
    plt.yticks(fontsize=8)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.show()

# (f)
# fit Naïve Bayes classifier on the training set

# find mean, variance, count
select_feature = ["Top25perc", "Outstate", "Books", "PhD", "Elite"]
mu             = train[select_feature].groupby("Elite").mean()
sd             = train[select_feature].groupby("Elite").std()
count          = train[select_feature].groupby("Elite").count().iloc[:,0]. rename('counts')

'''
print("Mean:\n")
display(mu)
print("Standard deivation:\n")
display(sd)
print("Counts:\n")
display(count)
'''

def dnorm(x, mu, sd):
    exponent = np.exp(-(x - mu)**2 / (2 * sd**2))
    return 1/np.sqrt(2 * np.pi * sd**2) * exponent

prior = count / count.sum()

pr_high_all = []
pr_low_all = []
pred_label = []

# 判斷class0 & class1的機率，然後比大小
for num in range(test.shape[0]):
    data    = test[select_feature].iloc[num,:].drop('Elite')
    pr_high = 1
    pr_low  = 1
    for col in range(data.shape[0]):
        pr_high = pr_high * dnorm(data.iloc[col], mu.iloc[1].iloc[col], sd.iloc[1].iloc[col])
        pr_low  = pr_low  * dnorm(data.iloc[col], mu.iloc[0].iloc[col], sd.iloc[0].iloc[col])
    pr_high  = pr_high * prior['1']
    pr_low   = pr_low  * prior['0']
    evidence = pr_high + pr_low
    pr_high  = pr_high / evidence
    pr_low   = pr_low / evidence
    pr_high_all.append(pr_high)
    pr_low_all.append(pr_low)
    if pr_high > pr_low:
        pred_label.append('1')
    else:
        pred_label.append('0')

from sklearn.metrics import confusion_matrix
accuracy = np.sum((test['Elite'] == pred_label))/len(test)
conf_mat = confusion_matrix(test['Elite'], pred_label, labels=["1", "0"])
print("Accuracy of Normal distribution: ", round(accuracy,3))
print("Confusion matrix of Normal distribution: \n", conf_mat)

# (g)
def lognormal_pdf(x, mu, sigma):
    """Lognormal distribution pdf."""
    return (1 / (x * sigma * np.sqrt(2*np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2*sigma**2))

log_prior = count / count.sum()

log_pr_high_all = []
log_pr_low_all = []
log_pred_label = []

for num in range(test.shape[0]):
    data    = test[select_feature].iloc[num,:].drop('Elite')
    log_pr_high = 0
    log_pr_low  = 0
    for col in range(data.shape[0]):
        log_pr_high = log_pr_high + lognormal_pdf(data.iloc[col], mu.iloc[1].iloc[col], sd.iloc[1].iloc[col])
        #print(log_pr_high)
        log_pr_low  = log_pr_low  + lognormal_pdf(data.iloc[col], mu.iloc[0].iloc[col], sd.iloc[0].iloc[col])
        #print(log_pr_low)
    #log_pr_high  = log_pr_high + log_prior['1']
    #log_pr_low   = log_pr_low  + log_prior['0']
    #log_evidence = log_pr_high + log_pr_low
    #log_pr_high  = log_pr_high - log_evidence
    #log_pr_low   = log_pr_low  - log_evidence
    log_pr_high_all.append(log_pr_high)
    log_pr_low_all.append(log_pr_low)
    if log_pr_high > log_pr_low:
        log_pred_label.append('1')
    else:
        log_pred_label.append('0')

log_accuracy = np.sum((test['Elite'] == log_pred_label))/len(test)
log_conf_mat = confusion_matrix(test['Elite'], log_pred_label, labels=["1", "0"])
print("Accuracy of Log-Normal distribution: ", round(log_accuracy,3))
print("Confusion matrix of Log-Normal distribution: \n", log_conf_mat)
