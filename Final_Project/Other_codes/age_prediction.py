import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df  = pd.read_csv("ML_train.csv")

### Check NaN
df.isna().sum(axis=0)


# add new label
# 1. 0 for NORM, 1 for STTC, 2 for CD, 3 for MI
# 2. 0 for NORM, 1 for not NORM

df['Label_num'] = ''
df.loc[df.loc[:,'Label'] == 'NORM', 'Label_num']    = '0'
df.loc[~(df.loc[:,'Label'] == 'NORM'), 'Label_num'] = '1'
'''
df.loc[df.loc[:,'Label'] == 'STTC', 'Label_num'] = 1
df.loc[df.loc[:,'Label'] == 'CD',   'Label_num'] = 2
df.loc[df.loc[:,'Label'] == 'MI',   'Label_num'] = 3
'''

# 分data
train_set = df.sample(frac=0.7, random_state = 30) # random_state用來固定其為同一組隨機數據
train_set = train_set.sort_values('SubjectId')     # sort values by SubjectId
test_set  = df.drop(train_set.index) 
train_set


'''
plt.figure(figsize = (20, 20), dpi=200)
plt.scatter(df.iloc[(df.loc[:, "Label"] == 'NORM').values, 0], 
            df.iloc[(df.loc[:, "Label"] == 'NORM').values, 1],
            label='NORM')
plt.scatter(df.iloc[(df.loc[:, "Label"] == 'MI').values, 0], 
            df.iloc[(df.loc[:, "Label"] == 'MI').values, 1],
            label='MI')
plt.scatter(df.iloc[(df.loc[:, "Label"] == 'STTC').values, 0], 
            df.iloc[(df.loc[:, "Label"] == 'STTC').values, 1],
            label='STTC')
plt.scatter(df.iloc[(df.loc[:, "Label"] == 'CD').values, 0], 
            df.iloc[(df.loc[:, "Label"] == 'CD').values, 1],
            label='CD')
plt.xlabel('Data', fontsize = 25)
plt.ylabel('Age', fontsize = 25)
plt.title('age vs. Data', fontsize = 18)
plt.legend(fontsize = 18)
plt.show()
'''


# 放棄超過100歲的data
df_draw = df.drop(df[df['age']>100].index)

# plot histogram
plt.figure()
plt.hist(df.iloc[(df.loc[:, "Label_num"] == '0').values, 1], bins=20, rwidth=1, edgecolor='black', linewidth=0.4)
plt.hist(df.iloc[(df.loc[:, "Label_num"] == '1').values, 1], bins=20, rwidth=1, edgecolor='black', linewidth=0.4)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Value', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.show()

# 看分布圖
plt.figure(figsize = (20, 20), dpi=200)
plt.scatter(df_draw.iloc[(df_draw.loc[:, "Label"] == 'NORM').values, 0], 
            df_draw.iloc[(df_draw.loc[:, "Label"] == 'NORM').values, 1],
            label='NORM')
plt.scatter(df_draw.iloc[(df_draw.loc[:, "Label"] == 'MI').values, 0], 
            df_draw.iloc[(df_draw.loc[:, "Label"] == 'MI').values, 1],
            label='MI')
plt.scatter(df_draw.iloc[(df_draw.loc[:, "Label"] == 'STTC').values, 0], 
            df_draw.iloc[(df_draw.loc[:, "Label"] == 'STTC').values, 1],
            label='STTC')
plt.scatter(df_draw.iloc[(df_draw.loc[:, "Label"] == 'CD').values, 0], 
            df_draw.iloc[(df_draw.loc[:, "Label"] == 'CD').values, 1],
            label='CD')
plt.xlabel('Data', fontsize = 25)
plt.ylabel('Age', fontsize = 25)
plt.title('age vs. Data', fontsize = 18)
plt.legend(fontsize = 18)
plt.show()

train_set_1 = df_draw.sample(frac=0.7, random_state = 30) # random_state用來固定其為同一組隨機數據
train_set_1 = train_set_1.sort_values('SubjectId')     # sort values by SubjectId
test_set_1  = df_draw.drop(train_set_1.index) 
#display(test_set_1)

select_feature = ["age", "Label_num"]
mu_1           = train_set_1[select_feature].groupby("Label_num").mean()
sd_1           = train_set_1[select_feature].groupby("Label_num").std()
count_1        = train_set_1[select_feature].groupby("Label_num").count().iloc[:,0]. rename('counts')

print("Mean:\n")
#display(mu_1)
print("Standard deivation:\n")
#display(sd_1)
print("Counts:\n")
#display(count_1)
print(mu_1.iloc[0])

def dnorm(x, mu, sd):
    exponent = np.exp(-(x - mu)**2 / (2 * sd**2))
    return 1/np.sqrt(2 * np.pi * sd**2) * exponent

def disc_func(data, mu, sd, prior):
    pr_0 = dnorm(data, mu.iat[0, 0], sd.iat[0, 0])
    pr_0 = pr_0 * prior['0']
    #display(type(pr_0))
    pr_1 = dnorm(data, mu.iat[1, 0], sd.iat[1, 0])
    pr_1 = pr_1 * prior['1']
    #print(pr_1)
    return pr_0, pr_1
    
prior = count_1 / count_1.sum()
print(prior)
print(type(prior))

pr_0_all   = []
pr_1_all   = []
pred_label = []

for num in range(test_set_1.shape[0]):
    data       = int(test_set_1.iat[num, 1])
    #print(data)
    pr_0, pr_1 = disc_func(data, mu_1, sd_1, prior)
    pr_0_all.append(pr_0)
    pr_1_all.append(pr_1)
    if (pr_0 > pr_1):
        pred_label.append('0')
    else:
        pred_label.append('1')


from sklearn.metrics import confusion_matrix
accuracy = np.sum((test_set_1['Label_num'] == pred_label))/len(test_set_1)
conf_mat = confusion_matrix(test_set_1['Label_num'], pred_label, labels=["1", "0"])
print("Accuracy of Normal distribution: ", round(accuracy,3))
print("Confusion matrix of Normal distribution: \n", conf_mat)
