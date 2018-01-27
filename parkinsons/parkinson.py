# coding: utf-8
import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import seaborn as sns
np.random.seed(sum(map(ord, "aesthetics")))
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
        
sinplot()
import matplotlib.pyplot as plt
sinplot()
plt.show(
)
sns.set_style("whitegrid")
sinplot()
plt.show(
)
get_ipython().magic(u'ls ')
df = pd.read_csv("parkinsons.data")
df
scatter_matrix(data)
scatter_matrix(df)
plt.show()
get_ipython().magic(u'ls ')
l
df
df['status']
y = df['status']
y
"""
In the dataset, there are 31 people, where 21 people have parkinsons 
disease. 

"""
df
df_drop_index = df.drop(['name'], axis=1)
df_drop_index
for i in df_drop_index.columns:
    print i
    
scatter_matrix(data)
scatter_matrix(df_drop_index)
plt.show()
from scipy.stats.stats import pearsonr
df_drop_index.describe
df_drop_index.describe()
X
get_ipython().magic(u'history ')
df_drop_index
X = df_drop_index.copy()
X
np.shape(X)
y.shape(y)
y.shape(y)
y.shape(Y)
get_ipython().magic(u'history ')
y = df['name']
y
y = df['status']
y
l
l
df_drop_index.hist()
plt.show()
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
df.plot(kind='box', sharex=False, sharey=False)
plt.show()
correlations = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
al
get_ipython().magic(u'ls ')
get_ipython().magic(u'history ')
correlations
historydef get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print(get_top_abs_correlations(df_drop_index, 3))
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print(get_top_abs_correlations(df_drop_index, 10))
df_drop_index.corr().abs()
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print(get_top_abs_correlations(df_drop_index, 10))
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print(get_top_abs_correlations(df_drop_index, 20))
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

A = get_top_abs_correlations(df_drop_index, 20)
A
A[:,0]
type(A)
for i in A:
    print i
    
for i,j,k in A:
    print i,j,k
    
    
A
get_ipython().magic(u'history ')
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

A = get_top_abs_correlations(df_drop_index, 20)
A
A[1]
A[:]
names = ['Shimmer:DDA', 'Shimmer:APQ3', 'MDVP:RAP', 'Jitter:DDP']
df_modified = df_drop_index(names)
df_modified = df_drop_index(names = names)
df_drop_index = df(names, axis=1)
df_drop_index = df(names, axis=1)
names
names = ['Shimmer:DDA', 'Shimmer:APQ3', 'MDVP:RAP', 'Jitter:DDP']
df_drop_index = df(['Shimmer:DDA', 'Shimmer:APQ3', 'MDVP:RAP', 'Jitter:DDP'], axis=1)
df_new= df_drop_index(['Shimmer:DDA', 'Shimmer:APQ3', 'MDVP:RAP', 'Jitter:DDP'], axis=1)
df_new = df_drop_index(['Shimmer:DDA', 'Shimmer:APQ3', 'MDVP:RAP', 'Jitter:DDP'])
df_drop_index
type(df_drop_index)
df_new = pd.DataFrame(df_drop_index)
df_new
df_new = pd.DataFrame(df_drop_index, columns = names)
df_new
get_ipython().magic(u'history ')
df_new
scatter_matrix(df_new)
plt.show()
scatter_matrix(df_new)
plt.show()
scatter_matrix(df_new)
df_new
X
X
X = df_new
X
y
np.shape(X)
np.shape(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size = 0.2)
X
X_train
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.predict(X_test)
y_test
from sklearn.metrics import accuracy_score
l
knn.predict(X_test)
y_predict = knn.predict(X_test)
y_predict
y_test
y_test[1]
y_test[1]
np.array(y_test)
y_test = np.array(y_test)
y_predict
accuracy_score(y_test,y_predict)
get_ipython().magic(u'ls ')
get_ipython().magic(u'pwd ')
get_ipython().magic(u'save 1-140 parkinson.py')
