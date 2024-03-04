''' I attempted a popularity classification of the shares, but the model accuracy wasn't that awesome. In this notebook, I will attempt to 
improve that previous model accuracy, and also attempt to do some other classfication challenges.
There will emphasize on feature selection algorthims that can be used in modelling the system for better results. Also, aside using Accuracy
as the evulation metric, I will also consider some other metric evaluation.
The process followed is highlighted below:
Data Cleaning - Noise removal
Data Transformation - Transform using log-transformation
Data Clustering - Grouping Similar Articles together.
Feature Selection and Evaluation
Machine Learning Classification
Summary and Conclusion.  '''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Libaries import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from sklearn.manifold import TSNE
''' 
TSNE is an abbreviation for t-distributed Stochastic Neighbor Embedding. TSNE is a dimensionality reduction technique that aims to 
capture the local relationships between data points in a lower-dimensional space. It is often used for visualizing high-dimensional data 
in a way that reveals underlying structures and patterns. 
'''

data = pd.read_csv("OnlineNewsPopularity.csv")
data.head(n=4)

# Here we drop the two non-preditive (url and timedelta) attributes. They won't contribute anything
data.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
data.head(n=4)
# remove noise from n_tokens_content. those equals to 0
data  = data[data[' n_tokens_content'] != 0]
# Comment - Visualizing the n_non_stop_words data field shows that the present of a record with 1042 value, 
# futher observation of that data shows that it belongs to entertainment which is not actually. It belongs to world news or others.
# this particluar also contains 0 on a lot of attributes. This record is classifed as a noise and will be remove.
data = data[data[' n_non_stop_words'] != 1042]
# Here, we will go ahead and drop the field of ' n_non_stop_words. It doesn't contain relaible information.
data.drop(labels=[' n_non_stop_words'], axis = 1, inplace=True)

original_data = copy(data)

# describing the data
data.describe()
# from the data, there will be need to normailze the data if their will be need for condersing any machine learning model.

share_data = np.sort(data[' shares'].values)      # sort the values in ascending order
print(share_data.shape)
leng = share_data.shape[0]
middle = share_data[int(leng*0.7)-1]
middle

# very good shares
'''
share_data = np.sort(data[' shares'].values)
leng = share_data.shape[0]
top_70 = share_data[int(leng*0.80)-1]
print (top_70)
top_50 = share_data[int(leng*0.40)-1]
print (top_50)
#temp_data = data[(data[' shares'] >= top_70)]
temp_data = data[(data[' shares'] >= top_50) & (data[' shares'] < top_70)]
print(temp_data.shape)
'''

# create label grades for the classes
'''
Very good = 7746 # top 80%
Good = 7785 # top 60 - top 80
Average = 8585 # 40% - 60%
Poor = 14346 # less than 40%
'''
share_label = list()
for share in data[' shares']:
    if share <= 1400:
        share_label.append('Unpopular')
    else:
        share_label.append('Popular')
# Update this class label into the dataframe
data = pd.concat([data.reset_index(drop=True), pd.DataFrame(share_label, columns=['popularity'])], axis=1)
data.head(4)

# Evaluating features (sensors) contribution towards the label
fig = plt.figure(figsize=(5,4))
ax = sns.countplot(x='popularity',data=data,alpha=0.5)
data_channel_data = data.groupby('popularity').size().reset_index()
data_channel_data.columns = ['popularity','No of articles']
data_channel_data

# Normal distribution analysis for Shares
print("Skewness: %f" % data[' shares'].skew())
print("Kurtosis: %f" % data[' shares'].kurt())
from scipy.stats import norm, probplot
#histogram and normal probability plot
temp_data = data[data[' shares'] <= 100000]
fig,ax = plt.subplots(figsize=(7,7))
sns.distplot(data[' shares'], fit=norm);
fig = plt.figure()
res = probplot(data[' shares'], plot=plt)
'''
'Shares' doesn't have a normal distribution. It shows 'peakedness', positive skewness and does not follow the diagonal line.
Thus some statistic analysis might not be suitable for it
'''

#applying log transformation
new_shares_data = copy(data)
new_shares_data.loc[new_shares_data[' shares'] > 0, ' shares'] = np.log(data.loc[data[' shares'] > 0, ' shares'])
new_shares_log = new_shares_data[' shares']
#transformed histogram and normal probability plot
fig,ax = plt.subplots(figsize=(7,7))
sns.distplot(new_shares_log, fit=norm);
fig = plt.figure()
res = probplot(new_shares_log, plot=plt)

# use log transformation to transform each features to a normal distribution
# note log transformation can only be performed on data without zero value
for col in data.iloc[:,:-1].columns:
    #applying log transformation
    temp = data[data[col] == 0]
    # only apply to non-zero features
    if temp.shape[0] == 0:
        data[col] = np.log(data[col])
        print (col)

# Evaluating the impact of log transformation

# before log transformation
sns.distplot(original_data[' n_tokens_content'], fit=norm);         # number of written words in the content 

# after log transformation
sns.distplot(data[' n_tokens_content'], fit=norm);

# the data after log transformation and robust scaler
data.describe()

data.iloc[:,:-2]

# Data Clustering - Grouping Similar Articles together

from sklearn.decomposition import PCA        # Principal Component Analysis
from sklearn.cluster import KMeans

# Kmeans perform poorly on high feature space
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.iloc[:,:-2])
reduced_data.shape

# plotting the clusters PCA      
plt.figure(figsize=(7,7))
plt.plot(reduced_data[:,0], reduced_data[:,1], 'r.')
plt.title('PCA Transformation')
plt.show()

tsne = TSNE(n_components=2, n_iter=300)
reduced_tsne = tsne.fit_transform(data.iloc[:,:-2])
# plotting the clusters TSNE
plt.figure(figsize=(10,10))
plt.plot(reduced_tsne[:,0], reduced_tsne[:,1], 'r.')
plt.title('TSNE Transformation')
plt.show()

k=list(range(1,9))
ssd=[]
for i in k:
    kmeans=KMeans(n_clusters=i).fit(reduced_tsne)
    ssd.append(kmeans.inertia_)
plt.plot(k,ssd,'o-')
plt.xlabel('k')
plt.ylabel('Sum of squared error')
plt.show()

