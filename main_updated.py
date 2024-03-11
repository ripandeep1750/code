''' 
The process followed is highlighted below:
Data Cleaning - Noise removal
Data Transformation - Transform using log-transformation
Data Clustering - Grouping Similar Articles together.
Feature Selection and Evaluation
Machine Learning Classification
Summary and Conclusion.
'''

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

from google.colab import drive
drive.mount('/content/drive')

# Reading the data
data = pd.read_csv("/content/drive/MyDrive/structural virality/dataset/OnlineNewsPopularity.csv")
data.head(n=4)

# Data Processing and Noise Removal
# Here we drop the two non-preditive (url and timedelta) attributes. 
data.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
data.head(n=4)
# remove noise from n_tokens_content. those equals to 0
data  = data[data[' n_tokens_content'] != 0]
# Comment - Visualizing the n_non_stop_words data field shows that the present of a record with 1042 value,
# futher observation of that data shows that it belongs to entertainment which is not actually. It belongs to world news or others.
# this particluar also contains 0 on a lot of attributes. This record is classifed as a noise and will be remove.
data = data[data[' n_non_stop_words'] != 1042]
# Here, we will go ahead and drop the field of ' n_non_stop_words. 
data.drop(labels=[' n_non_stop_words'], axis = 1, inplace=True)
original_data = copy(data)

data.describe()

share_data = np.sort(data[' shares'].values)       # sort the values in ascending order
print(share_data.shape)
leng = share_data.shape[0]
middle = share_data[int(leng*0.7)-1]            # calculates value at 70th percentile. useful for dividing the data into two groups based on their popularity
middle

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
fig = plt.figure(figsize=(15,5))
ax = sns.countplot(x='popularity',data=data,alpha=0.5)
data_channel_data = data.groupby('popularity').size().reset_index()
data_channel_data.columns = ['popularity','No of articles']
data_channel_data

# Data Transformation - Log Transform
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

# before log transformation
sns.distplot(original_data[' n_tokens_content'], fit=norm);

# after log transformation
sns.distplot(data[' n_tokens_content'], fit=norm);

# Scale features using statistics that are robust to outliers.
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
# scalled all the feature selections aside shares and populairty
scalled_data = scaler.fit_transform(data.iloc[:, :-2])
# update the dataframe back with the scalled data
data.iloc[:, :-2] = scalled_data

# the data after log transformation and robust scaler
data.describe()

data.iloc[:,:-2]

# Data Clustering - Grouping Similar Articles together
from sklearn.decomposition import PCA    # Principal Component Analysis. For dimensionality reduction.
from sklearn.cluster import KMeans      

# Kmeans perform poorly on high feature space
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.iloc[:,:-2])
reduced_data.shape

# plotting the clusters PCA    # linear dimensionality reduction, computationally fast.
plt.figure(figsize=(5,5))
plt.plot(reduced_data[:,0], reduced_data[:,1], 'r.')
plt.title('PCA Transformation')
plt.show()

tsne = TSNE(n_components=2, n_iter=300)        # nonlinear dimensionality reduction, visualizing clusters, reveal complex or nonlinear structures
reduced_tsne = tsne.fit_transform(data.iloc[:,:-2])
# plotting the clusters TSNE
plt.figure(figsize=(10,10))
plt.plot(reduced_tsne[:,0], reduced_tsne[:,1], 'r.')
plt.title('TSNE Transformation')
plt.show()

# Elbow Method, find optimal number of clusters for KMeans clustering
k=list(range(1,9))
ssd=[]
for i in k:
    kmeans=KMeans(n_clusters=i).fit(reduced_tsne)
    ssd.append(kmeans.inertia_)
plt.plot(k,ssd,'o-')
plt.xlabel('k')
plt.ylabel('Sum of squared error')
plt.show()
# minimizes the SSD, indicating that the data points are well-clustered around their centroids.

# Predicts the clusters
kmeans=KMeans(init='k-means++',n_clusters=5)
kmeans.fit(reduced_tsne)
kmeans_preds=kmeans.predict(reduced_tsne)

centroids = kmeans.cluster_centers_
clusters = np.unique(kmeans_preds)
# ploting the result of of the clusters
ax, fig = plt.subplots(figsize=(10,7))
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
# ploting the cluster numbers
for i in range(clusters.shape[0]):
    plt.text(centroids[i, 0], centroids[i, 1], clusters[i], fontsize=20, color='white',
             bbox=dict(facecolor='black', alpha=0.5))
plt.scatter(reduced_tsne[:,0],reduced_tsne[:,1],c=kmeans_preds,marker='.')
plt.show()

# fussing the cluster data into the dataframe
data1=pd.concat([data.reset_index(drop=True), pd.DataFrame(kmeans_preds, columns=['clusters'])],axis=1)

data1.shape

# extrating individual cluster from the data
cluster1_data = data1[data1['clusters'] == 0]
cluster2_data = data1[data1['clusters'] == 1]
cluster3_data = data1[data1['clusters'] == 2]
cluster4_data = data1[data1['clusters'] == 3]
cluster5_data = data1[data1['clusters'] == 4]
print ('Cluster1 size: ',cluster1_data.shape)
print ('Cluster2 size: ',cluster2_data.shape)
print ('Cluster3 size: ',cluster3_data.shape)
print ('Cluster4 size: ',cluster4_data.shape)
print ('Cluster5 size: ',cluster5_data.shape)

# Feature Selection and Feature Extraction
''' 
Mutual Information
F-Score
Recursive Feature Elimination
PCA (Principal Component Analysis)
'''
# Mutual Information computation-------------------------------------------------------------
# our label is the popularity and will be disregarding the shares data
from sklearn.feature_selection import mutual_info_classif

# Mutual information for cluster 1
X1 = cluster1_data.iloc[:, :-3]
y1 = cluster1_data.iloc[:, -2]
mi_data_clus1 = mutual_info_classif(X1, y1)

# mututal information for cluster 2
X2 = cluster2_data.iloc[:, :-3]
y2 = cluster2_data.iloc[:, -2]
mi_data_clus2 = mutual_info_classif(X2, y2)

# mututal information for cluster 3
X3 = cluster3_data.iloc[:, :-3]
y3 = cluster3_data.iloc[:, -2]
mi_data_clus3 = mutual_info_classif(X3, y3)
# mututal information for cluster 4
X4 = cluster4_data.iloc[:, :-3]
y4 = cluster4_data.iloc[:, -2]
mi_data_clus4 = mutual_info_classif(X4, y4)
# mututal information for cluster 5
X5 = cluster5_data.iloc[:, :-3]
y5 = cluster5_data.iloc[:, -2]
mi_data_clus5 = mutual_info_classif(X5, y5)

# ploting the result of mutual information
plt.figure(figsize=(10, 5))
g = sns.barplot(x=X1.columns,y=mi_data_clus1)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Mutual Information for all Features - Cluster 1")

plt.figure(figsize=(10, 5))
g = sns.barplot(x=X2.columns,y=mi_data_clus2)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Mutual Information for all Features - Cluster 2")

plt.figure(figsize=(10, 5))
g = sns.barplot(x=X3.columns,y=mi_data_clus3)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Mutual Information for all Features - Cluster 3")

plt.figure(figsize=(10, 5))
g = sns.barplot(x=X4.columns,y=mi_data_clus4)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Mutual Information for all Features - Cluster 4")

plt.figure(figsize=(10, 5))
g = sns.barplot(x=X5.columns,y=mi_data_clus5)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Mutual Information for all Features - Cluster 5")

### an helper function for extracting the best features possible
def extract_best_features(feature_scores, feature_col, n=5, sort_metric=False):
    # this function extracts out the best features.
    # inputs
    temp = np.hstack((feature_scores.reshape(-1,1), feature_col.reshape(-1,1)))
    features = pd.DataFrame(temp, columns=['score', 'name'])
    # sort the features
    features = features.sort_values(by=['score'], ascending=sort_metric).reset_index(drop=True)
    # extract the best features
    best_features = features.iloc[:n, :].to_numpy()
    return best_features

best_features = extract_best_features(mi_data_clus4, X4.columns.values, n=10)
best_features

# F-Score ---------------------------------------------------------------------------------------
from sklearn.feature_selection import f_classif

# F-Score for cluster 1
f_test_data = f_classif(X1, y1)
f_score_1=f_test_data[0]
plt.figure(figsize=(10, 5))
g = sns.barplot(x=X1.columns,y=f_score_1)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("F score for all Features - Cluster 1")

# F-Score for cluster 2
f_test_data = f_classif(X2, y2)
f_score_2=f_test_data[0]
plt.figure(figsize=(10, 5))
g = sns.barplot(x=X2.columns,y=f_score_2)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("F score for all Features - Cluster 2")

# F-Score for cluster 3
f_test_data = f_classif(X3, y3)
f_score_3=f_test_data[0]
plt.figure(figsize=(10, 5))
g = sns.barplot(x=X3.columns,y=f_score_3)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("F score for all Features - Cluster 3")

# F-Score for cluster 4
f_test_data = f_classif(X4, y4)
f_score_4=f_test_data[0]

# F-Score for cluster 5
f_test_data = f_classif(X5, y5)
f_score_5=f_test_data[0]

plt.figure(figsize=(10, 5))
g = sns.barplot(x=X4.columns,y=f_score_4)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("F score for all Features - Cluster 4")

plt.figure(figsize=(10, 5))
g = sns.barplot(x=X4.columns,y=f_score_5)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("F score for all Features - Cluster 5")

best_features = extract_best_features(f_score_1, X1.columns.values, n=10)
best_features

# Recursive Feature Selection ---------------------------------------------------------------------
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Random forest is used as the model for RFE
# RFE for Cluster 1
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10)
# for 5 features
rfe = RFE(estimator=model, n_features_to_select=5)
rfe = rfe.fit(X1, y1)
rfe_5_features_clus1 = X1.columns.values[rfe.get_support()]
# for 10 features
rfe = RFE(estimator=model, n_features_to_select=10)
rfe = rfe.fit(X1, y1)
rfe_10_features_clus1 = X1.columns.values[rfe.get_support()]
# for 20 features
rfe = RFE(estimator=model, n_features_to_select= 20)
rfe = rfe.fit(X1, y1)
rfe_20_features_clus1  = X1.columns.values[rfe.get_support()]
# for 30 features
rfe = RFE(estimator=model, n_features_to_select=30)
rfe = rfe.fit(X1, y1)
rfe_30_features_clus1 = X1.columns.values[rfe.get_support()]

# Random forest is used as the model for RFE
# RFE for Cluster 2
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10)
# for 5 features
rfe = RFE(estimator=model, n_features_to_select=  5)
rfe = rfe.fit(X2, y2)
rfe_5_features_clus2 = X2.columns.values[rfe.get_support()]
# for 10 features
rfe = RFE(estimator=model, n_features_to_select=  10)
rfe = rfe.fit(X2, y2)
rfe_10_features_clus2 = X2.columns.values[rfe.get_support()]
# for 20 features
rfe = RFE(estimator=model, n_features_to_select=  20)
rfe = rfe.fit(X2, y2)
rfe_20_features_clus2  = X2.columns.values[rfe.get_support()]
# for 30 features
rfe = RFE(estimator=model, n_features_to_select=  30)
rfe = rfe.fit(X2, y2)
rfe_30_features_clus2 = X2.columns.values[rfe.get_support()]

# Random forest is used as the model for RFE
# RFE for Cluster 3
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10)
# for 5 features
rfe = RFE(estimator=model, n_features_to_select=5)
rfe = rfe.fit(X3, y3)
rfe_5_features_clus3 = X3.columns.values[rfe.get_support()]
# for 10 features
rfe = RFE(estimator=model, n_features_to_select= 10)
rfe = rfe.fit(X3, y3)
rfe_10_features_clus3 = X3.columns.values[rfe.get_support()]
# for 20 features
rfe = RFE(estimator=model, n_features_to_select= 20)
rfe = rfe.fit(X3, y3)
rfe_20_features_clus3  = X3.columns.values[rfe.get_support()]
# for 30 features
rfe = RFE(estimator=model, n_features_to_select= 30)
rfe = rfe.fit(X3, y3)
rfe_30_features_clus3 = X3.columns.values[rfe.get_support()]

# Random forest is used as the model for RFE
# RFE for Cluster 4
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10)
# for 5 features
rfe = RFE(estimator=model, n_features_to_select=  5)
rfe = rfe.fit(X4, y4)
rfe_5_features_clus4 = X4.columns.values[rfe.get_support()]
# for 10 features
rfe = RFE(estimator=model, n_features_to_select=  10)
rfe = rfe.fit(X4, y4)
rfe_10_features_clus4 = X4.columns.values[rfe.get_support()]
# for 20 features
rfe = RFE(estimator=model, n_features_to_select=  20)
rfe = rfe.fit(X4, y4)
rfe_20_features_clus4  = X4.columns.values[rfe.get_support()]
# for 30 features
rfe = RFE(estimator=model, n_features_to_select= 30)
rfe = rfe.fit(X4, y4)
rfe_30_features_clus4 = X4.columns.values[rfe.get_support()]

# Random forest is used as the model for RFE
# RFE for Cluster 5
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10)
# for 5 features
rfe = RFE(estimator=model, n_features_to_select= 5)
rfe = rfe.fit(X5, y5)
rfe_5_features_clus5 = X5.columns.values[rfe.get_support()]
# for 10 features
rfe = RFE(estimator=model, n_features_to_select= 10)
rfe = rfe.fit(X5, y5)
rfe_10_features_clus5 = X5.columns.values[rfe.get_support()]
# for 20 features
rfe = RFE(estimator=model, n_features_to_select= 20)
rfe = rfe.fit(X5, y5)
rfe_20_features_clus5  = X5.columns.values[rfe.get_support()]
# for 30 features
rfe = RFE(estimator=model, n_features_to_select= 30)
rfe = rfe.fit(X5, y5)
rfe_30_features_clus5 = X5.columns.values[rfe.get_support()]

# Plot the result of RFE for 5 features - Cluster 1
plt.figure(figsize=(10, 5))
g = sns.barplot(x=rfe_5_features_clus1, y=[1] * len(rfe_5_features_clus1))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("RFE for 5 Features - Cluster 1")
plt.show()
# Plot the result of RFE for 10 features - Cluster 1
plt.figure(figsize=(10, 5))
g = sns.barplot(x=rfe_10_features_clus1, y=[1] * len(rfe_10_features_clus1))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("RFE for 10 Features - Cluster 1")
plt.show()
# Plot the result of RFE for 20 features - Cluster 1
plt.figure(figsize=(10, 5))
g = sns.barplot(x=rfe_20_features_clus1, y=[1] * len(rfe_20_features_clus1))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("RFE for 20 Features - Cluster 1")
plt.show()
# Plot the result of RFE for 30 features - Cluster 1
plt.figure(figsize=(10, 5))
g = sns.barplot(x=rfe_30_features_clus1, y=[1] * len(rfe_30_features_clus1))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("RFE for 30 Features - Cluster 1")
plt.show()

#PCA-----------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA

# PCA for cluster 1
# for 5 features
transformer = PCA(n_components=5)
pca_clus1_5 = transformer.fit_transform(X1)
# for 10 features
transformer = PCA(n_components=10)
pca_clus1_10 = transformer.fit_transform(X1)
# for 20 features
transformer = PCA(n_components=20)
pca_clus1_20 = transformer.fit_transform(X1)
# for 30 features
transformer = PCA(n_components=30)
pca_clus1_30 = transformer.fit_transform(X1)

# ploting the result of PCA 
num_components = [5, 10, 20, 30]
# Perform PCA and visualize the results
for n in num_components:
    # Perform PCA
    transformer = PCA(n_components=n)
    pca_result = transformer.fit_transform(X1)
    # Create a dataframe with PCA results
    pca_results_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n)])
    # Plot the explained variance ratio
    plt.figure(figsize=(15, 5))
    sns.barplot(x=pca_results_df.columns, y=transformer.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA for {n} Features - Cluster 1')
    plt.show()

# PCA for cluster 2
# for 5 features
transformer = PCA(n_components=5)
pca_clus2_5 = transformer.fit_transform(X2)
# for 10 features
transformer = PCA(n_components=10)
pca_clus2_10 = transformer.fit_transform(X2)
# for 20 features
transformer = PCA(n_components=20)
pca_clus2_20 = transformer.fit_transform(X2)
# for 30 features
transformer = PCA(n_components=30)
pca_clus2_30 = transformer.fit_transform(X2)

# PCA for cluster 3
# for 5 features
transformer = PCA(n_components=5)
pca_clus3_5 = transformer.fit_transform(X3)
# for 10 features
transformer = PCA(n_components=10)
pca_clus3_10 = transformer.fit_transform(X3)
# for 20 features
transformer = PCA(n_components=20)
pca_clus3_20 = transformer.fit_transform(X3)
# for 30 features
transformer = PCA(n_components=30)
pca_clus3_30 = transformer.fit_transform(X3)

# PCA for cluster 4
# for 5 features
transformer = PCA(n_components=5)
pca_clus4_5 = transformer.fit_transform(X4)
# for 10 features
transformer = PCA(n_components=10)
pca_clus4_10 = transformer.fit_transform(X4)
# for 20 features
transformer = PCA(n_components=20)
pca_clus4_20 = transformer.fit_transform(X4)
# for 30 features
transformer = PCA(n_components=30)
pca_clus4_30 = transformer.fit_transform(X4)

# PCA for cluster 5
# for 5 features
transformer = PCA(n_components=5)
pca_clus5_5 = transformer.fit_transform(X5)
# for 10 features
transformer = PCA(n_components=10)
pca_clus5_10 = transformer.fit_transform(X5)
# for 20 features
transformer = PCA(n_components=20)
pca_clus5_20 = transformer.fit_transform(X5)
# for 30 features
transformer = PCA(n_components=30)
pca_clus5_30 = transformer.fit_transform(X5)

# Classification of Article Popularity
# encoding the label set with a label encoder
from sklearn.preprocessing import LabelEncoder
labelEn = LabelEncoder()
encoded_labels = labelEn.fit_transform(y1.values)
class_names = labelEn.classes_
class_names

# Splitting the data for Training and Testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import recall_score

# KNN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PCA Features(KNN)
# defining the model
from sklearn.neighbors import KNeighborsClassifier
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
pca_data = [pca_clus1_5, pca_clus1_10, pca_clus1_20, pca_clus1_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 1::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
pca_data = [pca_clus2_5, pca_clus2_10, pca_clus2_20, pca_clus2_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 2::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
pca_data = [pca_clus3_5, pca_clus3_10, pca_clus3_20, pca_clus3_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 3::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
pca_data = [pca_clus4_5, pca_clus4_10, pca_clus4_20, pca_clus4_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 4::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
pca_data = [pca_clus5_5, pca_clus5_10, pca_clus5_20, pca_clus5_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 5::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

#### KNN Cross-Validation
import matplotlib.pyplot as plt
# Cros Validation for any of the features
k_range = np.arange(1,120)
accuracy = []
for n in k_range:
    neigh = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    neigh.fit(X_train, y_train)
    # predict the result
    y_pred = neigh.predict(X_test)
    #print ("Random Forest Classifer Result")
    #print ("Performance - " + str(100*accuracy_score(y_pred, y_test_2)) + "%")
    accuracy.append(100*accuracy_score(y_pred, y_test))
plt.figure(figsize=(20,13))
plt.plot(k_range, accuracy, 'r-', label='KNN Accuracy Vs KNN Neighbors size')
plt.plot(k_range, accuracy, 'bx')
plt.xlabel('KNN Neighbors size')
plt.ylabel('KNN Accuracy')
plt.legend()
plt.grid()
plt.title('KNN Accuracy Vs Neighbors size')
plt.show()

# Mutual Information(KNN)-------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus1, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(best_features[:,1]))
    data_clus_mi = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 1::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus2, X2.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(best_features[:,1]))
    data_clus_mi = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 2::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus3, X3.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(best_features[:,1]))
    data_clus_mi = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 3::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))


# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus4, X4.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(best_features[:,1]))
    data_clus_mi = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 4::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus5, X5.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(best_features[:,1]))
    data_clus_mi = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 5::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# F-score (KNN)-----------------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_1, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(best_features[:,1]))
    data_clus_mi = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 1::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_2, X2.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(best_features[:,1]))
    data_clus_mi = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 2::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_3, X3.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(best_features[:,1]))
    data_clus_mi = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 3::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_4, X4.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(best_features[:,1]))
    data_clus_mi = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 4::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_5, X5.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(best_features[:,1]))
    data_clus_mi = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 5::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# RFE(KNN)-------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
rfe_features_clus1 = [rfe_5_features_clus1, rfe_10_features_clus1, rfe_20_features_clus1, rfe_30_features_clus1]
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(rfe_features_clus1[i]))
    data_clus_rfe = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 1::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
rfe_features_clus2 = [rfe_5_features_clus2, rfe_10_features_clus2, rfe_20_features_clus2, rfe_30_features_clus2]
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(rfe_features_clus2[i]))
    data_clus_rfe = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 2::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
rfe_features_clus3 = [rfe_5_features_clus3, rfe_10_features_clus3, rfe_20_features_clus3, rfe_30_features_clus3]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(rfe_features_clus3[i]))
    data_clus_rfe = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 3::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
rfe_features_clus4 = [rfe_5_features_clus4, rfe_10_features_clus4, rfe_20_features_clus4, rfe_30_features_clus4]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(rfe_features_clus4[i]))
    data_clus_rfe = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)

    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 4::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
rfe_features_clus5 = [rfe_5_features_clus5, rfe_10_features_clus5, rfe_20_features_clus5, rfe_30_features_clus5]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(rfe_features_clus5[i]))
    data_clus_rfe = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    neigh = KNeighborsClassifier(n_neighbors=63, n_jobs=-1)
    neigh.fit(X_train, y_train)
    
    # predict the result
    y_pred = neigh.predict(X_test)
    print ("KNN - Cluster 3::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# Random Forest ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PCA
# defining the model
from sklearn.ensemble import RandomForestClassifier
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
pca_data = [pca_clus1_5, pca_clus1_10, pca_clus1_20, pca_clus1_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 1::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# RandomForest(MI) ---------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus1, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(best_features[:,1]))
    data_clus_mi = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 1::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus2, X2.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(best_features[:,1]))
    data_clus_mi = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 2::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus3, X3.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(best_features[:,1]))
    data_clus_mi = X3.drop(drop_these, axis=1, inplace=False)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 3::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus4, X4.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(best_features[:,1]))
    data_clus_mi = X4.drop(drop_these, axis=1, inplace=False)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 4::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus5, X5.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(best_features[:,1]))
    data_clus_mi = X5.drop(drop_these, axis=1, inplace=False)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 5::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))
# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
pca_data = [pca_clus2_5, pca_clus2_10, pca_clus2_20, pca_clus2_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.3, shuffle=False)

    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 2::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
pca_data = [pca_clus3_5, pca_clus3_10, pca_clus3_20, pca_clus3_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.3, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=100,
                                random_state=0)
    clf.fit(X_train, y_train)
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 3::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
pca_data = [pca_clus4_5, pca_clus4_10, pca_clus4_20, pca_clus4_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.3, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 4::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
pca_data = [pca_clus5_5, pca_clus5_10, pca_clus5_20, pca_clus5_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    # For PCA Feature Extraction
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.3, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=100,
                                 random_state=0)
    clf.fit(X_train, y_train)
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 5::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

#### Cross validation For Random Forest
nns = [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 3000]
accuracy = []
for n in nns:
    clf = RandomForestClassifier(n_estimators=n, n_jobs=5, max_depth=500,
                                 random_state=0)
    clf.fit(X_train, y_train)
    # predict the result
    y_pred = clf.predict(X_test)
    #print ("Random Forest Classifer Result")
    #print ("Performance - " + str(100*accuracy_score(y_pred, y_test_2)) + "%")
    accuracy.append(100*accuracy_score(y_pred, y_test))
plt.figure(figsize=(10,7))
plt.plot(nns, accuracy, 'r-', label='Random Forest Accuracy Vs Number of Tress')
plt.plot(nns, accuracy, 'bx')
plt.xlabel('Random Forest Tree Sizes')
plt.ylabel('Random Forest Accuracy')
plt.legend()
plt.grid()
plt.title('Random Forest Accuracy Vs Number of Tress')
plt.show()

# Random Forest (F-Score)-------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_1, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(best_features[:,1]))
    data_clus_mi = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 1::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_2, X2.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(best_features[:,1]))
    data_clus_mi = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 2::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_3, X3.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(best_features[:,1]))
    data_clus_mi = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 3::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_4, X4.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(best_features[:,1]))
    data_clus_mi = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 4::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    best_features = extract_best_features(f_score_5, X5.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(best_features[:,1]))
    data_clus_mi = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 5::F-score - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# Random Forest (RFE)-----------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
rfe_features_clus1 = [rfe_5_features_clus1, rfe_10_features_clus1, rfe_20_features_clus1, rfe_30_features_clus1]
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(rfe_features_clus1[i]))
    data_clus_rfe = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)

    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 1::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
rfe_features_clus2 = [rfe_5_features_clus2, rfe_10_features_clus2, rfe_20_features_clus2, rfe_30_features_clus2]
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(rfe_features_clus2[i]))
    data_clus_rfe = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)

    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 2::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
rfe_features_clus3 = [rfe_5_features_clus3, rfe_10_features_clus3, rfe_20_features_clus3, rfe_30_features_clus3]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(rfe_features_clus3[i]))
    data_clus_rfe = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)

    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 3::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
rfe_features_clus4 = [rfe_5_features_clus4, rfe_10_features_clus4, rfe_20_features_clus4, rfe_30_features_clus4]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(rfe_features_clus4[i]))
    data_clus_rfe = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)

    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 4::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
rfe_features_clus5 = [rfe_5_features_clus5, rfe_10_features_clus5, rfe_20_features_clus5, rfe_30_features_clus5]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(rfe_features_clus5[i]))
    data_clus_rfe = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("Random Forest - Cluster 5::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# SVM(RFE)+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.svm import SVC

# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
rfe_features_clus1 = [rfe_5_features_clus1, rfe_10_features_clus1, rfe_20_features_clus1, rfe_30_features_clus1]
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(rfe_features_clus1[i]))
    data_clus_rfe = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 1::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
rfe_features_clus2 = [rfe_5_features_clus2, rfe_10_features_clus2, rfe_20_features_clus2, rfe_30_features_clus2]
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(rfe_features_clus2[i]))
    data_clus_rfe = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 2::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
rfe_features_clus3 = [rfe_5_features_clus3, rfe_10_features_clus3, rfe_20_features_clus3, rfe_30_features_clus3]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(rfe_features_clus3[i]))
    data_clus_rfe = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 3::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
rfe_features_clus4 = [rfe_5_features_clus4, rfe_10_features_clus4, rfe_20_features_clus4, rfe_30_features_clus4]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(rfe_features_clus4[i]))
    data_clus_rfe = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 4::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
rfe_features_clus5 = [rfe_5_features_clus5, rfe_10_features_clus5, rfe_20_features_clus5, rfe_30_features_clus5]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(features_list)):
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(rfe_features_clus5[i]))
    data_clus_rfe = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_rfe, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 5::RFE - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# SVM (PCA)-------------------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
pca_data = [pca_clus1_5, pca_clus1_10, pca_clus1_20, pca_clus1_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 1::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
pca_data = [pca_clus2_5, pca_clus2_10, pca_clus2_20, pca_clus2_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 2::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
pca_data = [pca_clus3_5, pca_clus3_10, pca_clus3_20, pca_clus3_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 3::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
pca_data = [pca_clus4_5, pca_clus4_10, pca_clus4_20, pca_clus4_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 4::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
pca_data = [pca_clus5_5, pca_clus5_10, pca_clus5_20, pca_clus5_30]
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
for i in range(len(pca_data)):
    X_train, X_test, y_train, y_test = train_test_split(pca_data[i], encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 5::PCA - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

SVM(MI)-------------------------------------------------------------------------------------------------------------
# For Cluster 1
encoded_labels = labelEn.fit_transform(y1.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus1, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X1.columns.values) - set(best_features[:,1]))
    data_clus_mi = X1.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 1:: MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 2
encoded_labels = labelEn.fit_transform(y2.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus2, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X2.columns.values) - set(best_features[:,1]))
    data_clus_mi = X2.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 2::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 3
encoded_labels = labelEn.fit_transform(y3.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus3, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X3.columns.values) - set(best_features[:,1]))
    data_clus_mi = X3.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 3::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 4
encoded_labels = labelEn.fit_transform(y4.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus4, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X4.columns.values) - set(best_features[:,1]))
    data_clus_mi = X4.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 4::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# For Cluster 5
encoded_labels = labelEn.fit_transform(y5.values)
features_list = ['5 Features', '10 Features', '20 Features', '30 Features']
n_features = [5, 10, 20, 30]
for i in range(len(features_list)):
    best_features = extract_best_features(mi_data_clus5, X1.columns.values, n=n_features[i])
    # the feautures are stored in the seconds column
    drop_these = list(set(X5.columns.values) - set(best_features[:,1]))
    data_clus_mi = X5.drop(drop_these, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(data_clus_mi, encoded_labels, test_size=0.2, shuffle=False)
    clf = SVC(gamma='auto')
    # commence training - NOTE: It takes hours to be complete
    clf.fit(X_train, y_train)
    
    # predict the result
    y_pred = clf.predict(X_test)
    print ("SVC - Cluster 5::MI - " + str(features_list[i]))
    print ("Accuracy - " + str(100*accuracy_score(y_pred, y_test)) + "%")
    print ("Recall - " + str(recall_score(y_test, y_pred, average='micro')))

# SVM (F-Score) ---------------------------------------------------------------------------------------------------
