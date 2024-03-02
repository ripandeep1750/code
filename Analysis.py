''' 
In this project, the goal is the explore the dataset given and be able to find critical insights that can be used to influence potential
article popularity. Also, machine learning models was built to be able to predict the popularity of a given article.

The process followed is highlighted below:
Data Cleaning - Noise detection and removal
Subjective analysis - Using our intuition to evaluate a data variable/feature and decide whether a variable influences the popularity of 
the article or not.
Quantitative Analysis - How correct is our intuition? Here we carry our several analysis to accept or debunk our initial hypothesis
Normal Distribuiton Observation on the dataset
Feature Selection and Evaluation
Machine Learning Classification
Summary and Conclusion. 
'''


# Libaries import
# Libaries import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy

#reading the data
data = pd.read_csv("OnlineNewsPopularity.csv")
data.head(n=4)
origianl_data = copy(data)
data.columns

# initial_columns = data.shape[1]
# print(initial_columns)
initial_rows, initial_columns = data.shape
print("Number of rows:", initial_rows)
print("Number of columns:", initial_columns)

# Here we drop the two non-preditive (url and timedelta) attributes. They won't contribute anything
data.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
data.head(n=4)

# describing the data
data.describe()
# from the data, there will be need to normailze the data if their will be need for condersing any machine learning model.

# creating a grading criteria for the shares
share_data = data[' shares']
data[' shares'].describe()

# create label grades for the classes
share_label = list()
for share in share_data:
    if share <= 645:
        share_label.append('Very Poor')
    elif share > 645 and share <= 861:
        share_label.append('Poor')
    elif share > 861 and share <= 1400:
        share_label.append('Average')
    elif share > 1400 and share <= 31300:
        share_label.append('Good')
    elif share > 31300 and share <= 53700:
        share_label.append('Very Good')
    elif share > 53700 and share <= 77200:
        share_label.append('Excellent')
    else:
        share_label.append('Exceptional')

# Update this class label into the dataframe
data = pd.concat([data, pd.DataFrame(share_label, columns=['popularity'])], axis=1)
data.head(4)

# Merging the weekdays columns channels as one single column
publishdayMerge=data[[' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday', 
                      ' weekday_is_thursday', ' weekday_is_friday',' weekday_is_saturday' ,' weekday_is_sunday' ]]
temp_arr=[]
for r in list(range(publishdayMerge.shape[0])):
    for c in list(range(publishdayMerge.shape[1])):
        if ((c==0) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Monday')
        elif ((c==1) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Tueday')
        elif ((c==2) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Wednesday')
        elif ((c==3) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Thursday')
        elif ((c==4) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Friday')
        elif ((c==5) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Saturday') 
        elif ((c==6) and (publishdayMerge.iloc[r,c])==1):
            temp_arr.append('Sunday')

# Merging the data channels as one single column
DataChannelMerge=data[[' data_channel_is_lifestyle',' data_channel_is_entertainment' ,' data_channel_is_bus',
                        ' data_channel_is_socmed' ,' data_channel_is_tech',' data_channel_is_world' ]]
#logic to merge data channel
DataChannel_arr=[]
for r in list(range(DataChannelMerge.shape[0])):
    if (((DataChannelMerge.iloc[r,0])==0) and ((DataChannelMerge.iloc[r,1])==0) and ((DataChannelMerge.iloc[r,2])==0) and ((DataChannelMerge.iloc[r,3])==0) and ((DataChannelMerge.iloc[r,4])==0) and ((DataChannelMerge.iloc[r,5])==0)):
        DataChannel_arr.append('Others')
    for c in list(range(DataChannelMerge.shape[1])):
        if ((c==0) and (DataChannelMerge.iloc[r,c])==1):
            DataChannel_arr.append('Lifestyle')
        elif ((c==1) and (DataChannelMerge.iloc[r,c])==1):
            DataChannel_arr.append('Entertainment')
        elif ((c==2) and (DataChannelMerge.iloc[r,c])==1):
            DataChannel_arr.append('Business')
        elif ((c==3) and (DataChannelMerge.iloc[r,c])==1):
            DataChannel_arr.append('Social Media')
        elif ((c==4) and (DataChannelMerge.iloc[r,c])==1):
            DataChannel_arr.append('Tech')
        elif ((c==5) and (DataChannelMerge.iloc[r,c])==1):
            DataChannel_arr.append('World')

# merge the the new data into the dataframe
data.insert(loc=11, column='weekdays', value=temp_arr)
data.insert(loc=12, column='data_channel', value=DataChannel_arr)

# Now I drop the old data
data.drop(labels=[' data_channel_is_lifestyle',' data_channel_is_entertainment' ,' data_channel_is_bus',
                        ' data_channel_is_socmed' ,' data_channel_is_tech',' data_channel_is_world', 
                 ' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday', 
                      ' weekday_is_thursday', ' weekday_is_friday',' weekday_is_saturday' ,' weekday_is_sunday'], axis = 1, inplace=True)
print(data.shape)
data.head(n=4)

data.columns

# Evaluating features (sensors) contribution towards the label
fig = plt.figure(figsize=(9,5))
ax = sns.countplot(x='popularity',data=data,alpha=0.5)

# Fetch the counts for each class
class_counts = data.groupby('popularity').size().reset_index()
class_counts.columns = ['Popularity','No of articles']
class_counts

# Visualizaing the "low" expectation hypothesis
# n_non_stop_words
print(data[' n_non_stop_words'].describe())
# Comment - Visualizing the n_non_stop_words data field shows that the present of a record with 1042 value, 
# futher observation of that data shows that it belongs to entertainment which is not actually. It belongs to world news or others.
# this particluar also contains 0 on a lot of attributes. This record is classifed as a noise and will be remove.
data = data[data[' n_non_stop_words'] != 1042]
# Here, we will go ahead and drop the field of ' n_non_stop_words'
data.drop(labels=[' n_non_stop_words'], axis = 1, inplace=True)

# remove noise from n_tokens_content(number of words or meaningful units). those equals to 0
data  = data[data[' n_tokens_content'] != 0]
print ("After noise removal - ",data.shape)

# n_non_stop_unique_tokens
data[' n_non_stop_unique_tokens'].describe()
# a lot of unique words, it is better to use a different plot from bar plots
# line plot
temp_data = data[data[' shares'] <= 100000]
fig, axes = plt.subplots(figsize=(7,5))
# box plot
sns.boxplot(x='popularity', y=' n_non_stop_unique_tokens', data=data, ax=axes)
# box plot of the dataset shows majority (75%) of the data inrespective of their shares is in the range of 0.6 - 0.8.
# So does it offers any uniques? No, it doesn't.

#kw_min_min and related kw_ terms
data[' kw_min_min'].describe()
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg', 
            ' kw_max_avg', ' kw_avg_avg', ' shares']
# run a pairplot
# sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')

#Finding relationship between 'rate_positive_words', 'rate_negative_words', 'global_rate_positive_words', 'global_rate_negative_words', and 'shares'
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the these terms
kw_cols = [' rate_positive_words', ' rate_negative_words', ' global_rate_positive_words', ' global_rate_negative_words', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
'''
There is a linear relationship between rate_positive_words and rate_negative_words (it is expected)
rate_positive_words = No special relationship or observable trait was observed for this variable. Although most of articles tends 
to be on falls towards the 0.3 - 1
rate_negative_words = No special relationship or observable trait was observed for this variable. Although most of articles tends 
to be on falls towards the 0.8 - 0 = Note the articles with popularity less than "average" have the lowest negative score rate.
global_rate_positive_words - There is a slight relationship with shares. - Medium
global_rate_negative_words - There is a slight relationship with shares. - Medium
'''

# attempt polartiy
temp_data = data[data[' shares'] <= 100000]
sns.lmplot(x=' avg_positive_polarity', y=' shares', col='popularity', data=temp_data)

# attempt polartiy
temp_data = data[data[' shares'] <= 100000]
fig, axes = plt.subplots(figsize=(5,4))
sns.scatterplot(x=' avg_positive_polarity', y=' shares', hue='popularity', data=temp_data, ax=axes)

#Finding relationship between 'rate_positive_words', 'rate_negative_words', 'global_rate_positive_words', 'global_rate_negative_words', and 'shares'
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the terms
kw_cols = [' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
'''
avg_positive_polarity and avg_negative_polarity are good features with some clear observation 
'''

# attempt title_sentiment_polarity
temp_data = data[data[' shares'] <= 100000]
fig, axes = plt.subplots(figsize=(7,5))
sns.scatterplot(x=' title_sentiment_polarity', y=' shares', hue='popularity', data=temp_data, ax=axes)

# attempt title_subjectivity
temp_data = data[data[' shares'] <= 100000]
fig, axes = plt.subplots(figsize=(15,15))
sns.relplot(x=' title_subjectivity', y=' shares', hue='popularity', data=temp_data, ax=axes)
'''
sns.relplot() is used instead of sns.scatterplot(). The relplot() function in seaborn is a higher-level function that can create 
different types of relational plots, including scatter plots. It provides greater flexibility for creating various plot types and 
supports additional parameters for customization.
'''

temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' title_sentiment_polarity', ' abs_title_sentiment_polarity', ' title_subjectivity', ' abs_title_subjectivity', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')

# attempt self_reference_min_shares
temp_data = data[(data[' shares'] <= 100000) & (data[' self_reference_min_shares'] <= 30000)]
fig, axes = plt.subplots(figsize=(7,7))
sns.scatterplot(x=' self_reference_min_shares', y=' shares', hue= 'popularity', data=temp_data, ax=axes)

temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' self_reference_min_shares', ' self_reference_max_shares', ' self_reference_avg_sharess', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')

#### LDA - 0: 5   (Latent Dirichlet Allocation)
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
''' 
LDA (Latent Dirichlet Allocation): LDA is a topic modeling technique used to discover latent topics in a collection of documents. 
It assumes that each document is a mixture of various topics, and each topic is characterized by a distribution of words. 
The LDA algorithm aims to uncover these latent topics and their word distributions.
In the context of your code, the labels LDA_00, LDA_01, LDA_02, LDA_03, and LDA_04 represent the probability distribution of an article
belonging to each of the five identified topics. Each label corresponds to the probability of an article being associated with a specific
topic. For example, LDA_00 represents the probability of an article belonging to Topic 0, LDA_01 represents the probability of an article
belonging to Topic 1, and so on. In summary, the LDA labels (LDA_00, LDA_01, LDA_02, LDA_03, LDA_04) in your code represent the
probability distribution of topics assigned to each article based on the Latent Dirichlet Allocation (LDA) topic modeling algorithm. 
'''

# extact the weekdays articles distrubution
weekdays_data = data.groupby('weekdays').size().reset_index()
weekdays_data.columns = ['weekdays','count']
weekdays_data

# shows the days when articles are usually posted
fig, axes = plt.subplots(figsize=(7,5))
ax = sns.countplot(x='weekdays',data=data,alpha=0.5, ax=axes)

# shows relationship with the number of shares and the weekdays
temp_data = data[(data['popularity'] == 'Very Poor') | (data['popularity'] == 'Poor') | (data['popularity'] == 'Average') | (data['popularity'] == 'Good')]
ax = sns.catplot(x='weekdays', col="popularity", data=temp_data, kind="count", height=10, aspect=.7)

# shows relationship with the number of shares and the weekdays (compare only the best three popularity)
temp_data = data[(data['popularity'] == 'Exceptional') | (data['popularity'] == 'Excellent') | (data['popularity'] == 'Very Good')]
ax = sns.catplot(x='weekdays', col="popularity", data=temp_data, kind="count", height=20, aspect=.7)
'''
It seems the best popular articles are usually posted on Mondays and Wednesday (and a bit of tuesdays) Sundays and Saturdays 
(Weekends generally) are the worsts days to publish an articles. Your chances are low 
'''

temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' average_token_length', ' num_keywords', ' global_subjectivity', ' global_sentiment_polarity', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
''' 
'average_token_length': average length of tokens (such as words) in the content.
'num_keywords': number of keywords associated with the content.
'global_subjectivity': the subjectivity score of the content, indicating how subjective or opinion-based the content is.
'global_sentiment_polarity': the sentiment polarity score of the content, indicating the overall sentiment expressed (positive or negative).
'''

## Seeing the distribution of the articles across the data channels
# extact the weekdays articles distrubution
data_channel_data = data.groupby('data_channel').size().reset_index()
data_channel_data.columns = ['Data Channels','No of articles']
data_channel_data

# Shows the distribution of the articles across the channels
sns.catplot(x='data_channel', data=data, kind="count", height=10, aspect=.7)

#the ranking of the channels in regards to the shares popularity
temp_data = data[(data['popularity'] == 'Very Poor') | (data['popularity'] == 'Poor') | (data['popularity'] == 'Average') | (data['popularity'] == 'Good')]
ax = sns.catplot(x='data_channel', col="popularity", data=temp_data, kind="count", height=8, aspect=.7)

# shows relationship with the number of shares and the ranking of the channels (compare only the best three popularity)
temp_data = data[(data['popularity'] == 'Exceptional') | (data['popularity'] == 'Excellent') | (data['popularity'] == 'Very Good')]
ax = sns.catplot(x='data_channel', col="popularity", data=temp_data, kind="count", height=10, aspect=.7)

fig,ax = plt.subplots(figsize=(7,7))
temp_data = data[data[' num_imgs'] <= 25]
sns.boxplot(x='popularity',y=' num_imgs', hue='data_channel', data=temp_data, showfliers=False)

#n_tokens_content
sns.scatterplot(x=' n_tokens_content',y='popularity', data=data)
''' 
n_tokens_content' likely represents the number of tokens (such as words) present in the content of each article. It can be seen 
than good articles will generally tend to have n_tokens_content less than 2000 and greater than 100 words 
'''

#n_tokens_title
temp_data = data[data[' shares'] <= 200000]
sns.scatterplot(x=' n_tokens_title',y=' shares', hue='popularity', data=temp_data)
# it can be seen than good articles will generally tend to have n_tokens_title between 6 and 17 words

#n_tokens_title
temp_data = data[data[' shares'] <= 200000]
plt.figure(figsize=(10,10))
sns.scatterplot(x=' n_unique_tokens',y=' shares', hue='popularity', data=data)

#num_hrefs
temp_data = data[data[' shares'] <= 100000]
sns.scatterplot(x=' num_hrefs',y=' shares', hue='popularity', data=temp_data)
# 'num_hrefs' represents the number of hyperlinks present in each article.

#num_imgs
temp_data = data[data[' shares'] <= 100000]
#plt.figure(figsize=(30,10))
#sns.barplot(x=' num_imgs',y=' shares', hue='popularity', data=temp_data)
sns.lmplot(x=' num_imgs', y=' shares', col='popularity', data=temp_data)

#num_videos
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' num_videos'] == 0]
print (noise_data.shape)
# plt.figure(figsize=(30,10))
# sns.barplot(x=' num_imgs',y=' shares', hue='popularity', data=temp_data)
sns.lmplot(x=' num_videos', y=' shares', col='popularity', data=temp_data)

#average_token_length
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' average_token_length'] == 0]
print (noise_data.shape)
#plt.figure(figsize=(30,10))
sns.scatterplot(x=' average_token_length',y=' shares', hue='popularity', data=temp_data)

#num_keywords
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' num_keywords'] == 0]
print (noise_data.shape)
#plt.figure(figsize=(30,10))
sns.scatterplot(x=' num_keywords',y=' shares', hue='popularity', data=temp_data)

#self_reference_avg_sharess
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' self_reference_avg_sharess'] == 0]
print (noise_data.shape)
#plt.figure(figsize=(30,10))
sns.scatterplot(x=' self_reference_avg_sharess',y=' shares', hue='popularity', data=temp_data)
sns.lmplot(x=' self_reference_avg_sharess', y=' shares', col='popularity', data=temp_data)

# Normal Distribution analysis for 'Shares'

print("Skewness: %f" % data[' shares'].skew())
print("Kurtosis: %f" % data[' shares'].kurt())

''' 
Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable. A skewness value of 0 
indicates a perfectly symmetrical distribution. A positive skewness value indicates a right-skewed (positively skewed) distribution,
where the tail of the distribution is elongated towards the right. A negative skewness value indicates a left-skewed (negatively skewed) 
distribution, where the tail of the distribution is elongated towards the left.
"Skewness: 34.952465", it means that the 'shares' column has a significant positive skewness
Kurtosis is a measure of the tailedness or peakedness of the probability distribution of a real-valued random variable. Kurtosis compares
the shape of the distribution to that of a normal distribution. A kurtosis value of 0 indicates a normal distribution. Positive kurtosis 
(greater than 0) indicates heavy-tailedness or more outliers compared to a normal distribution. Negative kurtosis (less than 0) indicates
light-tailedness or fewer outliers compared to a normal distribution.
"Kurtosis: 1909.931080", it means that the 'shares' column has a very high positive kurtosis, indicating that the distribution of the 
'shares' data has heavy-tailedness and contains a large number of outliers compared to a normal distribution.
'''

from scipy.stats import norm, probplot       # Normal distribution
#histogram and normal probability plot       # determine if the 'shares' data follows a normal distribution
temp_data = data[data[' shares'] <= 100000]
fig,ax = plt.subplots(figsize=(10,10))
sns.distplot(data[' shares'], fit=norm);
fig = plt.figure()
res = probplot(data[' shares'], plot=plt)    # probability plot or quantile-quantile (Q-Q) plot
'''
'Shares' doesn't have a normal distribution. It shows 'peakedness', positive skewness and does not follow the diagonal line.
Thus some statistic analysis might not be suitable for it
The histogram shows the distribution of the 'shares' data, while the curve represents what the distribution would look like if it were a
normal distribution. This plot compares the 'shares' data to what it would look like if it followed a normal distribution. If the 'shares'
data closely matches the diagonal line on the plot, it means it follows a normal distribution.
''' 

from copy import copy

#applying log transformation
new_shares_data = copy(data)
new_shares_data.loc[new_shares_data[' shares'] > 0, ' shares'] = np.log(data.loc[data[' shares'] > 0, ' shares'])
'''
selects the rows where the 'shares' value is greater than 0 and replaces those values with their natural logarithm using the np.log()
function. This transformation is performed to address the positive skewness in the data. 
'''
new_shares_log = new_shares_data[' shares']     # contains the log-transformed 'shares' data.
#transformed histogram and normal probability plot
fig,ax = plt.subplots(figsize=(7,7))
sns.distplot(new_shares_log, fit=norm);
fig = plt.figure()
res = probplot(new_shares_log, plot=plt)

#Check for missing data
total = new_shares_data.isnull().sum().sort_values(ascending=False)
percent = (new_shares_data.isnull().sum()/new_shares_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()

# Features selection on whole dataset

# shares data is not needed for classification
data_without_shares = data.drop(labels=[' shares'], axis = 1, inplace=False)
training_set = data_without_shares.iloc[:, :(data_without_shares.shape[1]-1)]
'''
The iloc indexer is used to select rows and columns by their integer positions. The : before the comma represents all rows, 
and :(data_without_shares.shape[1]-1) represents all columns except the last one. The shape[1] gives the number of columns in the 
'data_without_shares' dataframe.
'''
# convert categorical variables into dummy - it use one-hot encoding
training_set = pd.get_dummies(training_set)
'''
One-hot encoding is a technique used to convert categorical variables into a numerical representation that can be used by machine 
learning algorithms.
'''
# extract the label data in this case popularity
label_set = data_without_shares.iloc[:, (data_without_shares.shape[1]-1):].value
'''
stores the values of the last column from the 'data_without_shares' dataframe. This column represents the target variable or label for
classification. The : after the comma represents all rows, and (data_without_shares.shape[1]-1): represents the last column. The 
.values at the end converts the selected column into a NumPy array.
'''
data_feature2 = copy(training_set)

# Feature selection based on best hypothesis observed

# only the best observed features are extracted here
data_feature1 =training_set[[' n_tokens_title',' n_tokens_content',' n_unique_tokens',' num_hrefs',
                       ' num_self_hrefs',' num_imgs',' num_videos',' average_token_length',' num_keywords',
                       ' kw_avg_avg',' self_reference_avg_sharess',' global_subjectivity',
                       ' global_sentiment_polarity',' global_rate_positive_words',' global_rate_negative_words',' avg_positive_polarity',
                       ' avg_negative_polarity',' title_sentiment_polarity','weekdays_Friday', 'weekdays_Monday', 'weekdays_Saturday',
       'weekdays_Sunday', 'weekdays_Thursday', 'weekdays_Tueday',
       'weekdays_Wednesday', 'data_channel_Business',
       'data_channel_Entertainment', 'data_channel_Lifestyle',
       'data_channel_Others', 'data_channel_Social Media', 'data_channel_Tech',
       'data_channel_World']]

data_feature1.columns

# Feature Selection Based on Fisher Discrimating Analysis

# helper function for evalating the fisher ndex
def fisher_index_calc(trainingSet, labelSet):
    (dim1_T, dim2_T) = trainingSet.shape
    (dim1_L, dim2_L) = labelSet.shape
    # create the fisher output variable - A vector of all the features
    fisher_ratios = np.zeros((1, dim2_T), dtype=float).flatten()
    # It's expected that the dim1_T and dim1_L be of the same size, else this input parameters is nulled.
    if dim1_L != dim1_T:
        return fisher_ratios
    # First extract out the number of features available.
    # grouped both data together, and create a pandas dataframe from it.
    train1 = pd.DataFrame(trainingSet)
    label1 = pd.DataFrame(labelSet, columns=['LABEL'])
    grouped = pd.concat([train1, label1], axis=1)
    # fetch the number of classes
    (no_classes, demo) = grouped.groupby('LABEL').count()[[0]].shape
    #print grouped
    # loop through all features
    for j in range(dim2_T):
        # the variance of the feature j
        j_variance = np.var(trainingSet[:,j])
        j_mean = np.mean(trainingSet[:,j])
        j_summation = 0
        for k in range(no_classes):
            output = grouped.groupby('LABEL').count()[[j]]
            k_feature_count = output.iloc[k,0]
            # mean for class k of feature j
            output = grouped.groupby('LABEL').mean()[[j]]
            k_feature_mean = output.iloc[k,0]
            currentSum = k_feature_count * np.square((k_feature_mean - j_mean))
            j_summation = j_summation + currentSum
        fisher_ratios[j] = j_summation / np.square(j_variance)
    return fisher_ratios
