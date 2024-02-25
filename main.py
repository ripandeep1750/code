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
''' TSNE is an abbreviation for t-distributed Stochastic Neighbor Embedding. TSNE is a dimensionality reduction technique that aims to 
capture the local relationships between data points in a lower-dimensional space. It is often used for visualizing high-dimensional data 
in a way that reveals underlying structures and patterns. '''

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
