#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Load dataset from csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[5]:


# Explore dataset
print(data.columns)


# In[6]:


print(data.shape) # num of transactions with columns


# In[9]:


print(data.describe()) # info abt each column


# In[10]:


# sample 10% of data
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)


# In[12]:


# plot histogram of each parameter
data.hist(figsize = (20,20))
plt.show()


# In[13]:


# if we look at classes, numbers near 0 are valid, numbers near 1 are fraudulent
# determine number of fraudulent transactions
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid)) # fraction of fraud : non-fraud
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Cases: {}'.format(len(Valid)))


# In[14]:


# Correlation matrix (any correlations in dataset)
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[15]:


# Get columns from DataFrame
columns = data.columns.tolist()

# Filter columns to remove specific undesired data
# We are removing Class since it would give the answers away
columns = [c for c in columns if c not in ["Class"]]

# Unsupervised learning (anomaly detection) so no labels
# Store variable we will predict
target = "Class"

X = data[columns] # everything except class
Y = data[target] # class

print(X.shape)
print(Y.shape)


# In[16]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest # isolates features by randomly selecting features and splitting them
from sklearn.neighbors import LocalOutlierFactor # find anomaly score of each sample, based on its neighbors

# define a random state
state = 1

# define outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                       contamination = outlier_fraction, 
                                       random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                              contamination = outlier_fraction)
}


# In[20]:


# fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit data, tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X) # -1 for outlier, 1 for non-outlier
        
    # reshape prediction values from 0 to 1 
    y_pred[y_pred == 1] = 0 # valid
    y_pred[y_pred == -1] = 1 # fraud
    
    n_errors = (y_pred != Y).sum()
    
    # run classification metrics
    print('{}:{}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred)) # compare target Y to the predicted y


# In[21]:


# recall: false negatives, precision: false positives, f1-score: combination
# Isolation forest shown to have better results than Local outlier factor


# In[ ]:




