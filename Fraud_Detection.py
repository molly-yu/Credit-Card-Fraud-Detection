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


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Load dataset from csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[7]:


# Explore dataset
print(data.columns)


# In[8]:


print(data.shape) # num of transactions with columns


# In[9]:


print(data.describe()) # info abt each column


# In[10]:


# sample 10% of data
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)


# In[11]:


# plot histogram of each parameter
data.hist(figsize = (20,20))
plt.show()


# In[12]:


# if we look at classes, numbers near 0 are valid, numbers near 1 are fraudulent
# determine number of fraudulent transactions
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid)) # fraction of fraud : non-fraud
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Cases: {}'.format(len(Valid)))


# In[13]:


# Correlation matrix (any correlations in dataset)
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[14]:


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


# In[ ]:




