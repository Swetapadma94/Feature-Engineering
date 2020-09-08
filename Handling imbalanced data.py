#!/usr/bin/env python
# coding: utf-8

# # Under Sampling

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[2]:


data=pd.read_csv(r'E:\Krish naik\python code\Feature-Engineering\handling imbalanced data\creditcard.csv',encoding='latin1')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[14]:


X=data.iloc[:,:-1]


# In[15]:


X


# In[19]:


Y=data.iloc[:,-1]


# In[20]:


Y


# In[23]:


# Define a random state 
state = np.random.RandomState(42)
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# # Exploratory Data Analysis

# In[27]:


data.isnull().values.any()


# In[28]:


count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[29]:


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[30]:


print(fraud.shape,normal.shape)


# In[31]:


from imblearn.under_sampling import NearMiss


# In[34]:



# Implementing Undersampling for Handling Imbalanced 
nm = NearMiss()
X_res,y_res=nm.fit_sample(X,Y)


# In[35]:


X_res


# In[36]:


y_res


# In[37]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# # Over Sampling

# In[38]:


from imblearn.combine import SMOTETomek


# In[39]:


smk=SMOTETomek(random_state=42)


# In[40]:


X_res,y_res=smk.fit_sample(X,Y)


# In[41]:


X_res.shape


# In[42]:


y_res.shape


# In[43]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[52]:


## RandomOverSampler to handle imbalanced data

from imblearn.over_sampling import RandomOverSampler


# In[55]:


##os =  RandomOverSampler(ratio=0.5-50%,1-100%)
os =  RandomOverSampler()


# In[59]:



X_train_res,y_train_res =os.fit_sample(X,Y)


# In[60]:


X_train_res.shape,y_train_res.shape


# In[61]:



print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))


# In[ ]:



# In this example I use SMOTETomek which is a method of imblearn. SMOTETomek is a hybrid method
# which uses an under sampling method (Tomek) in with an over sampling method (SMOTE).
os_us = SMOTETomek(ratio=0.5)

X_train_res1, y_train_res1 = os_us.fit_sample(X, Y)


# In[ ]:


X_train_res1.shape,y_train_res1.shape


# In[ ]:


print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res1)))


# In[ ]:




