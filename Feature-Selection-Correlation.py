#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data=load_boston()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['MEDV']=data.target
df.head()


# In[5]:


data.feature_names


# In[6]:


X=df.drop("MEDV",axis=1)
y=df.MEDV


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape


# In[10]:



import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[11]:


X_train.corr()


# In[12]:


# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[13]:


corr_features=correlation(X_train,0.7)
len(set(corr_features))


# In[14]:


corr_features


# In[15]:



X_train.drop(corr_features,axis=1)
X_test.drop(corr_features,axis=1)


# In[25]:


data=pd.read_csv(r"E:\Krish naik\Feature-Selection\standard.csv",encoding='latin1',nrows=10000)


# In[26]:


data.head()


# In[33]:


X=data.iloc[:,:-1]
X.head()
y=data.iloc[:,-1]
y


# In[35]:



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[36]:



import seaborn as sns
#Using Pearson Correlation
corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)


# In[37]:


corr_features = correlation(X_train, 0.9)
len(set(corr_features))


# In[38]:


corr_features


# In[39]:


X_train.drop(corr_features,axis=1)


# In[ ]:




