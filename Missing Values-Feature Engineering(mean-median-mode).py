#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df[df.Embarked.isnull()]


# In[6]:


df['cabin_null']=np.where(df['Cabin'].isnull(),1,0)


# In[7]:


df.head()


# In[8]:


df.Cabin.isnull().mean()


# # MISSING DATA NOT AT RANDOM(MDNAR)
# There is some relationship

# In[9]:


df.groupby(['Survived'])['cabin_null'].mean()


# In[10]:


df.groupby(['Survived'])['Age'].mean()


# ### Missing  at Random(MAR)

# #Technique to handle missing values
# 
# 1.Mean/Median/mode replacement
# 2.Dropping
# 3.Random Sample
# 4.Capturing NaN with a new feature
# 5.End of Distribution
# 6.Arbitary Imputation
# 7.frequent categories Imputation
# 
# 

# # Mean/Median/Mode imputation
# When should we apply:
# Mean/Median/mod has the assumption that the data are missing completely(MCAR-No relationship between missing data and other variable),at random
# solve replace NAN with most occurance data

# In[11]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Age','Fare','Survived'])


# In[12]:


df.head()


# In[13]:


df.isnull().mean()


# In[14]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)
    


# In[15]:


median=df['Age'].median()
median


# In[16]:


impute_nan(df,'Age',median)


# In[17]:


df.head()


# In[18]:


df['Age']=df['Age'].fillna(median)


# In[19]:



df.Age.std()


# In[20]:


df.Age_median.std()


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['Age'].plot(kind='kde',ax=ax)
df.Age_median.plot(kind='kde',ax=ax,color='red')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# ## advantages and disadvantages with mean/median/mode imputation
# #Advantages
# 1.Easy to implement
# 2.Robust to outlier
# 3.Faster way to obtain the complete dataset

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




