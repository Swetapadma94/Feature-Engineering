#!/usr/bin/env python
# coding: utf-8

# ## Random Sample imputation
# Aim: Random sample imputation consits of taking random observations from the data set and we use this observation to replace nan values.
# 
# When should we use:
# it assumes that the data is missing completely at random(MCAR)

# In[31]:


data=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Age','Fare','Survived'])


# In[32]:


data.head()


# In[33]:


data.isnull().sum()


# In[34]:


data.isnull().mean()


# In[35]:


data['Age'].isnull().sum()


# In[36]:


data['Age'].dropna().sample(data['Age'].isnull().sum(),random_state=0)


# In[37]:


data[data['Age'].isnull()].index


# In[43]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)
    df[variable+'_random']=df[variable]
    # it will have the random sample to fill the Na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas needs to have same index to merge data set
    
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample


# In[44]:


median=data.Age.median()
median


# In[45]:


impute_nan(data,'Age',median)


# In[46]:


data.head()


# In[47]:


data.tail()


# In[48]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


fig=plt.figure()
ax=fig.add_subplot(111)
data['Age'].plot(kind='kde',ax=ax)
data.Age_median.plot(kind='kde',ax=ax,color= 'red')
data.Age_random.plot(kind='kde',ax=ax,color= 'green')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# ####Advantages and Disadvantages
# 1. Easy to implement
# 2.There is no distortion in variance
# #Disadvantages
# 1.In every situation randomness won't work.
