#!/usr/bin/env python
# coding: utf-8

# Capturing NAN values with a new feature
# It works well if the data is not missing completely not at random

# In[1]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Age','Fare','Survived'])


# In[2]:


df.head()


# In[3]:


df.isna().sum()


# In[4]:


df['Age_NAN']=np.where(df.Age.isnull(),1,0)


# In[5]:


df.head()


# In[6]:


df['Age'].fillna(df.Age.median(),inplace=True)


# In[7]:


df.head(50)


# Advantages:
# 1.Easy to implement.
# 2.Capture the importance of missing values
# Disadvantages:
# 1.Creating Additional fetures(curs of Dimensionality)

# #End of Distribution Imputation

# In[8]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Age','Fare','Survived'])


# In[9]:


df.head()


# In[12]:


df.Age.hist(bins=50)


# In[24]:


# Value for End of Distribution (3 stdev away from mean )
extreme=df.Age.mean()+3*df.Age.std()
extreme


# In[15]:


sns.boxplot('Age',data=df)


# In[28]:


def impute_nan(df,variable,median,extreme):
    df[variable+'_enddistribution']=df[variable].fillna(extreme)
    df[variable].fillna(median,inplace=True)
    


# In[29]:


impute_nan(df,'Age',df.Age.median(),extreme)


# In[30]:


df.head()


# In[31]:


df.tail()


# In[34]:


df.Age.hist(bins=50)


# In[39]:


df.Age_enddistribution.hist(bins=50)


# In[37]:


sns.boxplot(df.Age_enddistribution,color='green')


# In[40]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['Age'].plot(kind='kde',ax=ax)
df.Age_enddistribution.plot(kind='kde',ax=ax,color= 'green')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[ ]:




