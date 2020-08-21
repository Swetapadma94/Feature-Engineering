#!/usr/bin/env python
# coding: utf-8

# Probability Ratio Encoding.

# In[1]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Cabin','Survived'])


# In[2]:


df.head()


# In[4]:


df['Cabin'].fillna("missing",inplace=True)


# In[6]:


df.head(20)


# In[7]:


df['Cabin'].unique()


# In[9]:


df['Cabin']=df['Cabin'].astype(str).str[0]


# In[10]:


df.head()


# In[12]:


df.Cabin.unique()


# In[15]:


#Probability Ratio Encoding.
#percentage of servived based on each cabin
prob_df=df.groupby(['Cabin'])['Survived'].mean()


# In[16]:


prob_df=pd.DataFrame(prob_df)


# In[17]:


prob_df


# In[18]:


prob_df['Died']=1-prob_df['Survived']


# In[19]:


prob_df


# In[20]:


prob_df['Probability_ratio']=prob_df['Survived']/prob_df['Died']


# In[21]:


prob_df


# In[23]:


probability=prob_df['Probability_ratio'].to_dict()


# In[24]:


df['Cabin_encoded']=df['Cabin'].map(probability)


# In[25]:


df.head()


# In[ ]:




