#!/usr/bin/env python
# coding: utf-8

# In[7]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Mercedes\train.csv',encoding='latin1',usecols=['X1','X2'])


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df.X1.unique()


# In[11]:


pd.get_dummies(df).shape


# In[12]:


len(df['X1'].unique())


# In[13]:


len(df['X2'].unique())


# In[15]:


for col in df.columns[0:]:
    print(col, ":", len(df[col].unique()), 'labels')
    


# In[18]:


df_frequency_map=df.X2.value_counts().to_dict()


# In[19]:


df_frequency_map


# In[20]:


df.X2=df.X2.map(df_frequency_map)


# In[21]:


df.head()


# In[17]:


df.X1.value_counts().to_dict()


# In[22]:


df_x1_map=df.X1.value_counts().to_dict()


# In[23]:


df_x1_map


# In[24]:


df.X1


# In[25]:


df.X1=df.X1.map(df_x1_map)


# In[26]:


df.head()


# Ordinal categorical/Label encoding

# In[30]:


import datetime
df_base=datetime.datetime.today()


# In[31]:


df_date_list=[df_base - datetime.timedelta(days=x) for x in range(0,20)]
df=pd.DataFrame(df_date_list)
df.columns=['day']
df


# In[39]:


df['day_of_week']=df['day'].datetime.weekday_name()


# In[36]:


weekday_map={'Monday':1,
            'Tuesday':2,
            'Wednesday':3,
            'Thurdday':4,
            'Friday':5,
            'Saturday':6,
            'Sunday':7}


# In[ ]:


df[day_ordinal]=df['day_of_week'].map(weekday_map)

