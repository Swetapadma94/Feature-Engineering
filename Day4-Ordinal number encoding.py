#!/usr/bin/env python
# coding: utf-8
Ordinal Number Encoding:>Ranking  
Ordinal->Grading System(A-1,B-2,C-3)


# In[22]:


import datetime as dt


# In[2]:


today=datetime.datetime.today()


# In[3]:


today


# In[ ]:


today-


# In[6]:


today-datetime.timedelta(3)


# In[9]:


## List Comprehension
days=[today-datetime.timedelta(x) for x in range (0,15)]


# In[10]:


days


# In[12]:


data=pd.DataFrame(days)
data.columns=["Day"]


# In[13]:


data.head()


# In[28]:


data['Day'].dt.day


# In[30]:


data["weekday"]=data['Day'].dt.day_name()


# In[31]:


data.head()


# In[32]:


dictionary={'Monday':1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}


# In[33]:


dictionary


# In[37]:


data["weekday_ordinal"]=data['weekday'].map(dictionary)


# In[38]:


data.head()


# --->Count or Frequncy Encoding

# In[39]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\adult.csv',encoding='latin1')


# In[40]:


df.head()


# In[42]:


cols=[1,3,5,6,7,8,9,10]


# In[46]:


df.columns


# In[49]:


df=df[['workclass','education','marital-status','occupation','relationship','race','gender','native-country']]


# In[50]:


df.head()


# In[53]:


for col in df.columns[:]:
    print(col,":",len(df[col].unique()),'labels')


# In[55]:


df["native-country"].value_counts().to_dict()


# In[56]:


country_map=df["native-country"].value_counts().to_dict()


# In[57]:


df["native-country"]=df["native-country"].map(country_map)


# In[60]:


df.head(30)


# In[62]:


mp=df["gender"].value_counts().to_dict()


# In[63]:


df["gender"]=df["gender"].map(mp)


# In[64]:


df.head()


# #Advantages
# 1. Easy to implement.
# 
# 2.Faster method to excute.
# 
# 3. Not increasing dimensions.
# 
# #### Disadvantages
# 
# 1.If two categories have same no.of frequency.
# --> it will provide the same weight.

# -->Target-Guided ordinal Encoding:
# 1. ordering the labels acording to the target.
# 2.Replace the labels by the joint probability of being 1 or 0.

# In[86]:


data=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Cabin','Survived'])


# In[87]:


data.head()


# In[88]:


data['Cabin'].fillna("missing",inplace=True)


# In[89]:


data.head(10)


# In[90]:


data['Cabin']=data['Cabin'].astype(str).str[0]


# In[91]:


data


# In[92]:


data.groupby(['Cabin'])['Survived'].mean()


# In[93]:


ordinal_labels=data.groupby(['Cabin'])['Survived'].mean().sort_values().index
ordinal_labels


# In[94]:


ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}


# In[95]:


ordinal_labels2


# In[96]:


data['Cabin_ordinal']=data['Cabin'].map(ordinal_labels2)


# In[97]:


data


# Mean Encoding(provides monotonic relationsg=hip)
# It may lead to over-fitting.

# In[100]:


mp=data.groupby(['Cabin'])['Survived'].mean().to_dict()
mp


# In[101]:


data['Cabin_mean_ordinal']=data['Cabin'].map(mp)


# In[102]:


data


# In[ ]:




