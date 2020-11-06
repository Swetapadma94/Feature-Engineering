#!/usr/bin/env python
# coding: utf-8

# In[2]:


iris=sns.load_dataset('iris')


# In[3]:


iris.head()


# In[5]:


sns.pairplot(data=iris,hue='species')


# In[6]:


iris.shape


# Univariate Analysis
# 

# In[7]:


iris.columns


# In[12]:


iris_setosa=iris.loc[iris['species']=='setosa']
iris_setosa


# In[17]:


iris_virginica=iris.loc[iris['species']=='virginica']
iris_virginica


# In[15]:


iris['species']


# In[19]:


iris_versicolor=iris.loc[iris['species']=='versicolor']
iris_versicolor


# In[41]:


plt.plot(iris_setosa['petal_length'])


# In[30]:


plt.plot(iris_setosa['petal_length'],np.zeros_like(iris_setosa['petal_length']),'o')
plt.plot(iris_versicolor['petal_length'],np.zeros_like(iris_versicolor['petal_length']),'o')
plt.plot(iris_virginica['petal_length'],np.zeros_like(iris_virginica['petal_length']),'o')
plt.show()
# here y axis=0,x=petal_length


# #Bivariate

# In[33]:


sns.FacetGrid(iris,hue='species',size=5).map(plt.scatter,'sepal_width','sepal_length').add_legend()


# In[34]:


sns.FacetGrid(iris,hue='species',size=5).map(plt.scatter,'sepal_width','petal_length').add_legend()


# In[38]:


sns.pairplot(data=iris,hue='species').add_legend()


# In[40]:


sns.pairplot(data=iris,size=5)


# In[ ]:




