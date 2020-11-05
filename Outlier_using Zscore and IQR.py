#!/usr/bin/env python
# coding: utf-8

# # Detecting Outliers using Z score

# In[27]:


data=[10,12,3,6,5,89,67,56,110,89,507,120,45,39,54,29,18,12,178,210,365,143,100,99,156,98,48,78,125,306,450]


# In[28]:


outliers=[]
def detect_outlier(data):
    threshold=3
    mean=np.mean(data)
    std=np.std(data)
    
    for i in data:
        z_score=(i-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(i)
    return outliers        
    


# In[29]:


outliers_predict=detect_outlier(data)


# In[30]:


outliers_predict


# # Detecting Outliers using IQR(Inter Quartile Range(75%-25%))

# # Steps:
# 1. First arrange the data in sorted oder.
# 2. Calculate the first quantile(Q1) and third quantile(Q3).
# 3.find IQR(Q3-Q1).
# 4.Find Lower bound Q1-*1.5(IQR)
# 5.Find Upper bound Q3+*1.5(IQR)
# Anything outside this range is outlier

# In[31]:


sorted(data)


# In[32]:


Quantile1,Quantile3=np.percentile(data,[25,75])


# In[33]:


print(Quantile1,Quantile3)


# In[34]:


IQR=134-34


# In[35]:


IQR


# In[38]:


Lower_Bound=Quantile1-1.5*IQR


# In[39]:


Upper_Bound=Quantile3+1.5*IQR


# In[40]:


print(Lower_Bound,Upper_Bound)


# Anything outside this range(-116,284) is outlier
