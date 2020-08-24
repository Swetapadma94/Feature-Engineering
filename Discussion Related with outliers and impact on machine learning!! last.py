#!/usr/bin/env python
# coding: utf-8

# ### Discussion Related with outliers and impact on machine learning!!
# To find out the suspicious activity.
# Outliers:some data points are different from other data points.
# Wheather we should remove outliers?
# If it impacts our model acuuracy.
# Depending on problem defination.
# Sales forecasting outlier should be kept 
# ### which machine learning models are sensitive to the outlier?
# 
# 1.Naive Bayes-Not Sensitive to outliers
# 2.Linear Reegression-Yes Sensitive/impact to outliers
# 3.SVM-Not Sensitive to outliers
# 4.Logistic Regression-Yes Sensitive/impact to outliers
# 5.Decision Tree Regressor or Classifier-Not Sensitive to outliers
# 6.Ensemble Technique(XGBoost,Random Forest,Gradient-Boosting)-Not Sensitive to outliers
# 7.KNN-Not Sensitive to outliers
# 8.kmeans-Yes Sensitive/impact to outliers
# 9.Hieararchecal-Yes Sensitive/impact to outliers
# 10.PCA-very very  Sensitive to outliers(before applying PCA we should remove all outliers)
# 11.Neural Network-yes Sensitive to outliers
# 12.LDA(Linear Discriminant Analysis)-yes Sensitive to outliers
# 13.DB SCAN-Yes Sensitive/impact to outliers

# In[1]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1')


# In[2]:


df.head()


# In[4]:


df['Age'].isnull().sum()


# In[6]:


sns.distplot(df.Age.dropna())


# In[7]:


sns.distplot(df.Age.fillna(100))


# In[8]:


sns.distplot(df.Age.fillna(df.Age.mean()))


# In[9]:


df.Age.hist(bins=50)


# In[10]:


#This data is distributed normally 
df.boxplot(column='Age')


# In[11]:


df.Age.describe()


# In[14]:


##### ASSUMING age follows gussian distribution we will calculate the boundaries which differentiate the outlier.
upper=df.Age.mean()+3*df.Age.std()
upper


# In[15]:


lower=df.Age.mean()-3*df.Age.std()
lower


# In[16]:


df.Age.mean()


# If data is normally distributed we should consider the upper and lower range(+-3* stdev) (points which are away of 3* stdev are outliers)

# In[27]:


##### If feature is skeweed
df.Fare.hist(bins=50)


# In[28]:


df.boxplot(column='Fare')


# In[29]:


df.Fare.describe()


# In[31]:


IQR=df.Fare.quantile(0.75)-df.Fare.quantile(0.25)
IQR


# In[32]:


upper=df.Fare.mean()+3*df.Age.std()
print(upper)
lower=df.Fare.mean()-3*df.Age.std()
print(lower)


# In[33]:


lower_bridge=df.Fare.quantile(0.25)-(IQR*1.5)
print(lower_bridge)
upper_bridge=df.Fare.quantile(0.75)+(IQR*1.5)
print(upper_bridge)


# In[35]:


extreme_lower_bridge=df.Fare.quantile(0.25)-(IQR*3)
print(extreme_lower_bridge)
extreme_upper_bridge=df.Fare.quantile(0.75)+(IQR*3)
print(extreme_upper_bridge)


# Conclusion-When your data is skewed we should consider Extreme limit

# # How to Handle outliers

# In[36]:


data=df.copy()


# In[50]:


data.loc[data['Age']>=73,'Age']=73


# In[51]:


data.tail(100)


# In[46]:


data.loc[data['Fare']>=100,'Fare']=100


# In[47]:


data.head(50)


# In[48]:


data.Fare.hist(bins=50)


# In[52]:


data.Age.hist(bins=50)


# ### Applying Machine Learning Algorithm

# In[54]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[['Age','Fare']].fillna(0),data['Survived'],test_size=0.3)


# In[55]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred1=classifier.predict_proba(X_test)


# In[61]:


from sklearn.metrics import accuracy_score,roc_auc_score
print(accuracy_score(y_test,y_pred))
print('roc_auc_score: {}'.format(roc_auc_score(y_test,y_pred1[:,1])))


# In[63]:


# Logistic Regression
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred1=classifier.predict_proba(X_test)


# In[64]:


from sklearn.metrics import accuracy_score,roc_auc_score
print(accuracy_score(y_test,y_pred))
print('roc_auc_score: {}'.format(roc_auc_score(y_test,y_pred1[:,1])))


# In[ ]:




