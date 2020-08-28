#!/usr/bin/env python
# coding: utf-8

# In[2]:


credit=pd.read_csv(r'E:\Krish naik\kaggle dataset\creditcard.csv',encoding='latin1')


# In[3]:


credit.head()


# In[4]:


credit.shape


# In[6]:


credit.isna().sum()


# In[8]:


credit['Class'].value_counts()


# In[25]:


X=credit.drop("Class",axis=1)
y=credit.Class


# In[26]:


X.shape


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


# In[30]:


from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV


# In[31]:


grid={'C':10.0**np.arange(-2,3)}
grid


# In[49]:



log_class = LogisticRegression()

# Hyperparameters
grid = {'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}

# KFold
cv = KFold(n_splits=5,random_state=None,shuffle=False)


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)


# In[51]:



log_clf=GridSearchCV(log_class, grid, cv=cv, n_jobs=-1, scoring='f1_macro')
log_clf.fit(X_train,y_train)


# In[53]:


y_pred=log_clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[62]:


X_train.shape


# In[63]:


y_train.value_counts()


# In[64]:


class_weight=dict({0:1,1:100})


# In[54]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


classifier=RandomForestClassifier(class_weight=class_weight)


# In[69]:


classifier.fit(X_train,y_train)


# In[70]:


y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[71]:


classifier=RandomForestClassifier()


# In[72]:


classifier.fit(X_train,y_train)


# In[74]:


y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### Under Sampling##
# 0      1
# 
# 10000   100
# Reduce the points of the maximum labels.

# In[73]:


y_train.value_counts()


# In[77]:


from collections import Counter
Counter(y_train)


# In[80]:


from imblearn.under_sampling import NearMiss

ns=NearMiss(0.8)

X_train_ns,y_train_ns=ns.fit_sample(X_train,y_train)

print("The number of classes before fit {} ".format(Counter(y_train)))
print("The number of classes after fit {} ".format(Counter(y_train_ns)))


# In[84]:


classifier=RandomForestClassifier()
classifier.fit(X_train_ns,y_train_ns)


# In[86]:


y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[88]:


### Over Sampling##
from imblearn.over_sampling import RandomOverSampler

os=RandomOverSampler(0.5)

X_train_os,y_train_os=os.fit_sample(X_train,y_train)

print("The number of classes before fit {} ".format(Counter(y_train)))
print("The number of classes after fit {} ".format(Counter(y_train_ns)))


# In[89]:


classifier=RandomForestClassifier()
classifier.fit(X_train_os,y_train_os)


# In[90]:


y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ###SMOTETomek

# In[91]:


from imblearn.combine import SMOTETomek


# In[92]:


st=SMOTETomek(0.5)

X_train_st,y_train_st=st.fit_sample(X_train,y_train)

print("The number of classes before fit {} ".format(Counter(y_train)))
print("The number of classes after fit {} ".format(Counter(y_train_ns)))


# In[94]:


classifier=RandomForestClassifier()
classifier.fit(X_train_st,y_train_st)


# In[95]:


y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ##Ensemble Techniques

# In[96]:


from imblearn.ensemble import EasyEnsembleClassifier


# In[97]:


easy = EasyEnsembleClassifier()
easy.fit(X_train,y_train)


# In[98]:


y_pred = easy.predict(X_test)

print('Confustion Matrix : \n\n', confusion_matrix(y_test,y_pred))
print('\n Accuracy Score : ',   accuracy_score(y_test,y_pred))
print('\n Classification Report : \n \n', classification_report(y_test,y_pred))


# In[ ]:




