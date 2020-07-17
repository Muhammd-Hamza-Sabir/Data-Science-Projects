#!/usr/bin/env python
# coding: utf-8

# ## 1. Modeling

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


train = pd.read_csv(r'./titanic/train_set.csv')
test = pd.read_csv(r'./titanic/test_set.csv')


# In[7]:


train.head()


# In[9]:


plt.figure(figsize=(15, 8))
sns.heatmap(train.corr())


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[34]:


cols = list(train.columns)[1:]
train_x = train.loc[:,cols]
train_y = train.loc[:,'Survived']


# In[37]:


train_x.shape, train_y.shape


# ### 1.1 Data Scaling & Splitting

# In[38]:


scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)


# In[39]:


scaled_train_x


# In[42]:


train_set, test_set, train_label, test_label = train_test_split(scaled_train_x, train_y, test_size=0.1, shuffle=True)


# In[44]:


train_set.shape, train_label.shape


# In[45]:


test_set.shape, test_label.shape


# In[53]:


# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ### 1.2 Linear Regression

# In[54]:


lr_model = LogisticRegression()
lr_model.fit(train_set, train_label)
lr_predictions = lr_model.predict(test_set)


# In[67]:


# accuracy, precision & recall for Linear Regreesion
lr_precision = precision_score(test_label, lr_predictions, average='micro')
lr_recall = recall_score(test_label, lr_predictions, average='binary')
lr_accuracy = accuracy_score(test_label, lr_predictions)
print("Linar Regression Precision: ",lr_precision)
print("Linar Regression Recall: ",lr_recall)
print("Linar Regression Accuracy: ",lr_accuracy)


# ### 1.3 Decision Tree

# In[62]:


dt_model = DecisionTreeClassifier()
dt_model.fit(train_set, train_label)
dt_predictions = dt_model.predict(test_set)


# In[69]:


# accuracy, precision & recall for Decision Tree
dt_precision = precision_score(test_label, dt_predictions, average='micro')
dt_recall = recall_score(test_label, dt_predictions, average='binary')
dt_accuracy = accuracy_score(test_label, dt_predictions)
print("Decision Tree Precision: ",dt_precision)
print("Decision Tree Recall: ",dt_recall)
print("Decision Tree Accuracy: ",dt_accuracy)


# ### 1.4 Random Forest

# In[76]:


rf_model = RandomForestClassifier()
rf_model.fit(train_set, train_label)
rf_predictions = rf_model.predict(test_set)


# In[82]:


# accuracy, precision & recall for Random Forest
rf_precision = precision_score(test_label, rf_predictions, average='micro')
rf_recall = recall_score(test_label, rf_predictions, average='binary')
rf_accuracy = accuracy_score(test_label, rf_predictions)
print("Random Forest Precision: ",rf_precision)
print("Random Forest Recall: ",rf_recall)
print("Random Forest Accuracy: ",rf_accuracy)


# ### 1.5 AdaBoost

# In[79]:


AdaBoost_model = AdaBoostClassifier()
AdaBoost_model.fit(train_set, train_label)
AdaBoost_predictions = AdaBoost_model.predict(test_set)


# In[81]:


# accuracy, precision & recall for AdaBoost
adaBoost_precision = precision_score(test_label, AdaBoost_predictions, average='micro')
adaBoost_recall = recall_score(test_label, AdaBoost_predictions, average='binary')
adaBoost_accuracy = accuracy_score(test_label, AdaBoost_predictions)
print("AdaBoost Precision: ",adaBoost_precision)
print("AdaBoost Recall: ",adaBoost_recall)
print("AdaBoost Accuracy: ",adaBoost_accuracy)


# ### 1.6 Gradient Boosting

# In[85]:


GB_model = GradientBoostingClassifier()
GB_model.fit(train_set, train_label)
GB_predictions = GB_model.predict(test_set)


# In[86]:


# accuracy, precision & recall for Gradient Boosting
gb_precision = precision_score(test_label, GB_predictions, average='micro')
gb_recall = recall_score(test_label, GB_predictions, average='binary')
gb_accuracy = accuracy_score(test_label, GB_predictions)
print("Gradient Boosting Precision: ",gb_precision)
print("Gradient Boosting Recall: ",gb_recall)
print("Gradient Boosting Accuracy: ",gb_accuracy)


# ### 1.7 Support Vector Machine

# In[88]:


svc_model = SVC()
svc_model.fit(train_set, train_label)
svc_predictions = svc_model.predict(test_set)


# In[89]:


# accuracy, precision & recall for Support Vector Machine
svc_precision = precision_score(test_label, svc_predictions, average='micro')
svc_recall = recall_score(test_label, svc_predictions, average='binary')
svc_accuracy = accuracy_score(test_label, svc_predictions)
print("Support Vector Machine Precision: ",svc_precision)
print("Support Vector Machine Recall: ",svc_recall)
print("Support Vector Machine Accuracy: ",svc_accuracy)


# ### 1.8 Bagging

# In[90]:


bag_model = BaggingClassifier()
bag_model.fit(train_set, train_label)
bag_predictions = bag_model.predict(test_set)


# In[91]:


# accuracy, precision & recall for Bagging Classifier
bag_precision = precision_score(test_label, bag_predictions, average='micro')
bag_recall = recall_score(test_label, bag_predictions, average='binary')
bag_accuracy = accuracy_score(test_label, bag_predictions)
print("Bagging Classifier Precision: ",bag_precision)
print("Bagging Classifier Recall: ",bag_recall)
print("Bagging Classifier Accuracy: ",bag_accuracy)


# <p>Highest Recall achieved so far is 65% by <b>Linear Regression</b> & <b>Support Vector Machine</b>. The reason of not achieving good recall is the imbalance dataset. We will be handling this out in next steps.</p>

# In[ ]:




