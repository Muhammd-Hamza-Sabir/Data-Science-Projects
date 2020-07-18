#!/usr/bin/env python
# coding: utf-8

# ## 1. Modeling

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


train = pd.read_csv(r'./titanic/train_set.csv')
test = pd.read_csv(r'./titanic/test_set.csv')


# In[3]:


train.head()


# In[4]:


plt.figure(figsize=(15, 8))
sns.heatmap(train.corr())


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[6]:


cols = list(train.columns)[1:]
train_x = train.loc[:,cols]
train_y = train.loc[:,'Survived']


# In[7]:


train_x.shape, train_y.shape


# ### 1.1 Data Scaling & Splitting

# In[8]:


scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)


# In[9]:


scaled_train_x


# In[10]:


train_set, test_set, train_label, test_label = train_test_split(scaled_train_x, train_y, test_size=0.1, shuffle=True)


# In[11]:


train_set.shape, train_label.shape


# In[12]:


test_set.shape, test_label.shape


# ### 1.2 Data Modeling

# In[13]:


# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[19]:


models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,          GradientBoostingClassifier, SVC]


# In[34]:


scores = {}
for model in models:
    model = model()
    model.fit(X=train_set, y=train_label)
    predictions = model.predict(test_set)
    scores[model.__class__.__name__] = {'precision': precision_score(test_label, predictions, average='micro'),                                       'recall': recall_score(test_label, predictions, average='binary'),                                       'accuracy': accuracy_score(test_label, predictions)}


# In[35]:


scores


# In[36]:


models_score = pd.DataFrame(scores.values(), index=scores.keys())


# In[37]:


models_score


# <p>Highest Recall achieved so far is 72% by <b>BaggingClassifier</b>. The reason of not achieving good recall is the imbalance dataset. We will be handling this out in next steps. As we know data is imbalanced, that's why accuracy is not reliable even achieved about 80%.</p>

# ## 2. Imbalance Dataset

# In[45]:


train['Survived'].value_counts()


# ### 2.1 Under Sampling

# In[64]:


train_0 = train[train['Survived']==0].sample(n=263, replace=False)
train_1 = train[train['Survived']==1]


# In[67]:


train_ = pd.concat([train_0, train_1])
train_.reset_index(drop=True, inplace=True)


# In[69]:


train_.shape


# In[73]:


train_ = train_.sample(frac=1, random_state=42).reset_index(drop=True)


# In[75]:


train_.head()


# In[76]:


train_.tail()


# In[80]:


cols = list(train_.columns)[1:]
train_x = train_.loc[:,cols]
train_y = train_.loc[:,'Survived']


# In[83]:


train_x.shape, train_y.shape


# In[84]:


scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)


# In[85]:


train_set, test_set, train_label, test_label = train_test_split(scaled_train_x, train_y, test_size=0.1, shuffle=True)


# In[88]:


train_set.shape, train_label.shape


# In[89]:


test_set.shape, test_label.shape


# In[90]:


# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[91]:


models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,          GradientBoostingClassifier, SVC]


# In[92]:


scores = {}
for model in models:
    model = model()
    model.fit(X=train_set, y=train_label)
    predictions = model.predict(test_set)
    scores[model.__class__.__name__] = {'precision': precision_score(test_label, predictions, average='micro'),                                       'recall': recall_score(test_label, predictions, average='binary'),                                       'accuracy': accuracy_score(test_label, predictions)}


# In[93]:


scores


# In[94]:


models_score_2 = pd.DataFrame(scores.values(), index=scores.keys())


# In[95]:


models_score_2


# In[96]:


models_score['precision_under_sample'] = models_score_2['precision']
models_score['recall_under_sample'] = models_score_2['recall']
models_score['accuracy_under_sample'] = models_score_2['accuracy']


# In[97]:


models_score


# <p>Highest Recall achieved after under sampling is <b>74%</b> which is 2% more than earlier. We will apply oversampling method to check if there is possibility to improve the model performance.</p>

# In[ ]:




