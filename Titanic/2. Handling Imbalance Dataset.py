#!/usr/bin/env python
# coding: utf-8

# ## 1. Modeling

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[19]:


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

# In[21]:


scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)


# In[22]:


scaled_train_x


# In[23]:


train_set, test_set, train_label, test_label = train_test_split(scaled_train_x, train_y, test_size=0.1, shuffle=True)


# In[24]:


train_set.shape, train_label.shape


# In[25]:


test_set.shape, test_label.shape


# ### 1.2 Data Modeling

# In[26]:


# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[27]:


models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,          GradientBoostingClassifier, SVC]


# In[46]:


scores = {}
for model in models:
    model = model(random_state=50)
    model.fit(X=train_set, y=train_label)
    predictions = model.predict(test_set)
    scores[model.__class__.__name__] = {'precision': precision_score(test_label, predictions, average='micro'),                                       'recall': recall_score(test_label, predictions, average='binary'),                                       'accuracy': accuracy_score(test_label, predictions)}


# In[47]:


scores


# In[48]:


models_score = pd.DataFrame(scores.values(), index=scores.keys())


# In[49]:


models_score


# <p>Highest Recall achieved so far is 75% by <b>LogisticRegression & AdaBoostClassifier</b>. The reason of not achieving good recall is the imbalance dataset. We will be handling this out in next steps. As we know data is imbalanced, that's why accuracy is not reliable even achieved about 80%.</p>

# ## 2. Imbalance Dataset

# In[50]:


train['Survived'].value_counts()


# ### 2.1 Random Under Sampling

# In[51]:


train_0 = train[train['Survived']==0].sample(n=263, replace=False, random_state=42)
train_1 = train[train['Survived']==1]


# In[52]:


train_ = pd.concat([train_0, train_1])
train_.reset_index(drop=True, inplace=True)


# In[53]:


train_.shape


# In[54]:


train_ = train_.sample(frac=1, random_state=42).reset_index(drop=True)


# In[55]:


train_.head()


# In[56]:


train_.tail()


# In[57]:


cols = list(train_.columns)[1:]
train_x = train_.loc[:,cols]
train_y = train_.loc[:,'Survived']


# In[58]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('Survived', data=train_)
ax.set_title('Counter for each class')
plt.show()


# In[59]:


train_x.shape, train_y.shape


# In[60]:


scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)


# In[61]:


train_set, test_set, train_label, test_label = train_test_split(scaled_train_x, train_y, test_size=0.1, shuffle=True)


# In[62]:


train_set.shape, train_label.shape


# In[63]:


test_set.shape, test_label.shape


# In[64]:


# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[65]:


models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,          GradientBoostingClassifier, SVC]


# In[68]:


scores = {}
for model in models:
    model = model(random_state=50)
    model.fit(X=train_set, y=train_label)
    predictions = model.predict(test_set)
    scores[model.__class__.__name__] = {'precision': precision_score(test_label, predictions, average='micro'),                                       'recall': recall_score(test_label, predictions, average='binary'),                                       'accuracy': accuracy_score(test_label, predictions)}


# In[69]:


scores


# In[70]:


models_score_2 = pd.DataFrame(scores.values(), index=scores.keys())


# In[71]:


models_score_2


# In[72]:


models_score['precision_under_sample'] = models_score_2['precision']
models_score['recall_under_sample'] = models_score_2['recall']
models_score['accuracy_under_sample'] = models_score_2['accuracy']


# In[73]:


models_score


# <p>Highest Recall achieved after under sampling is <b>81%</b> which is 6% more than earlier. We will apply oversampling method to check if there is possibility to improve the model performance.</p>

# ### 2.2 Random Over Sampling

# In[74]:


from collections import Counter
from imblearn.over_sampling import RandomOverSampler


# In[85]:


Counter(train["Survived"])


# In[86]:


OverSampler = RandomOverSampler(sampling_strategy='minority')


# In[88]:


train_set


# In[92]:


train_set_over, train_label_over = OverSampler.fit_sample(train.iloc[:,1:].values, train["Survived"].values)


# In[93]:


np.shape(train_set_over), np.shape(train_label_over)


# In[94]:


scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_set_over)


# In[96]:


train_set, test_set, train_label, test_label = train_test_split(train_set_over, train_label_over, test_size=0.1, shuffle=True)


# In[97]:


train_set.shape, train_label.shape


# In[98]:


test_set.shape, test_label.shape


# In[99]:


# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[100]:


models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,          GradientBoostingClassifier, SVC]


# In[101]:


scores = {}
for model in models:
    model = model(random_state=50)
    model.fit(X=train_set, y=train_label)
    predictions = model.predict(test_set)
    scores[model.__class__.__name__] = {'precision': precision_score(test_label, predictions, average='micro'),                                       'recall': recall_score(test_label, predictions, average='binary'),                                       'accuracy': accuracy_score(test_label, predictions)}


# In[102]:


scores


# In[103]:


models_score_3 = pd.DataFrame(scores.values(), index=scores.keys())


# In[104]:


models_score_3


# In[105]:


models_score['precision_over_sample'] = models_score_3['precision']
models_score['recall_over_sample'] = models_score_3['recall']
models_score['accuracy_over_sample'] = models_score_3['accuracy']


# In[106]:


models_score


# <p>Highest Recall achieved after under sampling is <b>90%</b> which is 9% more than earlier. Hence, shown that oversampling performs better than undersampling.</p>
