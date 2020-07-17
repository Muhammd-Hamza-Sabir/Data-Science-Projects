#!/usr/bin/env python
# coding: utf-8

# In[312]:


# import libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re


# In[313]:


train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')


# In[314]:


full_data = [train, test]


# In[315]:


train.head()


# In[316]:


train.shape, test.shape


# In[317]:


print('Percentage of class Survived: ', round(len(train.loc[train['Survived']==1,:])/len(train),3))
print('Percentage of class not Survived: ', round(len(train.loc[train['Survived']==0,:])/len(train),3))


# In[318]:


# Above percentages clearly show that data is imbalanced, we will be handling this later
# let's plot them
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.countplot(train['Survived'])
ax.set_title('Survived vs. Not Survived')
plt.show()


# ### 1. Analysis of Categorical Variables

# #### 1.1 Pclass vs Survived

# In[319]:


train.loc[:,['Pclass', 'Survived']].groupby('Pclass').mean()


# In[320]:


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.countplot('Pclass', hue='Survived', data=train)
ax.set_title('Pclass along with Survived')
plt.show()


# <p>Highest survival rate for those belong to Pclass of 1</p>

# In[321]:


# let's visualize how survival rate varies as per the Pclass
fig = sns.factorplot('Pclass', 'Survived', data=train, size=5.5, aspect=1.5)
fig.fig.suptitle('Survival rate within each Pclass')
plt.show()


# #### 1.2 Sex vs Survived

# In[322]:


train['Sex'].isnull().sum()


# In[323]:


train.loc[:,['Sex', 'Survived']].groupby('Sex').sum()


# In[324]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('Sex', hue='Survived', data=train)
ax.set_title('Sex vs Survival Rate')
plt.show()


# In[325]:


# It's clearly seen that females were given priority during rescue. Let's plot factorplot
fig = sns.factorplot('Sex', 'Survived', data=train, size=5.5, aspect=1.5)
fig.fig.suptitle('Survival Rate with Sex')
plt.show()


# #### 1.3 Embarked vs Survived

# In[326]:


train['Embarked'].isnull().sum()


# In[327]:


train['Embarked'].value_counts()


# In[328]:


train.loc[train['Embarked'].isnull(),['Embarked']] = 'S'


# In[329]:


train['Embarked'].isnull().sum()


# In[330]:


train.loc[:,['Embarked', 'Survived']].groupby('Embarked').mean()


# In[331]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('Embarked', hue='Survived', data=train)
ax.set_title('Embarked vs Survived')
plt.show()


# In[332]:


fig = sns.factorplot('Embarked', 'Survived', data=train, size=5.5, aspect=1.5)
fig.fig.suptitle('Survival Rate with Embarked')
plt.show()


# <p>Highest survival rate for those with Embarked=C while least for Embarked=S</p>

# #### 1.4 Cabin vs Survived

# In[333]:


train['Cabin'].isnull().sum()


# In[334]:


print('Percentage of Null values in Cabin: {}. That;s why Cabin will not be considered for further analysis and modeling.'.format(round(train['Cabin'].isnull().sum()/len(train), 3)*100))


# In[335]:


for dataset in full_data:
    dataset.drop('Cabin', axis=1, inplace=True)


# #### 1.5 Ticket vs Survived

# In[336]:


train['Ticket'].isnull().sum()


# In[337]:


train['Ticket'].value_counts()


# <p>No null values in Ticket but ambigious values. That's why it will not be part of further analysis.</p>

# In[338]:


for dataset in full_data:
    dataset.drop('Ticket', axis=1, inplace=True)


# #### 1.6 Name vs Survived

# In[339]:


train['Name'].isnull().sum()


# In[340]:


train['Name'].value_counts()


# <p>Observed the same as in case of Ticket. Too much ambiguity. But let's try to find out title from name and analyze with survival</p>

# In[341]:


def get_title(name):
    title = re.search(" ([A-Za-z]+)\.", name)
    if title:
        return title.group(1)
    else:
        return ""


# In[342]:


for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[343]:


train[['Title', 'Name']]


# In[344]:


train['Title'].value_counts()


# In[345]:


test['Title'].value_counts()


# In[346]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Rev', 'Dr', 'Col', 'Major', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Countess', 'Capt'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms', 'Lady', 'Mme'], 'Miss')


# In[347]:


train['Title'].value_counts()


# In[348]:


test['Title'].value_counts()


# In[349]:


for dataset in full_data:
    dataset.drop('Name', axis=1, inplace=True)


# In[350]:


train.loc[:,["Title", "Survived"]].groupby("Title").mean()


# In[351]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('Title', hue='Survived', data=train)
ax.set_title('Title vs Survived')
plt.show()


# In[352]:


fig = sns.factorplot('Title', 'Survived', data=train, size=5.5, aspect=1.5)
fig.fig.suptitle('Survival Rate with Title')
plt.show()


# <p>Highest survival rate is observed with title <b>Miss</b> and <b>Mrs</b> which points towards the gender <b>Female</b>.</p>

# <p>Trends observed till now is that highest survival rate for pclass of 1 & 2, fares category with high values and gender with value female. Let's analyze them together to get better assence.</p>

# ### 2. Analysis of Continous Variables

# #### 2.1 Age vs Survived

# In[353]:


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.distplot(train['Age'])
ax.set_title("Distribution of Passenger's Age")
plt.show()


# In[354]:


# null values of age in training set
train['Age'].isnull().sum()


# In[355]:


mean_age = train['Age'].mean()
mean_age


# In[356]:


median_age = train['Age'].median()
median_age


# In[357]:


# mean is little greater than median, that's why we will be using median to fill null values
train.loc[train['Age'].isnull(),'Age'] = median_age
train['Age'].isnull().sum()


# In[358]:


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.distplot(train['Age'])
ax.set_title("Distribution of Passenger's Age")
plt.show()


# In[359]:


# Now, Age does not have null values & got more normalize than before
fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot('Survived', 'Age', data=train)
plt.show()


# In[360]:


# It seems that Age is not significant to detect survival of passenger. Let's catgorize them
for dataset in full_data:
    dataset['Age_Cat'] = pd.qcut(dataset['Age'], 5, duplicates='drop')


# In[361]:


train.loc[:,['Age_Cat', 'Survived']].groupby('Age_Cat').mean()


# In[362]:


for dataset in full_data:
    dataset.loc[dataset['Age']<=20, 'Age_Category'] = 0
    dataset.loc[(dataset['Age']>20) & (dataset['Age']<=28), 'Age_Category'] = 1
    dataset.loc[(dataset['Age']>28) & (dataset['Age']<=38), 'Age_Category'] = 2
    dataset.loc[dataset['Age']>38, 'Age_Category'] = 3


# In[363]:


train.loc[:,['Age_Category', 'Survived']].groupby('Age_Category').mean()


# In[364]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('Age_Category', hue='Survived', data=train)
ax.set_title('Age vs Survived')
plt.show()


# In[365]:


fig = sns.factorplot('Age_Category', 'Survived', data=train, size=5, aspect=1.5)
fig.fig.suptitle('Age vs Survived')


# <p>Highest survival rate for age category of 0 while least for 1.</p>

# #### 2.2 Fare vs Survived

# In[366]:


train['Fare'].isnull().sum()


# In[367]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.distplot(train['Fare'])
ax.set_title('Fare vs Survived')
plt.show()


# In[368]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot('Survived', 'Fare', data=train)
ax.set_title('Fare vs Survived')
plt.show()


# In[369]:


# As boxplot shows above, there are some outliers exist in Fare. Firstly, remove them before further processing
fare_25 = np.percentile(train['Fare'], 25)
fare_75 = np.percentile(train['Fare'], 75)
fare_iqr = fare_75 - fare_25


# In[370]:


fare_25, fare_75, fare_iqr


# In[371]:


lower_bound = fare_25 - (1.5 * fare_iqr)
upper_bound = fare_75 + (1.5 * fare_iqr)


# In[372]:


lower_bound, upper_bound


# In[373]:


full_data[0].shape


# In[378]:


train = train.loc[(train['Fare']>=lower_bound) & (train['Fare']<=upper_bound),:].reset_index(drop=True)
full_data[0] = train


# In[380]:


test = test.loc[(test['Fare']>=lower_bound) & (test['Fare']<=upper_bound),:].reset_index(drop=True)
full_data[1] = test


# In[381]:


train['Fare'].min(), train['Fare'].max()


# In[382]:


test['Fare'].min(), test['Fare'].max()


# In[383]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot('Survived', 'Fare', data=train)
ax.set_title('Fare vs Survived')
plt.show()


# In[384]:


for dataset in full_data:
    dataset['Fare_Category'] = pd.qcut(dataset['Fare'], 5)


# In[385]:


train['Fare_Category'].value_counts()


# In[386]:


train.loc[:,['Fare_Category', 'Survived']].groupby('Fare_Category').mean()


# In[387]:


for dataset in full_data:
    dataset['Fare_Category'] = 0
    dataset.loc[dataset['Fare']<=7.775, 'Fare_Category'] = 0
    dataset.loc[(dataset['Fare']>7.775) & (dataset['Fare']<=8.795), 'Fare_Category'] = 1
    dataset.loc[(dataset['Fare']>8.795) & (dataset['Fare']<=15.246), 'Fare_Category'] = 2
    dataset.loc[(dataset['Fare']>15.246) & (dataset['Fare']<=26.55), 'Fare_Category'] = 3
    dataset.loc[dataset['Fare']>26.55, 'Fare_Category'] = 4


# In[388]:


train.loc[:,['Fare_Category', 'Survived']].groupby('Fare_Category').mean()


# In[389]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('Fare_Category', hue='Survived', data=train)
ax.set_title('Fare vs Survived')
plt.show()


# In[390]:


fig = sns.factorplot('Fare_Category', 'Survived', data=train, size=5.5, aspect=1.5)
fig.fig.suptitle('Fare Category vs Survived')
plt.show()


# <p>Increase in survival rate with the increase of Fare. There is slightly drop in survival rate for fare category 4 but overall there is increase.</p>

# ### 3. Analysis of Discrete Variables

# #### 3.1 SibSp , Parch vs Survived

# In[391]:


for dataset in full_data:
    dataset['FamilyMembers'] = dataset['SibSp'].astype('int64') + dataset['Parch'].astype('int64') + 1


# In[392]:


train['FamilyMembers'].dtype, test['FamilyMembers'].dtype


# In[393]:


train.loc[:,['FamilyMembers', 'Survived']].groupby('FamilyMembers').mean()


# In[394]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot('FamilyMembers', hue='Survived', data=train)
ax.set_title('Family Members vs Survived')
plt.show()


# In[395]:


fig = sns.factorplot('FamilyMembers', 'Survived', data=train, size=5.5, aspect=1.5)
fig.fig.suptitle('Survival Rate a/c to Family Members')


# <p>Highest survival rate for passengers with 4 family members while least for 8.</p>

# In[396]:


# Since, there is an increase in survival rate with increase in fare as well as moving lower to standard in pclass.
# So, let's analyze of there is co-relation between Fare and Pclass
fig = sns.factorplot('Fare_Category', 'Survived', col='Pclass', data=train)


# <p>It shows that high class people were given priority during rescue. High survival rate for high fare and pclass of 1 & 2.</p>

# In[397]:


# Survival rate for Sex and Pclass together
sns.factorplot('Sex', 'Survived', col='Pclass', data=train)


# In[398]:


# Survival rate for Sex and Pclass together
sns.factorplot('Sex', 'Survived', col='Fare_Category', data=train)


# <p>Highest survival rate for female irrespective of pclass and fare.</p>

# ### 4. Variable Selection

# In[399]:


# remove uneccesary or redundant columns
for dataset in full_data:
    dataset.drop(['PassengerId', 'Age', 'Age_Cat', 'SibSp', 'Parch', 'Fare'], axis=1, inplace=True)


# In[400]:


train.head()


# In[401]:


train.shape


# In[402]:


test.shape


# ### 5. Encoding Categorical Variable

# In[403]:


train.head()


# In[404]:


train.shape, test.shape


# In[411]:


for index, dataset in enumerate(full_data):
    full_data[index] = pd.concat([dataset, pd.get_dummies(dataset['Sex'])], axis=1)
    full_data[index].drop('Sex', axis=1, inplace=True)


# In[412]:


full_data[0].shape, full_data[1].shape


# In[423]:


for index, dataset in enumerate(full_data):
    full_data[index] = pd.concat([dataset, pd.get_dummies(dataset['Embarked'], prefix='Embarked')], axis=1)
    full_data[index].drop('Embarked', axis=1, inplace=True)


# In[416]:


full_data[0].shape, full_data[1].shape


# In[417]:


for index, dataset in enumerate(full_data):
    full_data[index] = pd.concat([dataset, pd.get_dummies(dataset['Title'])], axis=1)
    full_data[index].drop('Title', axis=1, inplace=True)


# In[418]:


full_data[0].shape, full_data[1].shape


# In[419]:


train = full_data[0]
test = full_data[1]


# In[420]:


train.shape, test.shape


# In[421]:


train.head()


# In[424]:


test.head()


# In[426]:


train.to_csv(r'./titanic/train_set.csv', index=None)
test.to_csv(r'./titanic/test_set.csv', index=None)

