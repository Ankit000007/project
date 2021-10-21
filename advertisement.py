#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Advertisement.csv', index_col=[0])
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# # Data Cleaning

# In[6]:


df.isnull().sum()


# # checking for outlliers

# In[ ]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['radio'], ax = axs[1])
plt3 = sns.boxplot(df['newspaper'], ax = axs[2])
plt.tight_layout()


# # EDA

# In[8]:



# Let's see how Sales are related with other variables using scatter plot.

sns.scatterplot(x='TV', y='sales', data=df)


# In[9]:



sns.scatterplot(x='newspaper', y='sales', data=df)


# In[10]:


sns.scatterplot(x='radio', y='sales', data=df)


# In[11]:


# let's see the correlation between different variable

sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[12]:


#This is a linear model plot.

plt.figure(figsize=(20, 5))
sns.lmplot(data=df, x='TV', y='sales', fit_reg=True)
sns.lmplot(data=df, x='radio', y='sales', fit_reg=True)
sns.lmplot(data=df, x='newspaper', y='sales', fit_reg=True)


# # Model Building

# In[13]:



x = df.drop('sales', axis=1)
y = df['sales']


# In[14]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)


# In[15]:



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor


# In[17]:


lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.coef_)
predlr = lr.predict(x_test)
sns.scatterplot(y_test, predlr)
plt.xlabel("Y-Test")
plt.ylabel("Predicted Values")


# In[18]:


r = Ridge()
r.fit(x_train, y_train)
print(r.coef_)
predr = r.predict(x_test)
sns.scatterplot(y_test, predr)
plt.xlabel("Y-Test")
plt.ylabel("Predicted Values")


# In[20]:


l = Lasso()
l.fit(x_train,y_train)
predl = l.predict(x_test)
sns.scatterplot(y_test, predl)
plt.xlabel("Y-Test")
plt.ylabel("Predcited Values")


# In[21]:


ada = AdaBoostRegressor()
ada.fit(x_train, y_train)
predada = ada.predict(x_test)
sns.scatterplot(y_test, predada)
plt.xlabel("Y-Test")
plt.ylabel("Predcited Values")


# # Metrics Evaluation

# In[22]:


import sklearn.metrics as metrics


# In[23]:


print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predlr)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predlr)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predlr))))


# In[24]:


print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predr)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predr)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predr))))


# In[29]:


print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predl)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predl)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predl))))


# In[27]:


print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predada)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predada)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predada))))


# # Checking cross validation score

# In[30]:



from sklearn.model_selection import cross_val_score


# In[31]:


# cross validation score for linear regression

scores = cross_val_score(lr, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
    
print("Scores: ", scores)
print("Mean: ", scores.mean())


# In[32]:


# cross validation score for Ridge regression

scores = cross_val_score(r, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
    
print("Scores: ", scores)
print("Mean: ", scores.mean())


# In[33]:


# cross validation score for lasso regression

scores = cross_val_score(l, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
    
print("Scores: ", scores)
print("Mean: ", scores.mean())


# In[35]:


# cross validation score for ada boost regressor

scores = cross_val_score(ada, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
    
print("Scores: ", scores)
print("Mean: ", scores.mean())


# In[ ]:




