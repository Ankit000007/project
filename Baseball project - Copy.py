#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[5]:


baseball=pd.read_csv('baseball.csv')


# In[6]:


baseball


# In[7]:


len(baseball)


# In[8]:


print(baseball.shape)


# In[9]:


baseball = baseball.drop(['BB' , 'CG'], axis =1)
baseball


# In[10]:


baseball.head


# In[11]:


baseball.info()


# In[12]:



baseball['Wins'].value_counts()


# In[ ]:


print(baseball.isnull().sum(axis = 0))


# In[ ]:


plt.hist(baseball['Wins'])
plt.xlabel('Team Wins')
plt.title('Histogram Plot of Total Wins')


# In[ ]:


plt.scatter(baseball['Runs'],baseball['At Bats'])
plt.title('Runs per At Bat')
plt.xlabel('Runs')
plt.ylabel('At Bat')


# In[ ]:


sns.countplot(baseball.Runs)


# In[ ]:



#checking for the presence of values using heatmap
sns.heatmap(baseball.isnull(), annot= True)
plt.show()


# In[ ]:


baseball.skew()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
baseball['Runs'] = le.fit_transform(baseball['Runs'])
baseball['Wins'] = le.fit_transform(baseball['Wins'])
baseball.head(5)


# In[ ]:


baseball = pd.get_dummies(baseball)
baseball.head(5)


# In[ ]:


baseball.info()


# In[ ]:



#spliting the data

from sklearn.model_selection import train_test_split

y = baseball['Wins']
X = baseball

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


# In[ ]:


#Using Decision Tree Model

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=1)
dtree.fit(X_train, y_train)
y_pred1 = dtree.predict(X_test)
print(classification_report(y_test, y_pred1))


# In[ ]:



#Applying Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)
y_pred2 = rf.predict(X_test)
print(classification_report(y_test, y_pred2))


# In[ ]:


#Applying Lasso Regression

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))


# In[ ]:





# In[ ]:




