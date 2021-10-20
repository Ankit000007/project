#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import skew, zscore
import sklearn


# In[3]:


from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,mean_absolute_error
from sklearn.preprocessing import LabelEncoder,StandardScaler,power_transform
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,RandomForestClassifier,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")


# In[4]:


dh=pd.read_csv("abal.csv")
dh.head()


# In[5]:


pd.set_option("display.max_rows",None)
dh.info()


# In[6]:


dh.dtypes


# In[7]:


dh.columns


# In[8]:


dh.describe()


# # Exploratory Data Analysis

# In[9]:


dh.isnull().sum()


# In[10]:


plt.figure(figsize=(12,6))
sns.heatmap(dh.isnull())


# In[11]:


dh["Rings"].value_counts()


# In[12]:


sns.countplot(dh.Rings)


# In[13]:


sns.pairplot(dh)


# In[14]:


dh.corr()


# In[15]:


corr_mat=dh.corr()
corr_mat["Rings"].sort_values(ascending=False)


# In[16]:


plt.figure(figsize=(15,10))
plt.title("correlation matrix")
sns.heatmap(dh.corr(),annot=True)


# In[17]:


plt.figure(figsize=(12,8))
plt.subplot(3,3,1)
plt.title("shell weight vs rings")
plt.scatter(dh['Shell weight'],dh['Rings'])
plt.tight_layout()


# In[18]:


plt.figure(figsize=(12,8))
plt.subplot(3,3,2)
plt.title("diameter vs rings")
plt.scatter(dh['Diameter'],dh['Rings'])
plt.tight_layout()


# In[19]:


plt.figure(figsize=(12,8))
plt.subplot(3,3,3)
plt.title("length vs rings")
plt.scatter(dh['Length'],dh['Rings'])
plt.tight_layout()


# In[20]:


plt.figure(figsize=(12,8))
plt.subplot(3,3,4)
plt.title(" whole weight vs rings")
plt.scatter(dh['Whole weight'],dh['Rings'])
plt.tight_layout()


# In[21]:


plt.figure(figsize=(12,8))
plt.subplot(3,3,5)
plt.title("viscera weight vs rings")
plt.scatter(dh['Viscera weight'],dh['Rings'])
plt.tight_layout()


# In[22]:


plt.figure(figsize=(12,8))
plt.subplot(3,3,6)
plt.title("Height vs rings")
plt.scatter(dh['Height'],dh['Rings'])
plt.tight_layout()


# In[23]:


dh['Sex'].value_counts()


# In[24]:


Label=LabelEncoder()
dh['Sex']=Label.fit_transform(dh['Sex'])
dh.head()


# # count plot

# In[25]:


sns.countplot(x='Sex',data=dh)
plt.title('SEX')


# # Distribution plot

# In[26]:


for i in dh.columns:
    plt.figure()
    sns.distplot(dh[i],bins=20)


# In[27]:


dh.skew()


# # Removing skewness

# In[28]:


for i in dh.columns:
    plt.figure()
    sns.boxplot(dh[i])


# In[29]:


Y=dh.pop('Rings')


# In[30]:


x_train,y_train,x_test,y_test=train_test_split(dh,Y,test_size=.30,random_state=45)


# In[31]:


x_train.skew()


# In[32]:


dh_new=power_transform(x_train,method='yeo-johnson')
dh_new=pd.DataFrame(dh_new,columns=x_train.columns)
dh_new.skew()


# # Removing outliers:

# In[33]:


z=np.abs(zscore(dh))


# In[34]:


threshold=3
print(np.where(z>3))


# In[35]:


dh_new=dh[(z<3).all(axis=1)]
print(dh.shape)
print(dh_new.shape)


# # percentage of Dataloss

# In[36]:


loss=(4177-4084)/4177*100
print(loss)


# In[37]:


x=dh_new.iloc[:,:-1]
y=dh_new.iloc[:,-1]


# In[38]:


x_test.ndim


# In[39]:


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)


# In[ ]:




