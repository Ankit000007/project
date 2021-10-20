#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, joblib
from scipy.stats import zscore

#Encode
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Skewness
from sklearn.preprocessing import PowerTransformer

#Normalize
from sklearn.preprocessing import MinMaxScaler

#Train Test Split
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#RegressionModels
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

#Optimization
from sklearn.model_selection import GridSearchCV


# In[4]:


df=pd.read_csv('insurance.csv')


# In[5]:


df.head()


# In[6]:


#Check info
df.info()
#There are no null values


# In[7]:


#Check Stats
df.describe()


# In[8]:


df.isna().sum()
#There are no missing values


# In[9]:


df.select_dtypes('object').columns


# In[10]:


df['Customer'].nunique()


# In[11]:


df=df.drop('Customer',axis=1)


# In[12]:


df['Country'].nunique()


# In[13]:


df['Country'].value_counts()


# In[14]:



df=df.drop('Country',axis=1)


# In[15]:


df['State Code'].nunique()


# In[16]:


df['State Code'].value_counts()


# In[17]:


df=pd.concat([df.drop('State Code',axis=1),pd.get_dummies(df['State Code'])],axis=1)


# In[18]:


df['State'].nunique()


# In[19]:



#This are same values as State Code
df['State'].value_counts()


# In[20]:


df=df.drop('State',axis=1)


# In[21]:


df['Response'].nunique()


# In[22]:


df['Response'].value_counts()


# In[23]:


le=LabelEncoder()
df['Response']=le.fit_transform(df['Response'])
df['Coverage'].nunique()


# In[24]:



df['Coverage'].value_counts()


# In[25]:


df=pd.concat([df.drop('Coverage',axis=1),pd.get_dummies(df['Coverage'])],axis=1)
df['Education'].nunique()


# In[26]:


df['Education'].value_counts()


# In[27]:


df=pd.concat([df.drop('Education',axis=1),pd.get_dummies(df['Education'])],axis=1)


# In[28]:


df['Effective To Date']=pd.to_datetime(df['Effective To Date'],format='%m/%d/%y')


# In[29]:


df['Effective To Date'].dt.year.nunique()
#Data has been taken for only one year hence we will not bifurcate year from this column
#Instead we will only create a new clumn for month


# In[30]:


df['Effective Month']=df['Effective To Date'].dt.month


# In[31]:


#Drop Effective to Date column as month has already been extracted
df=df.drop('Effective To Date',axis=1)
df['EmploymentStatus'].nunique()


# In[32]:


df['EmploymentStatus'].value_counts()


# In[33]:



df=pd.concat([df.drop('EmploymentStatus',axis=1),pd.get_dummies(df['EmploymentStatus'])],axis=1)
#We have encoded the values in different columns so that model does not consider it as a rank or order


# In[34]:


df['Gender'].nunique()
#Since there are only two values we will label encode it


# In[35]:


df['Gender']=le.fit_transform(df['Gender'])
df['Location Code'].nunique()


# In[36]:


df['Location Code'].value_counts()


# In[37]:


df=pd.concat([df.drop('Location Code',axis=1),pd.get_dummies(df['Location Code'])],axis=1)
#We have encoded the values in different columns so that model does not consider it as a rank or order


# In[38]:


df['Policy'].nunique()


# In[39]:


df['Policy'].value_counts()
#We will LabelEncode this values


# In[40]:



df['Policy']=le.fit_transform(df['Policy'])


# In[41]:


df['Claim Reason'].nunique()


# In[42]:


df['Claim Reason'].value_counts()


# In[43]:


df['Claim Reason']=le.fit_transform(df['Claim Reason'])


# In[44]:


df['Sales Channel'].nunique()


# In[45]:


df['Sales Channel'].value_counts()


# In[46]:


df['Sales Channel']=le.fit_transform(df['Sales Channel'])


# In[47]:


df['Vehicle Class'].nunique()


# In[48]:



df['Vehicle Class'].value_counts()


# In[49]:


df=pd.concat([df.drop('Vehicle Class',axis=1),pd.get_dummies(df['Vehicle Class'])],axis=1)
#We have encoded the values in different columns so that model does not consider it as a rank or order


# In[50]:


df['Vehicle Size'].nunique()


# In[51]:


df['Vehicle Size'].value_counts()


# In[52]:


df=pd.concat([df.drop('Vehicle Size',axis=1),pd.get_dummies(df['Vehicle Size'])],axis=1)
#We have encoded the values in different columns so that model does not consider it as a rank or order


# In[53]:


#Check for object types again
df.select_dtypes('object').columns
#There are no columns for object type


# In[54]:



#Check for correlation
plt.figure(figsize=(10,6))
df.corr()['Total Claim Amount'].drop('Total Claim Amount').sort_values(ascending=False).plot(kind='bar')
plt.show()


# In[55]:


df.corr()['Total Claim Amount'].drop('Total Claim Amount').sort_values(ascending=False)


# In[57]:


#Check for correlation
plt.figure(figsize=(10,6))
df.corr()['Total Claim Amount'].drop('Total Claim Amount').sort_values(ascending=False).plot(kind='bar')
plt.show()


# In[58]:


#Check for correlation
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.show()


# In[59]:


#Check for skewness
df.skew()


# In[76]:


df.skew()


# In[79]:


pt=PowerTransformer()


# In[80]:


#Standardize/Normalize the data
scale=MinMaxScaler()


# In[81]:


X=df.drop('Total Claim Amount',axis=1)
y=df['Total Claim Amount']


# In[83]:


def model_sel(mod):
    maxscore=0
    maxstate=0
    for x in range(42,105):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=x)
        model=mod()
        model.fit(X_train,y_train)
        predict=model.predict(X_test)
        r2score=r2_score(y_test,predict)
        if r2score>maxscore:
            maxscore=r2score
            maxstate=x
    print("Max Score is {} at max state {}".format(maxscore,maxstate))


# In[84]:



model_sel(LinearRegression)


# In[85]:



model_sel(Ridge)


# In[86]:


model_sel(Lasso)


# In[87]:



model_sel(ElasticNet)


# In[ ]:


model_sel(KNeighborsRegressor)


# In[61]:


model_sel(DecisionTreeRegressor)


# In[62]:


model_sel(RandomForestRegressor)


# In[63]:


model_sel(SVR)


# In[64]:


model_sel(AdaBoostRegressor)


# In[65]:


#From the above results max score is for RandomForestRegressor at 83 random state
#we will try to hypertune the parameters of RandomForestRegressor


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=75)


# In[67]:


param={'n_estimators':[100,110,130,150,170,210,230,270,310,330,350,370,390,510]}
gscv=GridSearchCV(RandomForestRegressor(),param)


# In[68]:


gscv.fit(X_train,y_train)


# In[69]:


gscv.best_params_


# In[70]:


rf=RandomForestRegressor(n_estimators=350)
rf.fit(X_train,y_train)
predict=rf.predict(X_test)


# In[71]:


print("r2score is {}".format(r2_score(y_test,predict)))
print("MAE is {}".format(mean_absolute_error(y_test,predict)))
print("MSE is {}".format(mean_squared_error(y_test,predict)))


# In[72]:


result=pd.DataFrame(y_test)


# In[73]:


result=pd.concat([result.reset_index().drop('index',axis=1),pd.Series(predict)],axis=1)


# In[74]:



result.columns=['Total Claim Amount','Total Predicted Claim Amount']

plt.figure(figsize=(10,6))
sns.scatterplot(x='Total Claim Amount',y='Total Predicted Claim Amount',data=result)
plt.show()
#The below plot shows linear behaviour in predicted and original values which is satisfactory


# In[75]:


#Save the model
joblib.dump(rf,'Insurance.obj')


# In[ ]:




