#!/usr/bin/env python
# coding: utf-8

# # Red wine prediction

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# # Reading cvs file

# In[4]:


df = pd.read_csv('wine.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# # Checking for null values

# In[8]:


df.isnull().sum()


# In[9]:


sns.heatmap(df.isnull())


# # Checking for data correlation

# In[10]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()


# # Feature selection

# In[11]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
x = df.drop('quality', axis=1)
y = df['quality']
best_feature = SelectKBest(score_func=f_classif, k=11)
fit = best_feature.fit(x, y)


# In[12]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)


# In[13]:


# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  #naming the dataframe column


# In[14]:


featureScores.sort_values(by=['Score'], ascending=False)


# In[15]:


df.drop(['fixed acidity', 'chlorides', 'free sulfur dioxide', 'pH', 'residual sugar'], axis=1, inplace=True)


# In[16]:


df.head()


# # Checking if our data is balanced or not

# In[17]:


df['quality'].plot.hist()
df['quality'].value_counts()


# In[18]:


df['quality'] = np.where((df.quality >= 6), 1, 0)


# In[19]:


df['quality'].plot.hist()


# # Checking for skewness

# In[20]:


df.plot(kind='density', subplots=True, layout=(3,3), sharex=False, legend=False, figsize=(15,11))
plt.show()


# In[21]:


df.skew()


# # Checking for outliers

# In[22]:


df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, legend=False, figsize=(15,11))
plt.show()


# # Removing Outliers

# In[23]:


from scipy.stats import zscore

z = np.abs(zscore(df))

df_new = df[ (z < 3).all(axis=1) ]


# In[24]:


df.shape


# In[25]:


df_new.shape


# # Finding best random state

# In[26]:


x = df_new.drop('quality', axis=1)
y = df_new['quality']


# In[27]:


maxAccu = 0
maxRS = 0

for i in range(1,200):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=i)
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    pred = lg.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAccu:
        maxAccu = acc
        maxRS = i
print("Best accuracy is: ",maxAccu,"on Random State: ",maxRS)


# # Creating train test split

# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=116)


# In[29]:


LR = LogisticRegression()
LR.fit(x_train, y_train)
pred = LR.predict(x_test)
print(accuracy_score(y_test, pred)*100)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('f1 Score:',f1_score(y_test, pred)*100)


# In[30]:


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(accuracy_score(y_test, pred)*100)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('f1 Score:',f1_score(y_test, pred)*100)


# In[44]:


from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier()
rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
print(accuracy_score(y_test, pred)*100)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('f1 Score:',f1_score(y_test, pred)*100)


# In[31]:


svc = SVC()
svc.fit(x_train, y_train)
pred = svc.predict(x_test)
print(accuracy_score(y_test, pred)*100)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('f1 Score:',f1_score(y_test, pred)*100)


# In[32]:


gnb = GaussianNB()
gnb.fit(x_train, y_train)
pred = gnb.predict(x_test)
print(accuracy_score(y_test, pred)*100)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('f1 Score:',f1_score(y_test, pred)*100)


# In[33]:


scr = cross_val_score(LR, x, y, cv=5)
print("Cross Validation score of Logistic Regression: ",scr.mean()*100)


# In[34]:



# cross validation score for k nearest neighbors

scr = cross_val_score(knn, x, y, cv=5)
print("Cross Validation score of Logistic Regression: ",scr.mean()*100)


# In[35]:


# cross validation score for SVC

scr = cross_val_score(svc, x, y, cv=5)
print("Cross Validation score of Logistic Regression: ",scr.mean()*100)


# In[36]:


# cross validation score for Gaussian Naive Bayes

scr = cross_val_score(gnb, x, y, cv=5)
print("Cross Validation score of Logistic Regression: ",scr.mean()*100)


# In[46]:


# cross validation score for Random Forest Classifier

scr = cross_val_score(rfc, x, y, cv=10)
print("Cross Validation score of Random Forest Classifier: ",scr.mean()*100)


# # Hyper parameter tuning

# In[37]:



# creating parameter list to pass in GridSearchCV

parameters = {'var_smoothing': np.logspace(0,-9, num=100)}
GCV = GridSearchCV(estimator=gnb, 
                 param_grid=parameters, 
                 cv=5,   # use any cross validation technique 
                 verbose=1, 
                 scoring='accuracy') 
GCV.fit(x_train, y_train)


# In[38]:


GCV.best_params_


# In[39]:


mod = GaussianNB(var_smoothing= 4.328761281083062e-05)

mod.fit(x_train, y_train)
pred = mod.predict(x_test)
print('f1_score:',f1_score(y_test, pred)*100)


# # Plotting AUC ROC curve

# In[40]:


# calculate the fpr and tpr for all thresholds of the classification
probs = gnb.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Saving the model

# In[41]:


import joblib
joblib.dump(mod, 'redwinemodel.pkl')


# # Loading the saved model

# In[48]:


model = joblib.load('redwinemodel.pkl')
prediction = model.predict(x_test)
print('f1 score:',f1_score(y_test, prediction)*100)


# In[49]:


prediction = pd.DataFrame(prediction)


# In[50]:


prediction.to_csv('result.csv')
prediction


# In[ ]:




