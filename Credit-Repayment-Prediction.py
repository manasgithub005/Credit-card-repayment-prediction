#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


get_ipython().system('pip install lightgbm')


# In[3]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[4]:


credit_card_data=pd.read_csv('credit_card_clean.csv')
credit_card_data


# ### 3. Data Understanding
# Exploring the variable
# ID : ID of each client
# 
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
# 
# SEX: Gender (1=male, 2=female)
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# AGE: Age in years
# 
# PAY_1: Repayment status in September, 2005 (-2 = No consumption, -1 = paid in full, 0 = use of revolving credit (paid minimum only), 1 = payment delay for one month, 2 = payment delay for two months, ... 8 = payment delay for eight months, 9 = payment delay for nine months and above)
# 
# PAY_2: Repayment status in August, 2005 (scale same as above)
# 
# PAY_3: Repayment status in July, 2005 (scale same as above)
# 
# PAY_4: Repayment status in June, 2005 (scale same as above)
# 
# PAY_5: Repayment status in May, 2005 (scale same as above)
# 
# PAY_6: Repayment status in April, 2005 (scale same as above)
# 
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# 
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# 
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# 
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# 
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# 
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# 
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# 
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# 
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# 
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# 
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# 
# Target variable
# default.payment.next.month: Default payment (1=yes, 0=no)
# 
# Note: NT is Taiwain Dollars

# ### 3.Data Understanding

# In[5]:


credit_card_data.shape


# In[6]:


credit_card_data.isna().sum()


# ### 4.Data Preprocessing

# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
credit_card_data['SEX']       = le.fit_transform(credit_card_data['SEX'])
credit_card_data['EDUCATION'] = le.fit_transform(credit_card_data['EDUCATION'])
credit_card_data['MARRIAGE']  = le.fit_transform(credit_card_data['MARRIAGE'])
credit_card_data


# In[8]:


credit_card_data.dtypes


# In[9]:


credit_card_data


# In[10]:


pd.set_option('max_columns',None)  #no restrictions should be given for displaying columns


# In[11]:


credit_card_data.head(10)


# ### 5.Model Building

# In[12]:


X=credit_card_data.drop('DEFAULT',axis=1)
y=credit_card_data[['DEFAULT']]


# In[13]:


y.value_counts()   #highly imbalanced we have to twig the class


# In[14]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=123,stratify=y)


# In[15]:


X_train.shape,y_train.shape


# In[16]:


X_test.shape,y_test.shape     


# ###  6. Model Training

# In[17]:


rfclassifier    = RandomForestClassifier()
adb_classifier  = AdaBoostClassifier()
gb_classifier   = GradientBoostingClassifier()
xgb_classifer   = XGBClassifier()
lgbm_classifier = LGBMClassifier()


# In[18]:


get_ipython().run_cell_magic('time', '', 'rfclassifier.fit(X_train,y_train)')


# In[19]:


get_ipython().run_cell_magic('time', '', 'adb_classifier.fit(X_train,y_train)')


# In[20]:


get_ipython().run_cell_magic('time', '', 'gb_classifier.fit (X_train,y_train)')


# In[21]:


get_ipython().run_cell_magic('time', '', 'xgb_classifier.fit(X_train,y_train).fit')


# In[22]:


get_ipython().run_cell_magic('time', '', 'lgbm_classifier.fit(X_train,y_train)')


# ###  7.Model Testing

# In[23]:


rf_pred  = rfclassifier.predict(X_test)
adb_pred = adb_classifier.predict(X_test)
gb_pred  = gb_classifier.predict(X_test)
xgb_pred = xgb_classifer.predict(X_test)
lgb_pred = lgbm_classifier.predict(X_test)


# ### 8.Model Evaluation

# #### 1.Random Forest Evaluation

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print('Accuracy Score:',round(accuracy_score(y_test,rf_pred),4))
print('Precision Score:',round(precision_score(y_test,rf_pred),4))
print('Recall Score: ',round(recall_score(y_test,rf_pred),4))
print('Confusion Mtrix:\n ',confusion_matrix(y_test,rf_pred))


# In[ ]:





# ### 2.Adaboost Classifier

# In[ ]:


print('Accuracy Score:',round(accuracy_score(y_test,adb_pred),4))
print('Precision Score:',round(precision_score(y_test,adb_pred),4))
print('Recall Score: ',round(recall_score(y_test,adb_pred),4))
print('Confusion Mtrix:\n ',confusion_matrix(y_test,adb_pred))


# ### 3. Gradient Boosting Classifier

# In[ ]:


print('Accuracy Score:',round(accuracy_score(y_test,gb_pred),4))
print('Precision Score:',round(precision_score(y_test,gb_pred),4))
print('Recall Score: ',round(recall_score(y_test,gb_pred),4))
print('Confusion Mtrix:\n ',confusion_matrix(y_test,gb_pred))


# ### 4.Extreme Gradient Boosting Classifier

# In[ ]:


print('Accuracy Score:',round(accuracy_score(y_test,xgb_pred),4))
print('Precision Score:',round(precision_score(y_test,xgb_pred),4))
print('Recall Score: ',round(recall_score(y_test,xgb_pred),4))
print('Confusion Mtrix:\n ',confusion_matrix(y_test,xgb_pred))


# ### 5.LightGBM Classifier

# In[ ]:


print('Accuracy Score:',round(accuracy_score(y_test,lgb_pred),4))
print('Precision Score:',round(precision_score(y_test,lgbm_pred),4))
print('Recall Score: ',round(recall_score(y_test,lgb_pred),4))
print('Confusion Mtrix:\n ',confusion_matrix(y_test,lgb_pred))


# ### ==============================================================================================

# In[ ]:





# In[ ]:




