#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data1=pd.read_csv('datasets_180749_406301_led.csv')
data1.head()


# In[3]:


data2=pd.read_csv('datasets_446379_844964_continents2.csv')
data2.head()


# In[4]:


df = pd.merge(data1, data2[['name', 'region', 'sub-region']],
                                     left_on='Country', right_on='name')


# In[5]:


df.head()


# In[6]:


df.drop('name',axis=1,inplace=True)


# In[7]:


df["Lifeexpectancy"] = df.groupby(['region','Year'])["Lifeexpectancy"].transform(lambda x: x.fillna(x.mean()))


# In[8]:


df["AdultMortality"] = df.groupby(['region','Year'])["AdultMortality"].transform(lambda x: x.fillna(x.mean()))


# In[9]:


df["HepatitisB"] = df.groupby(['region','Year'])["HepatitisB"].transform(lambda x: x.fillna(x.mean()))


# In[10]:


df["BMI"] = df.groupby(['region','Year'])["BMI"].transform(lambda x: x.fillna(x.mean()))


# In[11]:


df["Polio"] = df.groupby(['region','Year'])["Polio"].transform(lambda x: x.fillna(x.mean()))


# In[12]:


df["Diphtheria"] = df.groupby(['region','Year'])["Diphtheria"].transform(lambda x: x.fillna(x.mean()))


# In[13]:


df["GDP"] = df.groupby(['region','Year'])["GDP"].transform(lambda x: x.fillna(x.mean()))


# In[14]:


df["Population"] = df.groupby(['region','Year'])["Population"].transform(lambda x: x.fillna(x.mean()))


# In[15]:


df["thinness10-19years"] = df.groupby(['region','Year'])["thinness10-19years"].transform(lambda x: x.fillna(x.mean()))


# In[16]:


df["thinness5-9years"] = df.groupby(['region','Year'])["thinness5-9years"].transform(lambda x: x.fillna(x.mean()))


# In[17]:


df["Incomecompositionofresources"] = df.groupby(['region','Year'])["Incomecompositionofresources"].transform(lambda x: x.fillna(x.mean()))


# In[18]:


df["Schooling"] = df.groupby(['region','Year'])["Schooling"].transform(lambda x: x.fillna(x.mean()))


# In[19]:


df["Alcohol"] = df.groupby(['region'])["Alcohol"].transform(lambda x: x.fillna(x.mean()))


# In[20]:


df["Totalexpenditure"] = df.groupby(['region'])["Totalexpenditure"].transform(lambda x: x.fillna(x.mean()))


# In[21]:


df['Year'] = df["Year"].replace({2000:1,2001:2,2002:3,2003:4,2004:5,2005:6,2006:7,2007:8,2008:9,2009:10,
                                2010:11,2011:12,2012:13,2013:14,2014:15,2015:16})


# In[22]:


df_new=pd.get_dummies(df,columns=['sub-region','region','Status'],drop_first=True)


# In[23]:


a=df['Country']


# In[24]:


df_new.drop('Country',axis=1,inplace=True)


# In[25]:


X=df_new.drop('Lifeexpectancy',axis=1)
y=df_new['Lifeexpectancy']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


from sklearn.metrics import mean_squared_error, r2_score


# In[28]:


X=df_new.drop(['Lifeexpectancy','infantdeaths','thinness5-9years','percentageexpenditure','Year','thinness10-19years','Status_Developing',
              'sub-region_Central Asia', 'sub-region_Eastern Asia',
       'sub-region_Eastern Europe',
       'sub-region_Latin America and the Caribbean', 'sub-region_Melanesia',
       'sub-region_Micronesia', 'sub-region_Northern Africa',
       'sub-region_Northern America', 'sub-region_Northern Europe',
       'sub-region_Polynesia', 'sub-region_South-eastern Asia',
       'sub-region_Southern Asia', 'sub-region_Southern Europe',
       'sub-region_Sub-Saharan Africa', 'sub-region_Western Asia',
       'sub-region_Western Europe', 'region_Americas', 'region_Asia',
       'region_Europe', 'region_Oceania', ],axis=1)
y=df_new['Lifeexpectancy']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[30]:


import xgboost
from xgboost import plot_importance
xgb_model = xgboost.XGBRegressor()
xgb_model.fit(X_train,y_train)
yhat1 = xgb_model.predict(X_test)
print(mean_squared_error(y_test,yhat1))

print('test_rmse: ',np.sqrt(mean_squared_error(y_test,yhat1)))
print('test_r2 score: ',r2_score(y_test, yhat1))


# In[31]:


import pickle
pickle.dump(xgb_model, open('Life_expectancy.pkl','wb'))

model = pickle.load(open('Life_expectancy.pkl','rb'))


# In[32]:


pip install gunicorn


# In[ ]:




