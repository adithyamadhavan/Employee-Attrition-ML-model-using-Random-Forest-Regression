#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[2]:


empleftdata = pd.read_excel('D:\datasets\Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', sheet_name = 2)


# In[3]:


empleftdata


# In[4]:


empleftdata.describe()


# In[5]:


empleftdata.isnull().sum()


# In[6]:


dat =pd.pivot_table(empleftdata,index = 'salary',columns =['dept','number_project','promotion_last_5years'],values = ['satisfaction_level'],aggfunc = 'sum')
dat


# In[7]:


df1 =pd.pivot_table(empleftdata,index = 'salary',columns ='dept',values = 'satisfaction_level',aggfunc = 'sum')
df1


# In[8]:


df1 =pd.pivot_table(empleftdata,index = 'salary',columns ='dept',values = 'satisfaction_level',aggfunc = 'mean')
df1


# In[9]:


sns.heatmap(df1,annot = True,fmt ='.0%').get_figure()


# In[10]:


df1 =pd.pivot_table(empleftdata,index = 'salary',columns ='dept',values = 'satisfaction_level',aggfunc = 'mean')
df1


# In[11]:


# histogram plot of employee satisfaction across different salary
plt.hist(df1,bins = 15)
plt.xlabel('salary')
plt.title('Employee Left')
plt.ylabel('satisfication level')
red_patch = mpatches.Patch(color='orange', label='high')
blue_patch = mpatches.Patch(color = 'blue',label = 'low')
green_patch = mpatches.Patch(color = 'green',label = 'medium')
plt.legend(handles=[red_patch,blue_patch,green_patch],loc = 'upper left')
plt.show()


# In[12]:


#Print all of the object data types and their unique values
for column in empleftdata.columns:
    if empleftdata[column].dtype == object:
        print(str(column) + ' : ' + str(empleftdata[column].unique()))
        print(empleftdata[column].value_counts())
     


# In[13]:


plt.subplots(figsize = (45,6))
sns.countplot(x ='satisfaction_level',hue ='dept',data =empleftdata,palette ='colorblind')


# In[14]:


empleftdata1 =empleftdata.drop('Emp ID',axis =1)
empleftdata1


# In[15]:


empleftdata1.corr()


# In[16]:


plt.figure(figsize =(10,10))
sns.heatmap(empleftdata1.corr(),annot = True,fmt ='.0%')


# In[17]:


newdata =  empleftdata1


# In[18]:


data = pd.get_dummies(empleftdata1['dept'])


# In[19]:


data


# In[20]:


newdata['IT dept'] = data['IT'].to_list()
newdata['RandD dept'] = data['RandD'].to_list()
newdata['accounting dept'] = data['accounting'].to_list()
newdata['hr dept'] = data['hr'].to_list()
newdata['management dept'] = data['management'].to_list()
newdata['marketing dept'] = data['marketing'].to_list()
newdata['product_mng dept'] = data['product_mng'].to_list()
newdata['sales dept'] = data['sales'].to_list()
newdata['support dept'] = data['support'].to_list()
newdata['technical dept'] = data['technical'].to_list()


# In[21]:


newdata


# In[22]:


newdata = newdata.drop('dept',axis =1)


# In[23]:


data1 = pd.get_dummies(empleftdata1['salary'])


# In[24]:


data1


# In[25]:


newdata['high salary'] = data1['high'].to_list()
newdata['low salary'] = data1['low'].to_list()
newdata['medium salary'] = data1['medium'].to_list()


# In[26]:


newdata


# In[27]:


newdata = newdata.drop('salary',axis =1)


# In[28]:


d = newdata[['satisfaction_level','average_montly_hours']]


# In[29]:


d


# In[30]:


d.describe()


# In[31]:


plt.hist(d,bins = 20)
plt.title('Satisfaction level vs monthly hours')
plt.xlabel('average monthly hours')
plt.ylabel('satisfaction level')
plt.show()


# In[32]:


dt =pd.pivot_table(empleftdata,index = ['salary'],columns =['dept'],values = ['satisfaction_level','average_montly_hours'],aggfunc = 'sum')
dt


# In[33]:


dt =pd.pivot_table(empleftdata,index = ['salary'],columns =['dept'],values = ['satisfaction_level','average_montly_hours'],aggfunc = 'count')
dt


# In[34]:


newdata


# In[35]:


#training model based on satisfaction level


# In[36]:


X = newdata.iloc[:,1:,].values
Y = newdata.iloc[:,0].values


# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.25,random_state =0)


# In[38]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =10,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[39]:


#accuracy using forest
forest.score(X_train,Y_train)


# In[40]:


#100 trees in random forest


# In[41]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =100,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[42]:


forest.score(X_train,Y_train)


# In[43]:


#Training model based on last evaluation


# In[44]:


X = newdata.iloc[:,0:,].values
Y = newdata.iloc[:,1].values


# In[45]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.25,random_state =0)


# In[46]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =10,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[47]:


forest.score(X_train,Y_train)


# In[48]:


#100 trees forest


# In[49]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =100,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[50]:


forest.score(X_train,Y_train)


# In[51]:


#Training model based on average working hours vs satisfication level


# In[52]:


X = dt.iloc[:,0:10].values
Y = dt.iloc[:,10:,].values


# In[53]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.25,random_state =0)


# In[54]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =10,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[55]:


forest.score(X_train,Y_train)


# In[56]:


X = dt.iloc[:,0:10].values
Y = dt.iloc[:,10:,].values

#Decison Tree of Satisfaction level 
# In[57]:


from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)


# In[58]:


plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color = 'red')
plt.title('Satisfacation Level and Last Evaluation')
plt.xlabel('Time Spend')
plt.ylabel("Satisfaction level")
plt.show()


# In[ ]:




