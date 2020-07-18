#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[2]:


existingempdata = pd.read_excel('D:\datasets\Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', sheet_name = 1)


# In[3]:


existingempdata


# In[4]:


existingempdata.describe()


# In[5]:


existingempdata.isnull().sum()


# In[6]:


df =pd.pivot_table(existingempdata,index = 'salary',columns ='dept',values = ['satisfaction_level','last_evaluation'],aggfunc = 'sum')
df


# In[7]:


df_1 =pd.pivot_table(existingempdata,index = 'salary',columns ='dept',values = 'satisfaction_level',aggfunc = 'sum')
df_1


# In[8]:


df_1 =pd.pivot_table(existingempdata,index = 'salary',columns ='dept',values = 'satisfaction_level',aggfunc = 'mean')
df_1


# In[9]:


sns.heatmap(df_1,annot = True,fmt ='.0%').get_figure()


# In[10]:


plt.hist(df_1,bins = 15)
plt.xlabel('salary')
plt.ylabel('satisfication level')
plt.title('Existing Employee ')
red_patch = mpatches.Patch(color='orange', label='high')
blue_patch = mpatches.Patch(color = 'blue',label = 'low')
green_patch = mpatches.Patch(color = 'green',label = 'medium')
plt.legend(handles=[red_patch,blue_patch,green_patch],loc = 'upper left')
plt.show()


# In[11]:


for column in existingempdata.columns:
    if existingempdata[column].dtype == object:
        print(str(column) + ' : ' + str(existingempdata[column].unique()))
        print(existingempdata[column].value_counts())
     


# In[12]:


plt.subplots(figsize = (45,6))
sns.countplot(x ='satisfaction_level',hue ='dept',data =existingempdata,palette ='colorblind')


# In[13]:


existingempdata1 =existingempdata.drop('Emp ID',axis =1)
existingempdata1


# In[14]:


existingempdata1.corr()


# In[15]:


plt.figure(figsize =(10,10))
sns.heatmap(existingempdata1.corr(),annot = True,fmt ='.0%')


# In[16]:


new_data =  existingempdata1


# In[17]:


data = pd.get_dummies( existingempdata1['dept'])


# In[18]:


data


# In[19]:


new_data['IT dept'] = data['IT'].to_list()
new_data['RandD dept'] = data['RandD'].to_list()
new_data['accounting dept'] = data['accounting'].to_list()
new_data['hr dept'] = data['hr'].to_list()
new_data['management dept'] = data['management'].to_list()
new_data['marketing dept'] = data['marketing'].to_list()
new_data['product_mng dept'] = data['product_mng'].to_list()
new_data['sales dept'] = data['sales'].to_list()
new_data['support dept'] = data['support'].to_list()
new_data['technical dept'] = data['technical'].to_list()


# In[20]:


new_data


# In[21]:


new_data = new_data.drop('dept',axis =1)


# In[22]:


data1 = pd.get_dummies(existingempdata1['salary'])


# In[23]:


data1 


# In[24]:


new_data['high salary'] = data1['high'].to_list()
new_data['low salary'] = data1['low'].to_list()
new_data['medium salary'] = data1['medium'].to_list()


# In[25]:


new_data


# In[26]:


new_data = new_data.drop('salary',axis =1)


# In[27]:


new_data


# In[28]:


es = new_data[['satisfaction_level','average_montly_hours']]


# In[29]:


es


# In[30]:


es.describe()


# In[38]:


plt.figure(figsize =(30,30))
plt.hist(es,bins = 20)
plt.title('Satisfaction level vs monthly hours')
plt.xlabel('average monthly hours')
plt.ylabel('satisfaction level')
plt.show()


# In[31]:


d_t =pd.pivot_table(existingempdata,index = ['salary'],columns =['dept'],values = ['satisfaction_level','average_montly_hours'],aggfunc = 'sum')
d_t


# In[32]:


drt =pd.pivot_table(existingempdata,index = ['salary'],columns =['dept'],values = ['satisfaction_level','average_montly_hours'],aggfunc = 'count')
drt


# In[33]:


new_data


# In[34]:


X = new_data.iloc[:,1:,].values
Y = new_data.iloc[:,0].values


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.25,random_state =0)


# In[36]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =10,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[37]:


forest.score(X_train,Y_train)


# In[39]:


X = drt.iloc[:,0:10].values
Y = drt.iloc[:,10:,].values


# In[40]:


from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)


# In[41]:


plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color = 'red')
plt.title('Satisfacation Level and Last Evaluation')
plt.xlabel('Time Spend')
plt.ylabel("Satisfaction level")
plt.show()


# In[42]:


#100 trees random forest based on satisfaction


# In[43]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =100,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[44]:


forest.score(X_train,Y_train)


# In[45]:


#300 trees random forest based on satisfaction


# In[46]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =300,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[47]:


forest.score(X_train,Y_train)


# In[48]:


#Training model based on last evaluation


# In[49]:


#10 trees


# In[50]:


X = new_data.iloc[:,0:,].values
Y = new_data.iloc[:,1].values


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.25,random_state =0)


# In[52]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =10,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[53]:


forest.score(X_train,Y_train)


# In[54]:


#100 trees


# In[55]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =100,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[56]:


forest.score(X_train,Y_train)


# In[57]:


#300 trees


# In[58]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators =300,criterion ="mae",random_state =0)
forest.fit(X_train,Y_train)


# In[59]:


forest.score(X_train,Y_train)

