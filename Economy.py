#!/usr/bin/env python
# coding: utf-8

# In[254]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,Binarizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import(mean_absolute_error,mean_squared_error)
import joblib


# In[255]:


economy = pd.read_csv("Indian_Economy_Sectorwise_Dataset.csv")


# In[256]:


economy.shape


# In[257]:


economy.info()


# In[258]:


economy.isnull().sum()


# In[259]:


economy["Sector"].unique()


# In[260]:


economy["Sector"].value_counts()


# In[261]:


df = economy.groupby("Sector")["GDP_Lakh_Crore"].sum()


# In[262]:


df.plot(kind = "bar")


# In[263]:


df.plot(kind = "pie",autopct="%1.1f%%")


# In[264]:


economy["Year"].value_counts()


# In[265]:


a= economy.groupby("Year")["GDP_Lakh_Crore"].sum()


# In[266]:


a.plot(kind="bar")


# In[267]:


economy["Quarter"].unique()


# In[268]:


b=economy.groupby(["Year","Quarter"])["GDP_Lakh_Crore"].sum()


# In[269]:


b.plot(kind="bar")


# In[270]:


economy.groupby("Sector")["Growth_%"].sum()


# In[271]:


economy["gnp"]= economy["GDP_Lakh_Crore"] * economy["Imports_Crore"]- economy["Exports_Crore"]


# In[272]:


economy.head()


# In[273]:


economy.describe()


# In[274]:


l = LabelEncoder()


# In[275]:


m = MinMaxScaler()


# In[276]:


economy[["GDP_Lakh_Crore","Growth_%","gnp"]] = m.fit_transform(economy[["GDP_Lakh_Crore","Growth_%","gnp"]])


# In[277]:


economy


# In[278]:


economy["Quarter_new"] = l.fit_transform(economy["Quarter"])


# In[279]:


economy


# In[280]:


economy["Sector_new"] = l.fit_transform(economy["Sector"])


# In[281]:


economy


# In[282]:


economy.drop(columns=["Sector","Quarter"],inplace=True)


# In[283]:


economy


# In[284]:


economy["Growth_%"] = m.fit_transform(economy[["GDP_Lakh_Crore"]])


# In[285]:


economy


# In[286]:


economy["GNP"] = m.fit_transform(economy[["gnp"]])


# In[287]:


economy


# In[288]:


b=Binarizer(threshold = 2020)


# In[289]:


economy["year_new"] = b.fit_transform(economy[['Year']])


# In[290]:


economy


# In[291]:


economy["GDP_Lakh_Crore_new"]=b.fit_transform(economy[["GDP_Lakh_Crore"]])


# In[292]:


economy


# In[293]:


economy.drop(columns=["GDP_Lakh_Crore_new"],inplace=True)


# In[294]:


economy


# In[295]:


economy.drop(columns=["Year"],inplace=True)


# In[296]:


economy


# In[297]:


x = economy[["year_new"]]


# In[298]:


y = economy[["GDP_Lakh_Crore"]]


# In[299]:


x


# In[300]:


y


# In[301]:


x_train, x_test, y_train, y_test = train_test_split(
x,
y,
test_size=0.20,
random_state=42,
)


# In[302]:


x_train.shape


# In[303]:


x_test.shape


# In[304]:


y_train.shape


# In[305]:


y_test.shape


# In[306]:


model = LinearRegression()


# In[307]:


model.fit(x_train,y_train)


# In[308]:


print("Intercept:",model.intercept_)


# In[309]:


print("Coefficients:",model.coef_)


# In[310]:


y_pred =model.predict(x_test)


# In[311]:


y_pred


# In[312]:


mae = mean_absolute_error(y_test,y_pred)


# In[313]:


mae


# In[314]:


mse = mean_squared_error(y_test,y_pred)


# In[315]:


mse


# In[316]:


new_data = pd.DataFrame({"year_new":[2020]})


# In[318]:


prediction = model.predict(new_data)


# In[319]:


prediction


# In[320]:


joblib.dump(model,"lregression.pkl")


# In[ ]:




