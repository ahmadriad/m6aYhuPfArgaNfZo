#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/97155/Downloads/term-deposit-marketing-2020.csv")
label_encoder = LabelEncoder()
columns_to_encode = [1, 2, 3, 4, 6, 7, 8,10,13]

for column in columns_to_encode:
    data.iloc[:, column] = label_encoder.fit_transform(data.iloc[:, column])
plt.hist(data[data.columns[5]],color="black")

plt.xlabel("balance")
plt.ylabel("Count")
plt.xlim(0,30000)
px.box(data,x=data["y"],y=data[data.columns[0]]).update_traces(marker_color="black").update_layout(template="seaborn")


# In[5]:


corr=data.corr()
sb.heatmap(corr,annot=True, fmt=".2f")


# In[ ]:




