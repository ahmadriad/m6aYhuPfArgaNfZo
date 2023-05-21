#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
data = pd.read_csv("C:/Users/97155/Downloads/term-deposit-marketing-2020.csv")
features = data.drop(["y"],axis=1)
target=pd.DataFrame(data["y"])
