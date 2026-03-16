#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# <h2>METHOD 1</h2>

# In[2]:


x= np.array([[1],[2],[3],[4],[5]])
y=np.array([[2],[4],[6],[8],[10]])
x_t = np.transpose(x)

theta= np.linalg.pinv(x_t@x)@(x_t@y)
print(theta)


# <h2>METHOD 2</h2>

# In[3]:


x= np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
y= np.array([[2],[4],[6],[8],[10]])
x_t = np.transpose(x)

theta= np.linalg.inv(x_t@x)@x_t@y
print(theta)


# In[ ]:




