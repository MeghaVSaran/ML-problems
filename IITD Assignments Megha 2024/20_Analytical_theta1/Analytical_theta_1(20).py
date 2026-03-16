#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


x= np.array([1,2,3,4,5])
y= np.array([2,4,6,8,10])

#for min cost function
theta1 = np.sum(x*y) / np.sum(x**2)


# In[3]:
# another method: simultaneous equation

lhs= np.array([[np.sum(x**2)]])
rhs= np.array([np.sum(x*y)])
soln =np.linalg.solve(lhs,rhs)
print(soln)


# In[4]:


print(theta1)
np.allclose(np.dot(theta1,x), y)

