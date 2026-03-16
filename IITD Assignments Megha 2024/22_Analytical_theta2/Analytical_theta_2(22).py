#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


# given data
x= np.array([1,2,3,4,5])
y=np.array([2,3,4,5,6])


lhs= np.array([[len(y), np.sum(x)], [np.sum(x),np.sum(x**2)]])
rhs= np.array([np.sum(y), np.sum(x*y)])
soln= np.linalg.solve(lhs,rhs)

# o/p
print(soln)
# checking if thetas satisfy
np.allclose(np.dot(lhs,soln) , rhs)


# In[ ]:




