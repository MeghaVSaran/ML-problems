#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[7]:

#  input
np.random.seed(29)
random_array = np.random.rand(20,3)*50
print('Original Array: \n',random_array)

variance = np.var(random_array, axis=0)
min_arr= np.min(random_array)

# normalization
norm_array = (random_array-min_arr)/ variance
print('\n\n')

# print and plot output 
print('Normalized Array: \n',norm_array)
plt.plot(random_array,norm_array)


# In[ ]:




