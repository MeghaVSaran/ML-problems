#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# input image
building_image=cv.imread('Building.jpg',cv.IMREAD_COLOR)
bi=cv.cvtColor(building_image, cv.COLOR_BGR2RGB)

# define kernels
derivative_fh=np.array([[1,-1]])
derivative_fv=np.array([[1],[-1]])

# apply filter
ci = cv.filter2D(bi, -1 ,derivative_fh)
plt.axis('off')
plt.imshow(ci, cmap='gray')
plt.show()


# In[15]:


ci = cv.filter2D(bi, -1 ,derivative_fv)
plt.axis('off')
plt.imshow(ci)
plt.show()


# In[ ]:




