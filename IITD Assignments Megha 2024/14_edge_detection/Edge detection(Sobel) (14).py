#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# In[4]:


#kernel def
sobel_fh= (1/8) * np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_fv=  (1/8) * np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

# read image
image =cv.imread('Building.jpg',cv.IMREAD_GRAYSCALE)

# show original image
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show()


# In[5]:


# sobel-vertical
ci = cv.filter2D(image, -1 ,sobel_fv)
plt.axis('off')
plt.imshow(ci, cmap='gray')
plt.show()


# In[6]:


# sobel-horizontal
ci = cv.filter2D(image, -1 ,sobel_fh)
plt.axis('off')
plt.imshow(ci, cmap='gray')
plt.show()


# In[ ]:




