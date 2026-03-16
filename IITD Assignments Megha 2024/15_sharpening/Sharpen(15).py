#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

sharpening= np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

# read image
building_image=cv.imread('Building.jpg',cv.IMREAD_COLOR)
bi=cv.cvtColor(building_image, cv.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.title('original')
plt.imshow(bi)

# apply filter
i=cv.filter2D(bi, -1,sharpening)

plt.subplot(1,2,2)
plt.title('sharpened')
plt.imshow(i)
plt.show()


# In[ ]:




