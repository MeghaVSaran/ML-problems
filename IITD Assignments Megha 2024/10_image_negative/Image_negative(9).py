#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# iamge read
gray=cv2.imread('fruits_blck_n_white.jpg',cv2.IMREAD_GRAYSCALE)
print(gray)
print(gray.shape)

# show original image
plt.imshow(gray, cmap='gray')
plt.axis('off')


# In[4]:


# logic
negative_gray=255-gray

# show output
print(negative_gray)
plt.imshow(negative_gray, cmap='gray')
plt.show()


# In[ ]:




