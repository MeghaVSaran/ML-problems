#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
gray_img= cv2.imread('grey_apple.jpg',cv2.IMREAD_GRAYSCALE)

print(gray_img.shape)
# show image
plt.imshow(gray_img, cmap='gray')
plt.show()

# calculate and show histogram
histg= cv2.calcHist([gray_img],[0],None,[256],[0,256])
plt.hist(histg.ravel(),edgecolor='black',bins=20)
plt.show()


# In[ ]:




