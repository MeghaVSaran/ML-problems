#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2 
import matplotlib.pyplot as plt


# In[3]:


gray1=cv2.imread('cat1.jpg',cv2.IMREAD_GRAYSCALE)
gray2=cv2.imread('cat2.jpg',cv2.IMREAD_GRAYSCALE)
if(gray1 is None or gray2 is None):
    print('Error')


# In[4]:


plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title('CAT 1')
plt.axis('off')
plt.imshow(gray1, cmap='gray')
#plt.show()
#plt.figure(figsize=(5,3))
plt.subplot(1,2,2)
plt.title('CAT 2')
plt.axis('off')
plt.imshow(gray2, cmap='gray')

plt.show()
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
keypoints_1, descriptors_1 = sift.detectAndCompute(gray1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2,None)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key=lambda x:x.distance)
img3 = cv2.drawMatches(gray1, keypoints_1, gray2, keypoints_2, matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(9,5))
plt.imshow(img3, cmap='gray')
plt.show()

