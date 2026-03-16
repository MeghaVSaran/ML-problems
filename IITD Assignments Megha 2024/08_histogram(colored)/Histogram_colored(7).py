#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
sky_img= cv2.imread('sky.png')

print(sky_img.shape)

# convert to RGB form
sky_img = cv2.cvtColor(sky_img, cv2.COLOR_BGR2RGB)

# show image
plt.imshow(sky_img)
plt.show()


# <h2>Histogram</h2>

# In[3]:


plt.figure(figsize=(8,6))
# calculate histograms of 3 colors separately
hist_r= cv2.calcHist([sky_img],[0],None,[256],[0,256])
hist_g= cv2.calcHist([sky_img],[1],None,[256],[0,256])
hist_b= cv2.calcHist([sky_img],[2],None,[256],[0,256])

# show histograms of 3 colors( separately )
plt.subplot(3,1,1)
plt.axis('off')
plt.hist(hist_r.ravel(), color='r',edgecolor='black',bins=12)

plt.subplot(3,1,2)
plt.axis('off')
plt.hist(hist_g.ravel(), color='g',edgecolor='black',bins=12)

plt.subplot(3,1,3)
plt.axis('off')
plt.hist(hist_b.ravel(), color='b',edgecolor='black',bins=12)


# In[ ]:




