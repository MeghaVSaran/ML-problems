#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#code for image color

# using for loop
img=np.zeros((255,255), dtype=np.uint8)
for i in range(255):
    for j in range(127,255):
        img[i,j]=255


# In[6]:


# using slicing
img=np.zeros((255,255))
img[:, 127:] = 255


# In[7]:


#saving the image
cv2.imwrite('half_black_half_white.png',img)


# In[12]:


#showig the image
plt.figure(figsize=(3,3))
plt.imshow(img, cmap='gray')
plt.title('Half Black Half White Image')
plt.show()


# <h1>Diagonal Pattern</h1>

# In[13]:


# initialize matrix
i1= np.zeros((255,255), dtype=np.uint8)

# logic for diagonal white&black (for loop)
for i in range(127):
    for j in range(127):
        i1[i,j]=255

for i in range(127,255):
    for j in range(127,255):
        i1[i,j]=255

# save image
cv2.imwrite('diagonal_black_n_white.png',i1)

# show image generated
plt.figure(figsize=(3,3))
plt.imshow(i1, cmap='gray')
plt.title('diagonal_black_n_white')
plt.show()


# In[ ]:




