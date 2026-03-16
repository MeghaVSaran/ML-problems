#!/usr/bin/env python
# coding: utf-8

# In[7]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# initialize image matrix
diag= np.zeros((255,255))

# logic (using for loop)
for i in range(127):
    for j in range(0,i):
        diag[i,j]=255

for i in range(127,255):
    for j in range(254-i , 0 , -1):
        diag[i,j]=255

for i in range(127):
    for j in range(254,254-i,-1):
        diag[i,j]=255

for i in range(127,255):
    for j in range(254,i,-1):
        diag[i,j]=255


# In[10]:


# save image
cv2.imwrite('b.png',diag)

# show image
plt.figure(figsize=(3,3))
plt.imshow(diag, cmap='gray')
plt.title('b')
plt.show()


# In[ ]:




