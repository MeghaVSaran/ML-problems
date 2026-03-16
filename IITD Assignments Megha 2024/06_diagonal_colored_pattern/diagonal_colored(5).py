#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# initialize matrix for image
img = np.zeros((255,255,3))

# logic
for i in range(0, 255):
    for j in range(0, 255):
        if i<j and i<128 and i+j<256:
            img[i, j]=[0,0,255]
        if i>j and j<128 and i+j<256:
            img[i, j]=[0,255,0]
        if j+i>256 and j>128 and i<j:
            img[i, j]=[255,0,0]

# save image
cv2.imwrite('diagonal_colored.png',img)

# show image
plt.figure(figsize=(3,3))
plt.imshow(img)
plt.title('diagonal_colored.png')
plt.show()


# In[ ]:




