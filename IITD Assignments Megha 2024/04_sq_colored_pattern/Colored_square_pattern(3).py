#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# initialize iamge matrix(triplet)
color1= np.zeros((255,255,3))

# logic
color1[:127, :127] = [255, 0, 0]
color1[127:,:127 ] = [0, 0, 255]
color1[:127,127: ]=[0,255,0]
color1[127:, 127:] = 255

# save image
cv2.imwrite('colored_img_sq.png',color1)

# show image
plt.figure(figsize=(3,3))
plt.imshow(color1)
plt.title('colored_img_sq')
plt.show()


# In[ ]:




