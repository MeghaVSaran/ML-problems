#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# In[2]:


# take input image
image =cv.imread('Building.jpg',cv.IMREAD_GRAYSCALE)

# show original image
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show()



# <h2>Convolution function (using for loop)</h2>

# In[3]:


def convolve(image,kernel):
    img_h,img_w=image.shape
    k_h,k_w= kernel.shape
    if(k_h==3 & k_w==3):
        output_height= img_h-2
        output_width = img_w-2
    else:
        output_height= img_h-k_h+1
        output_width= img_w-k_w+1
        
    output=np.zeros((output_height,output_width))
    for i in range(output_height):
        for j in range(output_width):
            superposition_region=image[ i:i+k_h , j:j+k_w]
            output[i,j]= np.sum(kernel*superposition_region)

    return output


# In[4]:


# convolution with blur filter(averaging)
blurr_filter = (1/9) * np.ones((3,3))
ci = convolve(image,blurr_filter)
# show output image
plt.imshow(ci, cmap='gray')
plt.axis('off')
plt.show()


# In[ ]:




