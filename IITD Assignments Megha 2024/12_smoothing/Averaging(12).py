#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# input image
image =cv.imread('Downloads/Building.jpg',cv.IMREAD_GRAYSCALE)

# show original image
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show()

# convolution function
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



# In[11]:


# Average kernel definitions
blurr_f = (1/9) * np.ones((3,3))
gaussian_f = (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]])

# convolution / filter
bi = convolve(image,blurr_f)
gi=convolve(image,gaussian_f)

plt.imshow(bi, cmap='gray')
plt.title('Average Filter')
plt.axis('off')


# In[12]:


# Show Blur filtered image
plt.title('Gaussian Filter')
plt.axis('off')
plt.imshow(gi, cmap='gray')
plt.show()


# In[13]:


#using in-buit function
ci = cv.GaussianBlur(image,(7,7), 0)
plt.imshow(ci, cmap='gray')
plt.axis('off')
plt.show()


# In[ ]:




