#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


def harris_corner_detection(image, block_size=2, ksize=3, k=0.04):

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Calculate the gradient images
  Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
  Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

  # Calculate the covariance matrix
  Ix2 = Ix * Ix
  Iy2 = Iy * Iy
  Ixy = Ix * Iy

  # Apply Gaussian smoothing to the covariance matrix elements
  Ix2 = cv2.GaussianBlur(Ix2, (ksize, ksize), 0)
  Iy2 = cv2.GaussianBlur(Iy2, (ksize, ksize), 0)
  Ixy = cv2.GaussianBlur(Ixy, (ksize, ksize), 0)

  # Calculate the Harris corner response
  R = (Ix2 * Iy2 - Ixy * Ixy) - k * (Ix2 + Iy2) * (Ix2 + Iy2)

  # Apply thresholding to find potential corners
  threshold = 0.01 * np.max(R)
  corners = np.where(R > threshold)

  return list(zip(corners[1], corners[0]))


# In[4]:


# Load the image
image = cv2.imread("corn.jpg")

# Perform Harris corner detection
corners = harris_corner_detection(image)

# Draw the detected corners on the image
for corner in corners:
  cv2.circle(image, corner, 10, (0, 0, 255), -1)

# Display the result
plt.imshow(image)


# In[ ]:




