#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# <h2>Using In-built function equalizeHist()</h2>

# In[29]:


# Load the image
image = cv2.imread('grey_apple.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

#display the equalized image
equalized_hist = cv2.equalizeHist(image)
plt.subplot(1,2,2)
plt.title('Equalized Image')
plt.imshow(equalized_hist, cmap='gray')

plt.show()


# <h2>Equalization (without equalizeHist())</h2>

# In[16]:


def histogram_equalization(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Calculate CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize CDF for plotting

    # Mask all zeros (if any)
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Normalize CDF
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Fill masked values with 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Map the original gray levels in the image to equalized gray levels
    equalized_image = cdf[image]

    return equalized_image, hist, cdf_normalized

# Load an example image in grayscale
image = cv2.imread('grey_apple.jpg', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized_image, original_hist, original_cdf = histogram_equalization(image)
equalized_hist, _ = np.histogram(equalized_image.flatten(), 256, [0, 256])
equalized_cdf = equalized_hist.cumsum()
equalized_cdf_normalized = equalized_cdf * equalized_hist.max() / equalized_cdf.max()

# Display the original and equalized images
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')

# Display the histograms and CDFs
plt.subplot(2, 2, 3)
plt.title('Original Histogram and CDF')
plt.hist(image.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Histogram')
plt.plot(original_cdf, color='b', label='CDF')
plt.legend(loc='upper left')

plt.subplot(2, 2, 4)
plt.title('Equalized Histogram and CDF')
plt.hist(equalized_image.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Histogram')
plt.plot(equalized_cdf_normalized, color='b', label='CDF')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# In[ ]:




