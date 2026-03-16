#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# initialize image matrix
i = np.zeros((255,255,3))

# logic
i[:127,:127]=[0,255,255]
i[:127, 127:]=[255,0,255]
i[127:, :127]=[255,255,0]

# save image
cv2.imwrite('colored_2.png',i)

# show image
plt.figure(figsize=(3,3))
plt.imshow(i)
plt.title('colored_2')
plt.show()


# In[ ]:




