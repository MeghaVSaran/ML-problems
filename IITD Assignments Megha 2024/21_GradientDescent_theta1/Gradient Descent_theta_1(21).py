#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# <h2>Using Iterations</h2>

# In[20]:


#define cost function
def cost_fn(theta1,x,y):
    return 1/(2*len(y)) * np.sum((theta1*x-y)**2)

# define gradient descent function
def GD(x,y,theta1,m,alpha,iterations):
    cost_total=[]
    for _ in range(iterations):
        cost=cost_fn(theta1,x,y)
        gradient = np.sum((1/m)*(theta1*x-y)*x)
        theta1-= alpha*gradient
        cost_total.append(cost)

    return theta1,cost_total


# In[21]:


x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])

t1=1
alpha=0.01
i=10000

coeff,cost_total= GD(x,y,t1,len(y),alpha,i)
print(coeff)
plt.plot(range(len(cost_total)),cost_total)
plt.xlabel('iterations')
plt.ylabel('corresponding cost function')
plt.title('plot')
plt.show()


# <h2>Using Gradient Check(Threshold)</h2>

# In[22]:


def GD1(x,y,theta1,m,alpha):
    gradient = np.sum((1/m)*(theta1*x-y)*x)
    threshold= 10**-10
    while(abs(gradient)>=threshold):
        theta1-=alpha*gradient*(1/m)
        gradient = np.sum((1/m)*(theta1*x-y)*x)

    return theta1



# In[23]:


x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])

t1=1
alpha=0.01
i=1000

coeff= GD1(x,y,t1,len(y),alpha)
print(coeff)


# In[ ]:




