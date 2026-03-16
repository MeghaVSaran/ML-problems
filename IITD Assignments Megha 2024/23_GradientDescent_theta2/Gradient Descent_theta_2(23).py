#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# <h2>Using Iterations</h2>

# In[2]:


# define cost function
def cost_fn(theta0,theta1,x,y):
    J = 1/(2*len(y))*np.sum((theta0 + theta1*x - y)**2)
    return J

# define GD
def GD(x,y,theta1,theta0,m,alpha,iterations):
    cost_array=[]
    for _ in range(iterations):
        grad_theta1 = (1/m) * np.sum(theta1*(x**2) + theta0*x -x*y)
        grad_theta0 = (1/m)* np.sum(theta0 + theta1*x-y)

        theta0-= alpha*grad_theta0
        theta1-= alpha*grad_theta1
        cost = cost_fn(theta0,theta1,x,y)
        cost_array.append(cost)

    return theta0,theta1,cost_array



# In[3]:


x= np.array([1,2,3,4,5])
y= np.array([2,3,4,5,6])
alpha=0.01

t0,t1,cf= GD(x,y,0,1,len(y),alpha,1000)
print(t0,'\n',t1)
plt.plot(range(1000),cf)
plt.xlabel('iterations')
plt.ylabel('cost function')
plt.show()


# <h2>Using Threshold for Gradient</h2>

# In[10]:


def cost_fn(t0,t1,x,y):
    return (1/((2*len(y))) * np.sum((t0+t1*x-y)**2))

def GD_algo(t0,t1,x,y,m,alpha):
    m=len(y)
    grad_t0 = (1/m)* np.sum(t0+t1*x-y)
    grad_t1= (1/m)* np.sum((t0+t1*x-y)*x)
    threshold = 10**(-8)
    while(abs(grad_t0)>threshold or abs(grad_t1)>threshold):
        t0 -= (alpha/m)*grad_t0
        t1 -= (alpha/m)*grad_t1
        grad_t0 = (1/m)* np.sum(t0+t1*x-y)
        grad_t1= (1/m)* np.sum((t0+t1*x-y)*x)
        
    return t0,t1



# In[11]:


x= np.array([1,2,3,4,5])
y= np.array([2,3,4,5,6])
alpha=0.03

t0,t1= GD_algo(0,0,x,y,len(y),alpha)
print(t0,'\n',t1)


# In[ ]:




