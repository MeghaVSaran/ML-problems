#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(theta, x, y):
    m = len(y)
    h = sigmoid(x @ theta)
    return (1/m) * (x.T @ (h - y))

def cost_fn(theta, x, y):
    m = len(y)
    h = sigmoid(x @ theta)
    J = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J

# Gradien Descent Function
def gd(theta, x, y, alpha, num_iterations):
    cost_arr = []
    for i in range(num_iterations):
        cost_val = cost_fn(theta, x, y)
        cost_arr.append(cost_val)
        grad_theta = gradient(theta, x, y)
        theta -= alpha * grad_theta
        if abs(grad_theta)< 10**(-8):
            break;
    return theta, cost_arr

# Input data
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([[0], [1], [0], [0], [1]])


theta = np.zeros((x.shape[1], 1))

alpha = 0.01
num_iterations = 1000

# Perform gradient descent
theta, cost_arr = gd(theta, x, y, alpha, num_iterations)

print("Theta value:\n", theta)
print("Number of iterations:", len(cost_arr))

# Plot cost function over iterations
plt.plot(range(len(cost_arr)), cost_arr, label='Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.legend()
plt.show()


# In[ ]:




