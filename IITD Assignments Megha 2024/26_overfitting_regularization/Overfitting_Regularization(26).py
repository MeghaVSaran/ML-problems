#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Original data
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2, 4, 6, 8, 10, 12, 14])

# Add significant outliers
x_outliers = np.append(x, [8, 9, 10, 11, 12])
y_outliers = np.append(y, [30, 40, 60, 70, 90])

plt.scatter(x_outliers, y_outliers, color='red', label='Outliers')
plt.scatter(x, y, color='blue', label='Original Data')
plt.legend()
plt.show()

# Function to fit and plot polynomial regression
def plot_polynomial_regression(degree, x, y, x_outliers, y_outliers):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x_outliers.reshape(-1, 1))
    X_poly = X_poly / X_poly.max(axis=0)

    alpha = 0.001
    iterations = 50000
    l2_penalty = 0.1

    m, n = X_poly.shape
    weights = np.random.randn(n)

    # Gradient Descent
    for i in range(iterations):
        predictions = X_poly @ weights
        errors = predictions - y_outliers
        gradient = (X_poly.T @ errors + l2_penalty * weights) / m
        weights -= alpha * gradient

    x_fit = np.linspace(1, 12, 100)
    X_fit_poly = poly.transform(x_fit.reshape(-1, 1))
    X_fit_poly = X_fit_poly / X_fit_poly.max(axis=0)
    y_gd_fit = X_fit_poly @ weights

    coefficients = np.polyfit(x_outliers, y_outliers, degree)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x_fit)

    return x_fit, y_fit, y_gd_fit

x_fit, y_fit, y_gd_fit = plot_polynomial_regression(4, x, y, x_outliers, y_outliers)

# Plot original data with outliers
plt.scatter(x_outliers, y_outliers, color='red', label='Outliers')
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x_fit, y_fit, color='green', linestyle='--', label='4th Order Polynomial Fit')
plt.legend()
plt.title("4th Order Polynomial Fit")
plt.show()

# Plot overfitted and regularized fit together
plt.scatter(x_outliers, y_outliers, color='red', label='Outliers')
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x_fit, y_fit, color='green', linestyle='--', label='4th Order Polynomial Fit')
plt.plot(x_fit, y_gd_fit, color='orange', label='Regularized (GD) Fit')
plt.legend()
plt.title("Overfitting vs Regularized Fit")
plt.show()


# In[ ]:




