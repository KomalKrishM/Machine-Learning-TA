# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:05:17 2024

@author: komal
"""

import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(26)
###### simulated data ##########
mean_0 = [-1, -1]
mean_1 = [1, 1]
cov = [[1, 0], [0, 10]]
l = 3
m = 10000
y = np.zeros((m,1))
x = np.zeros((m,l))

for i in range(m):
  y[i] = np.random.binomial(1, 0.5)
  if y[i] == 0:
    x[i,0] = 1
    x[i,1:] = np.random.multivariate_normal(mean_0, cov)
    # y[i] = -1
  else:
    x[i,0] = 1
    x[i,1:] = np.random.multivariate_normal(mean_1, cov)
    # y[i] = 1
    
# print(x)
# print(y)
# plt.figure()
# plt.scatter(x_train[:,1],x_train[:,2],c=y_train,marker='x')
# plt.show()

# print(x_train[5])
# print(y_train[5])
# X_Train = x_train*y_train
# print(X_Train[5])

m_test = 100
y_test = np.zeros((m_test,1))
x_test = np.zeros((m_test,l))
for i in range(m_test):
  y_test[i] = np.random.binomial(1, 0.5)
  if y_test[i] == 0:
    x_test[i,0] = 1
    x_test[i,1:] = np.random.multivariate_normal(mean_0, cov)
    # y_test[i] = -1
  else:
    x_test[i,0] = 1
    x_test[i,1:] = np.random.multivariate_normal(mean_1, cov)
    # y_test[i] = 1

# plt.figure()
# plt.scatter(x_test[:,1],x_test[:,2],c=y_test,marker='x')
# plt.show()

# n = 100
# x_train = [x[0:n,:], x[0:3*n,:], x[0:5*n,:]]
# y_train = [y[0:n], y[0:3*n], y[0:5*n]]
# m_train = [x_train[0].shape[0], x_train[1].shape[0], x_train[2].shape[0]]

def cl_accuracy(y, y_hat):
 return (np.sum(y_hat == y)/y.shape[0])*100

phi_hat = np.sum(y == 1)/m

y_0_len = np.sum(y==0)
y_1_len = np.sum(y==1)

mu_0_hat = 0
mu_1_hat = 0

for i in range(m):
    if y[i] == 0:
        mu_0_hat += x[i,1:]
    elif y[i] == 1:
        mu_1_hat += x[i,1:]

mu_0_hat /= y_0_len
mu_1_hat /= y_1_len 
n = 2

sigma2_hat = np.zeros((n,1))
for j in range(sigma2_hat.shape[0]):
    for i in range(m):
        if y[i] == 0:
            sigma2_hat[j,0] += (x[i,j+1] - mean_0[j])**2
        elif y[i] == 1:
            sigma2_hat[j,0] += (x[i,j+1] - mean_1[j])**2
    sigma2_hat[j,0] /= m
    
y_pred_train = np.zeros((m,1))
for i in range(m):
    x_norm_1 = 0
    x_norm_0 = 0
    for j in range(n):
        x_norm_1 += (x[i,j+1] -mu_1_hat[j])**2/sigma2_hat[j]
        x_norm_0 += (x[i,j+1] -mu_0_hat[j])**2/sigma2_hat[j]
    if x_norm_1 < x_norm_0:
        y_pred_train[i] = 1
    else:
        y_pred_train[i] = 0
        
train_acc = cl_accuracy(y_pred_train, y)

y_pred = np.zeros((m_test,1))
for i in range(m_test):
    x_norm_1 = 0
    x_norm_0 = 0
    for j in range(n):
        x_norm_1 += (x_test[i,j+1] -mu_1_hat[j])**2/sigma2_hat[j]
        x_norm_0 += (x_test[i,j+1] -mu_0_hat[j])**2/sigma2_hat[j]
    if x_norm_1 < x_norm_0:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        
test_acc = cl_accuracy(y_pred, y_test)

