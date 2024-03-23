# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:44:46 2024

@author: komal
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle

# np.random.seed(26)
###### simulated data ##########
mean_0 = [-1, -1]
mean_1 = [1, 1]
cov = [[1, 0], [0, 10]]
l = 3
m = 1000 
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

test_loss = []
n = 100
x_train = [x[0:n,:], x[0:3*n,:], x[0:5*n,:]]
y_train = [y[0:n], y[0:3*n], y[0:5*n]]
m_train = [x_train[0].shape[0], x_train[1].shape[0], x_train[2].shape[0]]

Kmax = 20

c = 0.05
eps = 1e-6


def sigmoid(z):
    z_hat = 1/(1+np.exp(-z))
    return z_hat

def cl_accuracy(y, y_hat):
 return (np.sum(y_hat == y)/y.shape[0])*100

def GD_Logistic_Reg(Kmax, x_train, y_train, l, eps):
    par_est = np.zeros([l,1]) + 1e-3
    accuracy = []
    for i in range(0,Kmax):
        par = par_est
        y_train_hat = sigmoid(x_train@par)
        grad = (y_train_hat - y_train).T@x_train
        par_est = par - stepsize*grad.T
        y_hat = sigmoid(x_train@par_est)
        y_hat[y_hat>=0.5] = 1
        y_hat[y_hat<0.5] = 0
        Accuracy = cl_accuracy(y_train,y_hat)
        accuracy.append(Accuracy)
        if np.linalg.norm(par-par_est)**2/np.linalg.norm(par)**2<=eps:
            print("optimal solution reached within " + str(i) + " iterations")
            break
    return par_est, accuracy

# for j in range(len(m_train)): 
#     stepsize = c/m_train[j]
#     par_est, accuracy = GD_Logistic_Reg(Kmax, x_train[j], y_train[j], l, eps)

#     y_test_hat = sigmoid(x_test@par_est)
#     y_test_hat[y_test_hat>=0.5] = 1
#     y_test_hat[y_test_hat<0.5] = 0
#     test_loss.append(cl_accuracy(y_test,y_test_hat))
#     plt.figure()
#     plt.plot(np.arange(Kmax),accuracy,label=str(m_train[j])+' samples')
#     plt.xlabel('Iterations')
#     plt.ylabel('Training accuracy')
#     plt.title('Training accuracy on simulated data')
#     plt.legend()
#     # plt.savefig('./Simulated data.png')
#     plt.show()
    
#########  Real data ########

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

###### Data processing for digits 0 and 9
X_train_0_9 = {'0':[],'9':[]}
Y_train_0_9 = {'0':[],'9':[]}
for i in range(len(Y_train)):
  if Y_train[i] == 0:
    X_train_0_9['0'].append(X_train[i].reshape(-1,1))
    Y_train_0_9['0'].append(0)
  elif Y_train[i] == 9:
    X_train_0_9['9'].append(X_train[i].reshape(-1,1))
    Y_train_0_9['9'].append(1)

X_test_0_9 = {'0':[],'9':[]}
Y_test_0_9 = {'0':[],'9':[]}
for i in range(len(Y_test)):
  if Y_test[i] == 0:
    X_test_0_9['0'].append(X_test[i].reshape(-1,1))
    Y_test_0_9['0'].append(0)
  elif Y_test[i] == 9:
    X_test_0_9['9'].append(X_test[i].reshape(-1,1))
    Y_test_0_9['9'].append(1)
    
X_Train_0_9 = np.zeros((len(X_train_0_9['0'])+len(X_train_0_9['9']), 785))
Y_Train_0_9 = np.array(Y_train_0_9['0']+ Y_train_0_9['9'])[:,np.newaxis]*1.
X_Test_0_9 = np.zeros((len(X_test_0_9['0'])+len(X_test_0_9['9']), 785))
Y_Test_0_9 = np.array(Y_test_0_9['0']+ Y_test_0_9['9'])[:,np.newaxis]*1.
    
for i in range(X_Train_0_9.shape[0]):
  if i <= len(X_train_0_9['0'])-1:
    X_Train_0_9[i,0] = 1
    X_Train_0_9[i,1:][np.newaxis] = (X_train_0_9['0'][i]).T
  elif i >= len(X_train_0_9['0']):
    X_Train_0_9[i,0] = 1
    X_Train_0_9[i,1:][np.newaxis] = (X_train_0_9['9'][i-len(X_train_0_9['0'])]).T

X_Train_0_9, Y_Train_0_9 = shuffle(X_Train_0_9, Y_Train_0_9)

m_Train = 800
X_Train_0_9 = X_Train_0_9[:m_Train,:]
Y_Train_0_9 = Y_Train_0_9[:m_Train,:]

for i in range(X_Test_0_9.shape[0]):
  if i <= len(X_test_0_9['0'])-1:
    X_Test_0_9[i,0] = 1
    X_Test_0_9[i,1:][np.newaxis] = (X_test_0_9['0'][i]).T
  elif i >= len(X_test_0_9['0']):
    X_Test_0_9[i,0] = 1
    X_Test_0_9[i,1:][np.newaxis] = (X_test_0_9['9'][i-len(X_test_0_9['0'])]).T
    
# for j in range(len(m_train)): 
stepsize = c/m_Train
par_est, accuracy = GD_Logistic_Reg(Kmax, X_Train_0_9, Y_Train_0_9, l, eps)

y_test_hat = sigmoid(X_Test_0_9@par_est)
y_test_hat[y_test_hat>=0.5] = 1
y_test_hat[y_test_hat<0.5] = 0
test_loss.append(cl_accuracy(Y_Test_0_9,y_test_hat))
plt.figure()
plt.plot(np.arange(Kmax),accuracy,label=str(m_Train)+' samples')
plt.xlabel('Iterations')
plt.ylabel('Training accuracy')
plt.title('Training accuracy on simulated data')
plt.legend()
# plt.savefig('./Real data.png')
plt.show()
    
