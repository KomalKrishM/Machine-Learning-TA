# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:53:42 2023

@author: komal
"""


import numpy as np
from matplotlib import pyplot as plt

# d = 4
# n = d+1
# m = [30, 100, 1000]
# var = [0, 10**-6, 10**-4]
# theta = np.resize(np.array([1,4,2,10,23]), (n,1))
# c = 0.5
# eps = 10**-6
# Tmax = 100

# def synth_data_gen(d, m):
#     return np.concatenate((np.ones((m,1)), np.random.randn(m,d)),axis=1)

# def GD(n, m, c, eps, Tmax, x, y, theta):
#     theta_est = np.zeros([n,1])
#     loss = []
#     stepsize = c/m
#     for i in range(0,Tmax):
#         var = theta_est
#         grad = 2*x.T@(x@var-y)
#         theta_est = var - stepsize*grad
#         # c = c/(i+1)
#         loss.append(np.linalg.norm(theta_est-theta)/np.linalg.norm(theta))
#         if loss[i]<=eps:
#             print("optimal solution reached within " + str(i) + " iterations")
#             break
#     return theta_est, loss

# def LS(x, y):
#     return np.linalg.inv(x.T@x)@x.T@y

# def PseudoInverse(x, y):
#     return np.linalg.pinv(x)@y

# var_loss = []
# for v in var:
#     mse_loss = []  
#     pinv_loss = []
#     GD_loss = []
    
#     for p in m:
#         print("True")
#         if v == 0:
#             x = synth_data_gen(d, p)
#             y = x@theta
#         else:
#             noise = np.random.normal(0, np.sqrt(v), (p,1))
#             x = synth_data_gen(d, p)
#             y = x@theta + noise
#         theta_est_by_mse =  LS(x, y) ###### least squares method
#         theta_est_by_pinv = PseudoInverse(x, y) ###### pseudo inverse method
#         theta_est_by_GD, loss = GD(n, p, c, eps, Tmax, x, y, theta) ###### gradient descent method
#         mse_loss.append(np.linalg.norm(theta_est_by_mse-theta)/np.linalg.norm(theta))
#         pinv_loss.append(np.linalg.norm(theta_est_by_mse-theta)/np.linalg.norm(theta))
#         GD_loss.append(loss[-1])
        
#     var_loss.append(pinv_loss)
    
#     plt.figure()
#     plt.plot(m,mse_loss,label = 'least squares', marker=matplotlib.markers.CARETDOWNBASE,markersize=15)
#     plt.plot(m,pinv_loss,label = 'pseudo inverse', marker=matplotlib.markers.CARETUPBASE,markersize=15)
#     plt.plot(m,GD_loss,label = 'gradient descent', marker="*",markersize=15)
#     plt.xlabel('Samples m')
#     plt.ylabel('Normalized Error')
#     if v==0:
#         plt.title('Normalized error without noise')
#     else:
#         plt.title('Normalized error with noise variance '+str(v))
#     plt.legend()
#     plt.xticks(m, labels=[str(30),str(100),str(1000)])
#     plt.savefig('./Synthetic data with noise variance {}.png'.format(v))
#     plt.show()

# plt.figure()
# plt.plot(m,var_loss[0],label= 'no noise')
# plt.plot(m,var_loss[1],label= 'noise var 10^-6')
# plt.plot(m,var_loss[2],label= 'noise var 10^-4')
# plt.xlabel('Samples m')
# plt.ylabel('Normalized Error')
# plt.title('Normalized error for pseudo inverse method')
# plt.legend()
# plt.xticks(m, labels=[str(30),str(100),str(1000)])
# plt.savefig('./Synthetic data error vs m with noise levels.png')
# plt.show()

#### Real Data
datContent = np.array([i.split() for i in open("C:/Users/komal/Downloads/airfoil+self+noise/airfoil_self_noise.dat").readlines()])

np.random.shuffle(datContent)
y = np.array([float(datContent[i,-1]) for i in range(len(datContent[:,-1]))])[:,np.newaxis]
x = np.zeros([1503,5])
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i,j] = float(datContent[i,j])


# y = (y-np.mean(y))/np.sqrt(np.var(y))
# for d in range(x.shape[1]):
#     x[:,d] = (x[:,d]-np.mean(x[:,d]))/np.sqrt(np.var(x[:,d]))

x = np.concatenate((np.ones((1503,1)), x),axis=1)

# n = 75
# m = [n, 3*n, 10*n]
# pred_err_train = []
# pred_err_test = []
# # pred = []
# for n_train in m:

#     y_train = y[:n_train,:]
#     x_train = x[:n_train,:]
#     # x_train = np.concatenate((np.ones((n_train,1)), x_train),axis=1)
#     y_test = y[n_train:,:]
#     x_test = x[n_train:,:]
#     # x_test = np.concatenate((np.ones((1503-n_train,1)), x_test),axis=1)
    
#     # theta_hat = np.linalg.pinv(x_train.T@x_train)@x_train.T@y_train
#     # theta_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train),x_train)),np.transpose(x_train)),y_train)
#     theta_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train),x_train)),np.transpose(x_train)),y_train)
#     # theta_hat = np.linalg.pinv(x_train)@y_train
#     y_hat_train = x_train@theta_hat
#     y_hat_test = x_test@theta_hat
    
#     pred_err_train.append((np.linalg.norm(y_train-y_hat_train, 2)/np.linalg.norm(y_train, 2))**2)
#     # pred.append(np.mean((y_train-y_hat_train)**2)/np.mean(y_train**2))
#     pred_err_test.append((np.linalg.norm(y_test-y_hat_test, 2)/np.linalg.norm(y_test, 2))**2)
    
# plt.figure()
# plt.plot(m,pred_err_train, label='training error')
# plt.plot(m,pred_err_test, label='test error')
# plt.xlabel('training size (m)')
# plt.ylabel('prediction error')
# plt.xticks(m)
# plt.title('generalization error for real data using least squares estimate')
# plt.legend()
# plt.show()
    

##### Gradient Descent Implementation

test_loss = []
n = 100
x_train = [x[0:n,:], x[0:3*n,:], x[0:10*n,:]]
y_train = [y[0:n], y[0:3*n], y[0:10*n]]
m_train = [x_train[0].shape[0], x_train[1].shape[0], x_train[2].shape[0]]
x_test = [x[n:,:], x[3*n:,:], x[10*n:,:]]
m_test = [x_test[0].shape[0], x_test[1].shape[0], x_test[2].shape[0]]
y_test = [y[n:], y[3*n:], y[10*n:]]
# x_train = x[0:n,:]
Kmax = 1000
# x = np.concatenate((np.ones((m,1)), x),axis=1)
c = 0.000000005
eps = 1e-6
par_est = np.zeros([n,1]) + 1e-3


for j in range(len(m_train)): 
    stepsize = c/m_train[j]
    loss = []
    for i in range(0,Kmax):
        par = par_est
        grad = 2*x_train[j].T@(x_train[j]@par-y_train[j])
        par_est = par - stepsize*grad
        loss.append(np.linalg.norm(y_train[j]-x_train[j]@par_est)/m_train[j])
        # loss.append(np.linalg.norm(par-par_est)**2/np.linalg.norm(par)**2)
        if np.linalg.norm(par-par_est)**2/np.linalg.norm(par)**2<=eps:
            print("optimal solution reached within " + str(i) + " iterations")
            break
        
    # test_loss.append(np.linalg.norm(y_test[j]-x_test[j]@par_est)/m_test[j])
    test_loss.append(np.linalg.norm(y_test[j]-x_test[j]@par_est)**2/np.linalg.norm(y_test[j])**2)
    plt.figure()
    plt.plot(np.arange(len(loss)),loss,label=str(m_train[j])+' samples')
    plt.xlabel('Iterations')
    plt.ylabel('Prediction Error')
    plt.title('Prediction error on real data')
    plt.legend()
    # plt.savefig('./Real data.png')
    plt.show()
        
# for k in range(len(m_test)):
#     test_loss.append(np.linalg.norm(y_test[k]-x_test[k]@par_est)/m_test[k])



# plt.figure()
# plt.plot(np.arange(len(test_loss)),test_loss)
# plt.xlabel('Samples m')
# plt.ylabel('Prediction Error')
# plt.title('Prediction error on real data')
# # plt.legend()
# # plt.savefig('./Real data.png')
# plt.show()
