# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:13:20 2024

@author: komal
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

def synthetic_data(theta, d, m):
    var = 0.01*np.linalg.norm(theta)**2
    noise = np.sqrt(var)*np.random.randn(m, 1)
    x = np.concatenate((np.ones((m,1)), np.random.randn(m,d)), axis=1)
    y = x@theta + noise
    return x, y

def least_squares_ridge(x, y, lamda):
    return np.linalg.inv(x.T @ x + lamda*np.eye(x.shape[1])) @ x.T @ y

def gradient_descent_dense(x, y, theta, c, eps, Tmax):
    m, n = x.shape
    theta_est = np.zeros((n, 1))
    loss = []
    stepsize = c / m
    for _ in range(Tmax):
        grad = 2 * x.T @ (x @ theta_est - y) + 2 *theta_est
        theta_est -= stepsize * grad
        loss.append(np.linalg.norm(theta_est - theta)**2 / np.linalg.norm(theta)**2)
        if loss[-1] <= eps:
            print(f"Optimal solution reached within {_ + 1} iterations")
            break
    return theta_est, loss

def shrink(input_, theta_):
    theta_ = np.maximum( theta_, 0.0 )
    return np.sign(input_) * np.maximum( np.abs(input_) - theta_, 0.0 )

def gradient_descent_sparse_synthetic(x, y, theta, c, eps, Tmax, thr_):
    m, n = x.shape
    theta_est = np.zeros((n, 1))
    loss = []
    stepsize = c / m
    for _ in range(Tmax):
        grad = 2 * x.T @ (x @ theta_est - y) 
        theta_est -= stepsize * grad
        theta_est = shrink(theta_est, thr_)
        loss.append(np.linalg.norm(theta_est - theta)**2 / np.linalg.norm(theta)**2)
        if loss[-1] <= eps:
            print(f"Optimal solution reached within {_ + 1} iterations")
            break
    return theta_est, loss

def gradient_descent_sparse_real(x, y, c, eps, Tmax, thr_):
    m, n = x.shape
    theta_est = np.zeros((n, 1))
    loss = []
    stepsize = c / m
    for _ in range(Tmax):
        grad = 2 * x.T @ (x @ theta_est - y) 
        theta_est -= stepsize * grad
        theta_est = shrink(theta_est, thr_)
        y_pred = x@theta_est
        loss.append(np.linalg.norm(y_pred - y)**2 / np.linalg.norm(y)**2)
        if loss[-1] <= eps:
            print(f"Optimal solution reached within {_ + 1} iterations")
            break
    return theta_est, loss, y_pred

def main_synthetic():

    d = 199
    m = 100
    
    ###### few important features #######
    theta_sparse = np.ones((d+1,1))
    theta_sparse[1:] = np.random.binomial(1, 0.5, (d,1))*np.random.randn(d,1)
    
    x_lasso, y_lasso = synthetic_data(theta_sparse, d, m)
    x_t_lasso, y_t_lasso = synthetic_data(theta_sparse, d, m)
    
    c     = 0.05
    eps   = 1e-6
    Tmax  = 500
    lam   = 0.01
    scale = (np.linalg.norm(x_lasso, 2) ** 2 )*1.001
    thr   = lam/scale
    theta_sparse_est, sparse_loss = gradient_descent_sparse_synthetic(x_lasso, y_lasso, theta_sparse, c, eps, Tmax, thr)
    
    plt.figure()
    plt.plot(np.arange(len(sparse_loss)), sparse_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Parameter error')
    # plt.title('')
    plt.show()
    
    lamda = 0.1
    theta_dense_est = least_squares_ridge(x_lasso, y_lasso, lamda)
    dense_loss      = np.linalg.norm(theta_sparse-theta_dense_est)**2/np.linalg.norm(theta_sparse)**2
    
    ###### sk-learn on sparse data ####### 
    reg = linear_model.LinearRegression()
    reg_lasso = linear_model.Lasso(alpha=0.1)
    reg_ridge = linear_model.Ridge(alpha=0.1)
    
    reg.fit(x_lasso, y_lasso)
    reg_lasso.fit(x_lasso, y_lasso)
    reg_ridge.fit(x_lasso, y_lasso)
    
    theta = reg.coef_.T
    theta_lasso = reg_lasso.coef_[np.newaxis,:].T
    theta_ridge = reg_ridge.coef_.T
    print(theta_ridge.shape)
    
    reg_par_loss = np.linalg.norm(theta-theta_sparse)**2/np.linalg.norm(theta_sparse)**2
    reg_lasso_par_loss = np.linalg.norm(theta_lasso-theta_sparse)**2/np.linalg.norm(theta_sparse)**2
    reg_ridge_par_loss = np.linalg.norm(theta_ridge-theta_sparse)**2/np.linalg.norm(theta_sparse)**2
    
    print('reg_par_loss {}'.format(reg_par_loss))
    print('reg_lasso_par_loss {}'.format(reg_lasso_par_loss))
    print('reg_ridge_par_loss {}'.format(reg_ridge_par_loss))
    
    reg_pred = x_t_lasso@theta
    reg_lasso_pred = x_t_lasso@theta_lasso
    reg_ridge_pred = x_t_lasso@theta_ridge
    
    reg_pred_loss = np.linalg.norm(reg_pred-y_t_lasso)**2/np.linalg.norm(y_t_lasso)**2
    reg_lasso_pred_loss = np.linalg.norm(reg_lasso_pred-y_t_lasso)**2/np.linalg.norm(y_t_lasso)**2
    reg_ridge_pred_loss = np.linalg.norm(reg_ridge_pred-y_t_lasso)**2/np.linalg.norm(y_t_lasso)**2    
    
    print('reg_pred_loss {}'.format(reg_pred_loss))
    print('reg_lasso_pred_loss {}'.format(reg_lasso_pred_loss))
    print('reg_ridge_pred_loss {}'.format(reg_ridge_pred_loss))
    
    # ###### dense features #######
    theta_dense = np.ones((d+1,1))
    theta_dense[1:] = np.random.randn(d,1)
    
    x_ridge, y_ridge = synthetic_data(theta_dense, d, m)
    x_t_ridge, y_t_ridge = synthetic_data(theta_dense, d, m)
    
    c     = 0.01
    eps   = 1e-6
    Tmax  = 500
    lam   = 0.01
    scale = (np.linalg.norm(x_ridge, 2) ** 2 )*1.001
    thr   = lam/scale
    theta_sparse_est, sparse_loss = gradient_descent_sparse_synthetic(x_ridge, y_ridge, theta_dense, c, eps, Tmax, thr)
    
    plt.figure()
    plt.plot(np.arange(len(sparse_loss)), sparse_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Parameter error')
    # plt.title('')
    plt.show()
    
    lamda = 0.01
    theta_dense_est = least_squares_ridge(x_ridge, y_ridge, lamda)
    dense_loss      = np.linalg.norm(theta_dense-theta_dense_est)**2/np.linalg.norm(theta_dense)**2
    print('Dense loss {}'.format(dense_loss))
    
    ###### sk-learn on dense data####### 
    reg = linear_model.LinearRegression()
    reg_lasso = linear_model.Lasso(alpha=0.1)
    reg_ridge = linear_model.Ridge(alpha=0.1)
    
    reg.fit(x_ridge, y_ridge)
    reg_lasso.fit(x_ridge, y_ridge)
    reg_ridge.fit(x_ridge, y_ridge)
    
    theta = reg.coef_.T
    theta_lasso = reg_lasso.coef_[np.newaxis,:].T
    theta_ridge = reg_ridge.coef_.T
    print(theta_ridge.shape)
    
    reg_par_loss = np.linalg.norm(theta-theta_dense)**2/np.linalg.norm(theta_dense)**2
    reg_lasso_par_loss = np.linalg.norm(theta_lasso-theta_dense)**2/np.linalg.norm(theta_dense)**2
    reg_ridge_par_loss = np.linalg.norm(theta_ridge-theta_dense)**2/np.linalg.norm(theta_dense)**2
    
    print('reg_par_loss {}'.format(reg_par_loss))
    print('reg_lasso_par_loss {}'.format(reg_lasso_par_loss))
    print('reg_ridge_par_loss {}'.format(reg_ridge_par_loss))
    
    reg_pred = x_t_ridge@theta
    reg_lasso_pred = x_t_ridge@theta_lasso
    reg_ridge_pred = x_t_ridge@theta_ridge
    
    reg_pred_loss = np.linalg.norm(reg_pred-y_t_ridge)**2/np.linalg.norm(y_t_ridge)**2
    reg_lasso_pred_loss = np.linalg.norm(reg_lasso_pred-y_t_ridge)**2/np.linalg.norm(y_t_ridge)**2
    reg_ridge_pred_loss = np.linalg.norm(reg_ridge_pred-y_t_ridge)**2/np.linalg.norm(y_t_ridge)**2    
    
    print('reg_pred_loss {}'.format(reg_pred_loss))
    print('reg_lasso_pred_loss {}'.format(reg_lasso_pred_loss))
    print('reg_ridge_pred_loss {}'.format(reg_ridge_pred_loss))

###### Real data #######
def main_real():        
        
    file_path_1  = "C:/Users/user/Downloads/blogfeedback/blogData_test-2012.02.01.00_00.csv"
    train_data_1 = np.array([np.array(line.split(sep=',')).astype(float) for line in open(file_path_1).readlines()])
    
    file_path_2  = "C:/Users/user/Downloads/blogfeedback/blogData_test-2012.02.02.00_00.csv"
    train_data_2 = np.array([np.array(line.split(sep=',')).astype(float) for line in open(file_path_2).readlines()])
    
    train_data = np.concatenate((train_data_1, train_data_2), axis=0)
    x_train = train_data[:,:-1]
    y_train = train_data[:,-1][:,np.newaxis]
    
    lamda = 10
    theta_dense_est = least_squares_ridge(x_train, y_train, lamda)
    
    y_pred_dense = x_train@theta_dense_est
    training_error_dense = np.linalg.norm(y_pred_dense - y_train)**2/np.linalg.norm(y_train)**2
    print('training error on ridge model {}'.format(training_error_dense))
    
    c     = 0.000000005
    eps   = 1e-6
    Tmax  = 3000
    lam   = 1
    scale = (np.linalg.norm(x_train, 2) ** 2 )*1.001
    thr   = lam/scale
    theta_sparse_est, training_error_sparse, y_pred_sparse = gradient_descent_sparse_real(x_train, y_train, c, eps, Tmax, thr)
    
    plt.figure()
    plt.plot(np.arange(len(training_error_sparse)), training_error_sparse)
    plt.xlabel('Iterations')
    plt.ylabel('Parameter error')
    # plt.title('')
    plt.show()
    
    ###### sk-learn on real data####### 
    reg = linear_model.LinearRegression()
    reg_lasso = linear_model.Lasso(alpha=0.1)
    reg_ridge = linear_model.Ridge(alpha=0.1)
    
    reg.fit(x_train, y_train)
    reg_lasso.fit(x_train, y_train)
    reg_ridge.fit(x_train, y_train)
    
    theta = reg.coef_.T
    theta_lasso = reg_lasso.coef_[np.newaxis,:].T
    theta_ridge = reg_ridge.coef_.T
    print(theta.shape)
    
    ####### testing #######
    
    file_path_3 = "C:/Users/user/Downloads/blogfeedback/blogData_test-2012.02.03.00_00.csv"
    test_data   = np.array([np.array(line.split(sep=',')).astype(float) for line in open(file_path_3).readlines()])
    
    x_test = test_data[:,:-1]
    y_test = test_data[:,-1][:,np.newaxis]
    
    y_test_dense_pred = x_test@theta_dense_est
    test_error_dense  = np.linalg.norm(y_test_dense_pred - y_test)**2/np.linalg.norm(y_test)**2
    print('test error on ridge model {}'.format(test_error_dense))
    
    y_test_sparse_pred = x_test@theta_sparse_est
    test_error_sparse  = np.linalg.norm(y_test_sparse_pred - y_test)**2/np.linalg.norm(y_test)**2
    print('test error on lasso model {}'.format(test_error_sparse))
    
    reg_pred = x_test@theta
    reg_lasso_pred = x_test@theta_lasso
    reg_ridge_pred = x_test@theta_ridge
    
    reg_pred_loss = np.linalg.norm(reg_pred-y_test)**2/np.linalg.norm(y_test)**2
    reg_lasso_pred_loss = np.linalg.norm(reg_lasso_pred-y_test)**2/np.linalg.norm(y_test)**2
    reg_ridge_pred_loss = np.linalg.norm(reg_ridge_pred-y_test)**2/np.linalg.norm(y_test)**2    
    
    print('reg_pred_loss {}'.format(reg_pred_loss))
    print('reg_lasso_pred_loss {}'.format(reg_lasso_pred_loss))
    print('reg_ridge_pred_loss {}'.format(reg_ridge_pred_loss))

synthetic = 0

if __name__ == "__main__":
    if synthetic == 0:
        main_synthetic()
    else:
        main_real()
