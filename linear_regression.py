# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:53:42 2023

@author: komal
"""

import numpy as np
import matplotlib.backend_bases
from matplotlib import pyplot as plt


def synth_data_gen(d, m):
    return np.concatenate((np.ones((m,1)), np.random.randn(m,d)), axis=1)

def least_squares(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y

def pseudo_inverse(x, y):
    return np.linalg.pinv(x) @ y

def gradient_descent(x, y, theta, c, eps, Tmax):
    m, n = x.shape
    theta_est = np.zeros((n, 1))
    loss = []
    stepsize = c / m
    for _ in range(Tmax):
        grad = 2 * x.T @ (x @ theta_est - y)
        theta_est -= stepsize * grad
        loss.append(np.linalg.norm(theta_est - theta)**2 / np.linalg.norm(theta)**2)
        if loss[-1] <= eps:
            print(f"Optimal solution reached within {_ + 1} iterations")
            break
    return theta_est, loss

def main_synthetic():
    theta = [1,4,2,10,23]
    d = 4
    n = d+1
    train_len = [30, 100, 1000]
    var = [0, 10**-6, 10**-4]

    theta = np.resize(np.array(theta), (n,1))
    c = 0.5
    eps = 10**-6
    Tmax = 100

    var_loss = []
    for v in var:
        mse_loss = []  
        pinv_loss = []
        gd_loss = []
        
        for p in train_len:
            if v == 0:
                x = synth_data_gen(d, p)
                y = x@theta
            else:
                noise = np.random.normal(0, np.sqrt(v), (p,1))
                x = synth_data_gen(d, p)
                y = x@theta + noise

            theta_est_by_mse =  least_squares(x, y) 
            theta_est_by_pinv = pseudo_inverse(x, y) 
            theta_est_by_GD, loss = gradient_descent(x, y, theta, c, eps, Tmax)

            mse_loss.append(np.linalg.norm(theta_est_by_mse-theta)**2/np.linalg.norm(theta)**2)
            pinv_loss.append(np.linalg.norm(theta_est_by_pinv-theta)**2/np.linalg.norm(theta)**2)
            gd_loss.append(loss[-1])
            
        var_loss.append(pinv_loss)
        
        labels = ['least squares', 'pseudo inverse', 'gradient descent']
        plot_error(train_len, [mse_loss, pinv_loss, gd_loss], labels,
                      title=f'Normalized error with noise variance {v}',
                      xlabel='Samples m', ylabel='Normalized Error')

    labels = [f'Noise variance {v}' for v in var]
    plot_error(train_len, [var_loss[0], var_loss[1], var_loss[2]], labels,
                      title=f'Normalized error for pseudo inverse method',
                      xlabel='Samples m', ylabel='Normalized Error')

def plot_error(m_values, errors, labels, title, xlabel, ylabel):
    markers = [matplotlib.markers.CARETDOWNBASE, matplotlib.markers.CARETUPBASE, "*"]
    plt.figure()
    for i, error in enumerate(errors):
        plt.plot(m_values, error, label=labels[i], marker=markers[i], markersize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(m_values, labels=[str(m_val) for m_val in m_values])
    plt.show()

#### Real Data
def load_data(file_path):
    datContent = np.array([i.split() for i in open(file_path).readlines()])
    np.random.shuffle(datContent)
    y = np.array([float(datContent[i,-1]) for i in range(len(datContent[:,-1]))])[:,np.newaxis]
    x = np.zeros([1503,5])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = float(datContent[i,j])
    return x, y

def gradient_descent_real(x, y, c, eps, Kmax):
    m, n = x.shape[0], x.shape[1]
    par_est = np.zeros([n,1]) + 1e-3
    stepsize = c / m
    for _ in range(Kmax):
        par = par_est
        grad = 2 * x.T @ (x @ par - y)
        par_est = par - stepsize * grad
        # loss.append(np.linalg.norm(y - x @ par_est)**2 / m)
        if np.linalg.norm(par - par_est)**2 / np.linalg.norm(par)**2 <= eps:
            print("estimating parameter does not change after " + str(_) + " iterations")
            break
    return par_est

def main_real():
    file_path = "/content/airfoil_self_noise.dat"
    x, y = load_data(file_path)
    m = y.shape[0]
    m_train = int(0.8*m)
    m_test = m - m_train
    m_t = [m_train//10, m_train//3, m_train]
    y_test = y[m_train:,:]
    x_test = x[m_train:,:]

    pred_err_train_ls = []
    pred_err_test_ls = []
    pred_err_train_gd = []
    pred_err_test_gd = []

    Kmax = 1000
    c = 0.0000000005
    eps = 1e-6

    for n_train in m_t:
        y_train = y[:n_train,:]
        x_train = x[:n_train,:]

        theta_hat_ls = least_squares(x_train, y_train)
        theta_hat_gd = gradient_descent_real(x_train, y_train, c, eps, Kmax)

        y_hat_train_ls = x_train @ theta_hat_ls
        y_hat_test_ls = x_test @ theta_hat_ls

        y_hat_train_gd = x_train @ theta_hat_gd
        y_hat_test_gd = x_test @ theta_hat_gd

        pred_err_train_ls.append(np.linalg.norm(y_train - y_hat_train_ls, 2)**2 / n_train)
        pred_err_train_gd.append(np.linalg.norm(y_train - y_hat_train_gd, 2)**2 / n_train)
        pred_err_test_ls.append(np.linalg.norm(y_test - y_hat_test_ls, 2)**2 / m_test)
        pred_err_test_gd.append(np.linalg.norm(y_test - y_hat_test_gd, 2)**2 / m_test)

    labels = ['least squares train', 'gradient descent train', 'least squares test', 'gradient descent test']
    plot_error_real(m_t, [pred_err_train_ls, pred_err_train_gd, pred_err_test_ls, pred_err_test_gd], 
               labels, title=f'linear regression on real data', 
               xlabel='training samples', ylabel='prediction error')
    
def plot_error_real(m_values, errors, labels, title, xlabel, ylabel):
    markers = [matplotlib.markers.CARETDOWNBASE, matplotlib.markers.CARETUPBASE, matplotlib.markers.CARETRIGHTBASE, matplotlib.markers.CARETLEFTBASE]
    plt.figure()
    for i, error in enumerate(errors):
        plt.plot(m_values, error, label=labels[i], marker=markers[i], markersize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(m_values, labels=[str(m_val) for m_val in m_values])
    plt.show()
    
synthetic_data = 1
    
if __name__ == "__main__":
    if synthetic_data == 1:
        main_synthetic()
    else:
        main_real()
