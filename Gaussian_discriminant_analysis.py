# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:05:17 2024

@author: komal
"""

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def synthetic_data(mean_0, mean_1, cov, m, l):
    y = np.random.binomial(1, 0.5, (m, 1))
    x = np.ones((m, l))
    x[y[:, 0] == 0, 1:] = np.random.multivariate_normal(mean_0, cov, np.sum(y == 0))
    x[y[:, 0] == 1, 1:] = np.random.multivariate_normal(mean_1, cov, np.sum(y == 1))
    return x,y

# Parameter estimation
def parameter_estimation(x, y):
    phi_hat = np.sum(y == 1) / len(y)

    mu_0_hat = np.mean(x[y[:, 0] == 0, 1:], axis=0)
    mu_1_hat = np.mean(x[y[:, 0] == 1, 1:], axis=0)

    sigma2_hat_0 = np.sum((x[y[:, 0] == 0, 1:] - mu_0_hat) ** 2, axis=0)
    sigma2_hat_1 = np.sum((x[y[:, 0] == 1, 1:] - mu_1_hat) ** 2, axis=0)
    sigma2_hat   = (sigma2_hat_0 + sigma2_hat_1)/len(y)
    return phi_hat, mu_0_hat, mu_1_hat, sigma2_hat

# Calculate classification accuracy
def cl_accuracy(y, y_hat):
    return (np.sum(y_hat == y) / y.shape[0]) * 100

# Prediction
def predict(x, mu_0_hat, mu_1_hat, sigma2_hat):
    x_norm_1 = np.sum((x - mu_1_hat) ** 2 / sigma2_hat, axis=1)
    x_norm_0 = np.sum((x - mu_0_hat) ** 2 / sigma2_hat, axis=1)
    return (x_norm_1 < x_norm_0).astype(int)[:,np.newaxis]

def main_synthetic():
    mean_0 = [-1, -1]
    mean_1 = [1, 1]
    cov    = [[1, 0], [0, 10]]
    l      = 3

    m_train = 10000
    m_test  = 100

    x_train, y_train = synthetic_data(mean_0, mean_1, cov, m_train, l)
    x_test, y_test   = synthetic_data(mean_0, mean_1, cov, m_test, l)

    phi_hat, mu_0_hat, mu_1_hat, sigma2_hat = parameter_estimation(x_train, y_train)

    y_pred_train = predict(x_train[:, 1:], mu_0_hat, mu_1_hat, sigma2_hat)
    train_acc = cl_accuracy(y_train, y_pred_train)

    y_pred_test = predict(x_test[:, 1:], mu_0_hat, mu_1_hat, sigma2_hat)
    test_acc = cl_accuracy(y_test, y_pred_test)

    print("Synthetic data - Training Accuracy: {} Test Accuracy: {}".format(train_acc, test_acc))


def extract_digits(x, y, d_1, d_2):
    x_t = {'d_1': [], 'd_2': []}
    y_t = {'d_1': [], 'd_2': []}
    for i, d in enumerate(y):
        if d == d_1:
            x_t['d_1'].append(x[i].reshape(-1, 1))
            y_t['d_1'].append(0)
        elif d == d_2:
            x_t['d_2'].append(x[i].reshape(-1, 1))
            y_t['d_2'].append(1)
    return x_t, y_t

# Prepare data for training and testing
def prepare_data(data_dict, label_dict):
    num_samples = len(data_dict['d_1']) + len(data_dict['d_2'])
    data_matrix = np.ones((num_samples, 785))
    labels = np.array(label_dict['d_1'] + label_dict['d_2'])[:, np.newaxis] * 1.

    for i in range(num_samples):
        digit = 'd_1' if i < len(data_dict['d_1']) else 'd_2'
        data_matrix[i, 1:] = data_dict[digit][i - len(data_dict['d_1'])].T

    return data_matrix, labels

def main_real():
    # Load MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    d_1, d_2 = 0, 9

    X_Train_0_9, Y_Train_0_9 = extract_digits(X_train, Y_train, d_1, d_2)
    X_Test_0_9, Y_Test_0_9   = extract_digits(X_test, Y_test, d_1, d_2)

    x_train_0_9, Y_train_0_9 = prepare_data(X_Train_0_9, Y_Train_0_9)
    x_test_0_9, Y_test_0_9   = prepare_data(X_Test_0_9, Y_Test_0_9)

    # Keep only useful features 
    train_feature_var = np.var(x_train_0_9, axis = 0)
    X_train_0_9 = x_train_0_9[:, train_feature_var > 0]
    X_test_0_9  = x_test_0_9[:, train_feature_var > 0]

    print("Training data shape:", X_train_0_9.shape, Y_train_0_9.shape)
    print("Testing data shape:", X_test_0_9.shape, Y_test_0_9.shape)

    phi_hat, mu_0_hat, mu_1_hat, sigma2_hat = parameter_estimation(X_train_0_9, Y_train_0_9)

    y_pred_train = predict(X_train_0_9[:, 1:], mu_0_hat, mu_1_hat, sigma2_hat)
    train_acc = cl_accuracy(y_pred_train, Y_train_0_9)

    y_pred_test = predict(X_test_0_9[:, 1:], mu_0_hat, mu_1_hat, sigma2_hat)
    test_acc = cl_accuracy(y_pred_test, Y_test_0_9)

    print("Real data - Training Accuracy: {} Test Accuracy: {}".format(train_acc, test_acc))

synthetic = 1

if __name__ == "__main__":
    if synthetic == 1:
       main_synthetic()
    else:
       main_real()


