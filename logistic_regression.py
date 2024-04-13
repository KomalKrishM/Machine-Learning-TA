# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:44:46 2024

@author: komal
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

# Generate synthetic data
def synthetic_data(mean_0, mean_1, cov, m, l):
    y = np.random.binomial(1, 0.5, (m, 1))
    x = np.ones((m, l))
    x[y[:, 0] == 0, 1:] = np.random.multivariate_normal(mean_0, cov, np.sum(y == 0))
    x[y[:, 0] == 1, 1:] = np.random.multivariate_normal(mean_1, cov, np.sum(y == 1))
    return x, y

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute classification accuracy
def cl_accuracy(y, y_hat):
    return (np.sum(y_hat == y) / y.shape[0]) * 100

# Gradient Descent for Logistic Regression
def GD_Logistic_Reg(Kmax, x_train, y_train, l, eps, stepsize):
    par_est = np.random.rand(l, 1)  # Random initialization creates randomness in the output even for real data
    accuracy = []

    for i in range(Kmax):
        par = par_est
        y_train_hat = sigmoid(x_train @ par)
        grad = (y_train_hat - y_train).T @ x_train
        par_est = par - stepsize * grad.T
        y_hat = sigmoid(x_train @ par_est)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        Accuracy = cl_accuracy(y_train, y_hat)
        accuracy.append(Accuracy)

        if np.linalg.norm(par - par_est) ** 2 / np.linalg.norm(par) ** 2 <= eps:
            print("Optimal solution reached within " + str(i) + " iterations")
            break

    return par_est, accuracy

# Iterate over different training data sizes
def main_synthetic():

    # Parameters
    mean_0 = [-1, -1]
    mean_1 = [1, 1]
    cov = [[1, 0], [0, 10]]
    l = 3
    m_train = 10000
    m_test = 100
    Kmax = 200
    c = 0.05
    eps = 1e-6

    # Generate train and test data
    x_train, y_train = synthetic_data(mean_0, mean_1, cov, m_train, l)
    x_test, y_test = synthetic_data(mean_0, mean_1, cov, m_test, l)

    test_loss = []

    # Different sizes of training data
    x_train_sizes = [x_train[0:m_train//10, :], x_train[0:m_train//3, :], x_train]
    y_train_sizes = [y_train[0:m_train//10], y_train[0:m_train//3], y_train]
    m_train_sizes = [x_train_sizes[0].shape[0], x_train_sizes[1].shape[0], x_train_sizes[2].shape[0]]

    plt.figure()
    for j in range(len(m_train_sizes)):
        stepsize = c / m_train_sizes[j]
        par_est, accuracy = GD_Logistic_Reg(Kmax, x_train_sizes[j], y_train_sizes[j], l, eps, stepsize)

        # Test the model on test data
        y_test_hat = sigmoid(x_test @ par_est)
        y_test_hat[y_test_hat >= 0.5] = 1
        y_test_hat[y_test_hat < 0.5] = 0
        test_loss.append(cl_accuracy(y_test, y_test_hat))

        # Plot training accuracy
        plt.plot(np.arange(len(accuracy)), accuracy, label=str(m_train_sizes[j]) + ' samples')
    plt.xlabel('Iterations')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy on simulated data')
    plt.legend()
    plt.show()

    print("Synthetic - Test Accuracy for training size of %s : %s" % (m_train_sizes,test_loss))

###### Data processing for digits 0 and 9
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
    c    = 3
    eps  = 1e-6
    Kmax = 200

    X_Train_0_9, Y_Train_0_9 = extract_digits(X_train, Y_train, d_1, d_2)
    X_Test_0_9, Y_Test_0_9   = extract_digits(X_test, Y_test, d_1, d_2)

    x_train_0_9, Y_train_0_9 = prepare_data(X_Train_0_9, Y_Train_0_9)
    x_test_0_9, Y_test_0_9   = prepare_data(X_Test_0_9, Y_Test_0_9)

    # Keep only useful features 
    train_feature_var = np.var(x_train_0_9, axis = 0)
    X_train_0_9 = x_train_0_9[:, train_feature_var > 0]
    X_test_0_9  = x_test_0_9[:, train_feature_var > 0]

    # Normalization to avoid runtime warning in sigmoid function
    X_train_0_9 /= 255.0
    X_test_0_9  /= 255.0

    print("Training data shape:", X_train_0_9.shape, Y_train_0_9.shape)
    print("Testing data shape:", X_test_0_9.shape, Y_test_0_9.shape)

    stepsize = c/X_train_0_9.shape[0]
    par_est, accuracy = GD_Logistic_Reg(Kmax, X_train_0_9, Y_train_0_9, X_train_0_9.shape[1], eps, stepsize)

    plt.figure()
    plt.plot(np.arange(len(accuracy)),accuracy)
    plt.xlabel('Iterations')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy on simulated data')
    # plt.legend()
    # plt.savefig('./Real data.png')
    plt.show()

    # Test the model on test data
    y_test_hat = sigmoid(X_test_0_9 @ par_est)
    y_test_hat[y_test_hat >= 0.5] = 1
    y_test_hat[y_test_hat < 0.5]  = 0
    test_acc = cl_accuracy(Y_test_0_9, y_test_hat)

    print("Real data - Test Accuracy: {}".format(test_acc))

synthetic = 0

if __name__ == "__main__":
   if synthetic == 1:
      main_synthetic()
   else:
      main_real()
