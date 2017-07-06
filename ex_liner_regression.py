#!/usr/bin/python
# -*- coding:utf-8 -*-

__author__ = 'boris 2017/6/22'
import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge

def regression(data, alpha, lamda, num_iter):
    n = len(data[0]) - 1
    theta = np.zeros(n)
    # print theta.shape
    for times in range(num_iter):
        for d in data:
            # print d.items[0]
            x = d[:-1]
            y = d[-1]
            # print 'data: ', data
            # print 'x: ', x
            # print y
            g = np.dot(theta, x) - y
            theta = theta - alpha * g * x + lamda * theta
            # print times, theta
    return theta

def compute_error(data, theta):
    x = data[:, 0:2]
    y = data[:, 2]
    # print theta.shape
    g = np.dot(x, theta) - y
    totalError = g**2
    totalError = np.sum(totalError, axis=0)
    mse = totalError/float(len(data))
    return mse

def plot_data(data,theta):
    x = data[:, 0:2]
    y = data[:, 2]
    y_predict = np.dot(x, theta)
    pylab.plot(x[:,1], y, 'o')
    pylab.plot(x[:,1], y_predict, 'k-')
    pylab.show()

if __name__ == "__main__":
    # pandas
    data = np.loadtxt('data.csv', delimiter=',')
    x_one = np.ones(len(data[:, 0]))    # add b item
    data = np.column_stack((x_one, data))
    # calc theta by gradient descent
    alpha = 0.00001   # learning rate, be used for update gradient
    lamda = 0.00001
    num_iter = 50
    theta = regression(data, alpha, lamda, num_iter)
    print theta

    x = data[:, 0:2]
    y = data[:, 2]
    y_hat = theta * x
    mse = compute_error(data, theta)
    # print y, y_hat
    print 'mse: ', mse

    plot_data(data, theta)
