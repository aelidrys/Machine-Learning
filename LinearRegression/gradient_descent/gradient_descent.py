import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

X = np.array([[1, 2.3],
              [1, 8.6],
              [1, 45],
              [1, 0.1]])

Y = np.array([[5],[46],[280],[1.211]])


def cost():

    def cost_f(X, Y, W):
        exampels = X.shape[0]
        pred = np.dot(X, W)
        error = pred - Y 
        # cost = np.sum(error ** 2 / 2 * exampels)
        cost = error.T.dot(error) / 2 * exampels

    def f_derive(X, Y, W):
        exampels = X.shape[0]
        pred = np.dot(X, W)
        error = pred - Y
        gr = X.T @ error / exampels # X.T.dot(error) / exampels



def gradient_descent(d_functs, init, lr=0.001, pr=0.0000001):
    cur_p = np.array(init)
    last_p = cur_p + 100 * pr
    list_lrn = [cur_p]

    iter = 0
    while norm(cur_p - last_p) > pr and iter < 200:
        last_p = cur_p.copy()
        gr = d_functs(cur_p)
        cur_p = cur_p - gr * lr
        list_lrn.append(cur_p)
        iter += 1
    return cur_p
