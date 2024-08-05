import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sys
sys.path.append("..")
from linear_reg_42.visualize import visualize_err

X = np.array([[1, 2.3],
              [1, 8.6],
              [1, 10],
              [1, 15],
              [1, 0.1]])

Y = np.array([[5],[18],[26],[49],[1.211]])

def matrix_shape(matrix):
    matrix_shape = np.shape(matrix)
    print("pred: [{}, {}]".format(matrix_shape[0], matrix_shape[1]))



def cost_f(X, Y, W):
    exampels = X.shape[0]
    pred = np.dot(X, W)
    error = pred - Y 
    # cost = np.sum(error ** 2 / 2 * exampels)
    cost = error.T.dot(error) / 2 * exampels
    return cost

def f_derive(X, Y, W):
    exampels = X.shape[0]
    pred = np.dot(X, W)
    error = pred - Y
    gr = X.T @ error / exampels # X.T.dot(error) / exampels
    return gr



def gradient_descent(f_deriv, init, lr=0.01, pr=0.0000001):
    cur_p = init.reshape((2, 1))
    last_p = cur_p + 100 * pr
    list_lrn = [cur_p]

    iter = 0
    while norm(cur_p - last_p) > pr and iter < 300:
        last_p = cur_p.copy()
        gr = f_deriv(X, Y, cur_p)
        cur_p = (cur_p - gr) * lr
        list_lrn.append(cur_p)
        iter += 1
    print("iter = ", iter)
    return cur_p

learn_parm = gradient_descent(f_derive, np.array([1,0.5]))
m = learn_parm[1]
c = learn_parm[0]
print('m = {}, c = {}'.format(m, c))
visualize_err(m, c, X, Y)
