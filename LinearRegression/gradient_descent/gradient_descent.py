import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

X = np.array([2.3,8.6,45,0.1])
Y = np.array([5,46,280,1.211])

def y_predicted(x, m, c):
    y_p = m * x + c
    return y_p

def cost():

    def cost_f(m,c):
        for x,y in zip(X, Y):
            y_p = y_predicted(x,m,c)
            sum += (y_p - y) ** 2
        return sum / 2 * len(X)

    def cost_dm(m, c):
        for x,y in zip(X, Y):
            y_p = y_predicted(x,m,c)
            sum += (y_p - y) * x
        return sum / len(X)

    def cost_dc(m, c):
        for x,y in zip(X, Y):
            y_p = y_predicted(x,m,c)
            sum += (y_p - y)
        return sum / len(X)
    
    def d_cost(m,c)
        return np.array([cost_dm(m,c), cost_dc(m, c)])

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
