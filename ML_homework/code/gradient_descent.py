import numpy as np
from numpy.linalg import norm



def cost_f(X, Y, W):
    exampels = X.shape[0]
    pred = np.dot(X, W)
    error = pred - Y 
    cost = error.T.dot(error) / 2 * exampels
    return cost

def f_derive(X, Y, W):
    exampels = X.shape[0]
    pred = np.dot(X, W)
    error = pred - Y
    gr = X.T @ error / exampels # X.T.dot(error) / exampels
    return gr



def grad_descent(X, Y, W1, lr=0.01, pr=0.000000001):
    cur_p = W1
    last_p = cur_p + 100
    list_lrn = [cur_p]

    iter = 0
    while norm(cur_p - last_p) > pr and iter < 10000:
        last_p = cur_p.copy()
        gr = f_derive(X, Y, cur_p)
        cur_p = (cur_p - gr) * lr
        list_lrn.append(cur_p)
        iter += 1
    cost = cost_f(X, Y, cur_p)[0][0]
    print(f'cost function: {cost:.2f}%')
    return cur_p