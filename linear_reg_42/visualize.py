import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

def visualize_err(m, c, X, Y):
    # calculate y = m * x + c
    # X = X * 0.001
    # Y = Y * 0.001
    line_y = []
    for x in X:
        line_y.append(Decimal(m) * Decimal(x) + c)
    print(line_y)
    plt.plot(X,Y, marker='o', label='current points')
    plt.plot(X,line_y, label='prediction line',color='red')
    plt.title('MSE')
    plt.xlabel("mileage")
    plt.ylabel("price")
    plt.legend()
    plt.show()

def visuale(lear_prms):
    x_set = [item[0] for item in lear_prms]
    y_set = [item[1] for item in lear_prms]

    print("x_set: ", x_set)
    print("y_set: ", y_set)
    plt.plot(x_set, y_set, label="f(x)")
    plt.title("gresient descent")
    plt.xlabel('X')
    plt.ylabel('Y')

    for i, (xp, yp) in enumerate(lear_prms):
        # yp = fun(xp)
        color = 'ro' if i%2 else 'bo'
        plt.plot(xp,yp,color)
    plt.show()

def visualize_points(X, Y, lR):
    X = X * lR
    Y = Y * lR
    plt.title("ft_liear_regression")
    plt.xlabel("mileage")
    plt.ylabel("price")
    for (x,y) in zip(X,Y):
        plt.plot(x,y,"bo")
    plt.show()
