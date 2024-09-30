import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_points(X, Y, xlabel='X', ylabel='Y'):
    print("----display_points----")
    for x, y in zip(X, Y):
        plt.scatter(x, y, color='red')
    plt.plot(X, Y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()