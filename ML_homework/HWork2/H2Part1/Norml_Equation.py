import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def normal_equations_solution(X, y): # (X.T @ X)^-1 @ X.T # Y
    X1 = np.dot(X.T,X)
    eq = np.linalg.inv(X1) @ X.T @ y
    return eq





