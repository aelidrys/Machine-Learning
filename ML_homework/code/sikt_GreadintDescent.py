import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# import os
# os.path()
from ML_homework.code.visualize import display_points

class ML_Tools:
    def cost(self, X, Y, W):
        exampels = X.shape[0]
        Pr = X.dot(W.T)
        print('Prediction: ', Pr)
        error = Pr - Y
        cost = error.T.dot(error) / (2 * exampels)
        print('Cost: ', cost[0][0])
        return cost[0][0]

data = pd.read_csv('./test_data.csv')
data = data.sort_values(by='f1')

F1 = np.array(data['f1']).reshape(-1,1)
F2 = np.array(data['f2']).reshape(-1,1)
ones = np.ones((7, 1))
target = np.array(data['label']).reshape(-1,1)
X = np.hstack([ones, F1, F2])
print(X.shape)

init_w = np.random.rand(7)
LReg = LinearRegression().fit(X, target, init_w) 
print(LReg.coef_)
Wights = np.array(LReg.coef_)
c = Wights[0][0]
m1 = Wights[0][1]
m2 = Wights[0][2]

ml = ML_Tools()
ml.cost(X, target, Wights)

# f1 = np.float64(input('enter f1: '))
# f2 = np.float64(input('enter f2: '))
# print('prediction: ', m1*f1+m2*f2+c)

# display_points(F1, target, 'F1', 'target')
