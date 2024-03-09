import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from visualize import *

getcontext().prec = 10
# mileage = input("enter the mileage: ")
# print("mileage = ", mileage)

file_name = "/nfs/homes/aelidrys/Desktop/ML/linear_reg_42/data.csv"
data_csv = pd.read_csv(file_name)
data_csv.head(5)

data_csv = data_csv.sort_values(by="km")
X = data_csv["km"]
# X = data_csv["km"]  * 0.0001
Y = data_csv["price"]

def cost_function(m, c):
    sum_er = Decimal('0')
    for (x, y_gt) in zip(X,Y):
        y_pd = m * Decimal(x) + c
        sum_er += Decimal((y_pd - y_gt) ** 2)
    return (sum_er / len(Y) / 2)

def d_cost_dm(m, c):
    sum_d = Decimal('0')
    for (x,y_gt) in zip(X,Y):
        y_pd = Decimal(m) * Decimal(x) + c
        sum_d += Decimal((y_pd - y_gt) * Decimal(m))
        # print("driv = ", Decimal((y_pd - y_gt) * Decimal(m)))
    return (sum_d / len(Y))


def d_cost_dc(m, c):
    sum_d1 = Decimal('0')
    for (x,y_gt) in zip(X,Y):
        y_pd = Decimal(m) * Decimal(x) + c
        sum_d1 += Decimal(y_pd - y_gt)
    return (sum_d1 / len(Y))


def gradient_descent(df_x, df_y, i_x, i_y, lR = 0.00001, precision = 0.0001):
    cur_xy = np.array([Decimal(i_x), Decimal(i_y)])
    last_xy = np.array([Decimal('0'), Decimal('0')])
    learning_params = [cur_xy]
    it = 0 

    while norm(cur_xy - last_xy) > precision:
        last_xy = cur_xy.copy()
        grd_x = df_x(cur_xy[0], cur_xy[1]) * Decimal(lR)
        grd_y = df_y(cur_xy[0], cur_xy[1]) * Decimal(lR)
        grd_xy = np.array([grd_x, grd_y])
        # print("it = ",it,"grd_x: ", grd_x)
        cur_xy -= grd_xy
        learning_params.append(cur_xy)
        it += 1
    return learning_params


print("loc_minimum: ", d_cost_dm(0.1, 4), d_cost_dc(0.01, -2))

lr_prms = gradient_descent(d_cost_dm, d_cost_dc, 0.2, 0)
local_minimum = lr_prms[len(lr_prms)-1]
visualize_err(local_minimum[0], local_minimum[1], X, Y)

# visualize points
# visualize_points(X, Y, 0.01)

