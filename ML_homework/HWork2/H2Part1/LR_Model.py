import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from linear_regression import LinearReg
from visualization import display_points, costs_VS_iters, featurs_Vs_target
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Norml_Equation import normal_equations_solution

import warnings
warnings.filterwarnings('ignore')

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example")

parser.add_argument('--dataset', type=str, default='hWork2.csv')

parser.add_argument('--preprocessing', type=int, default=1, #P4
    help='0 for no processing, 1 for min/max, 2 for standrizing')

parser.add_argument('--choice', type=int, default=2,
    help="0 for linerar verification" #P0
         "1 for training wih all featurs" #P1 / P3 / P7
         "2 for training with the best featurs" #P5
         "3 for normal equation" #p6
         "4 for sikit")

parser.add_argument('--step_size', type=float, default=0.01,help="Learning Rate default(0.01)")

parser.add_argument('--precision', type=float, default=0.0001, help="Precision defualt(0.0001)")

parser.add_argument('--max_iter', type=float, default=1000, help="number of iteration to learn defualt(1000)")

args = parser.parse_args()

# print(f'data_set: {args.dataset} | processe: {args.preprocessing}')


dataset = args.dataset
preprocessing = args.preprocessing
choice = args.choice
step_size = args.step_size
precision = args.precision
max_iter = args.max_iter

LR = LinearReg()

if choice == 0:
    # Early verifications 
    x = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t=5+x
    ones = np.ones((5,1))
    x = x.reshape(-1,1)
    t = t.reshape(-1,1)
    x = np.hstack([ones, x])
    wights, iter, wights_list = LR.gradient_descent(x,t,step_size, precision,max_iter)
    print(f"cost after learning: {LR.cost(x,t,wights)}\n")
    print(f"wights: {wights}\n")
    y_p = LR.predict(x,wights)
    display_points(x[:,1],y_p)
    exit()



df = pd.read_csv("/home/ayelidry/Desktop/ML/ML_homework/HWork2/dataset_200x4_regression.csv")
x = np.array(df[['Feat1', 'Feat2', 'Feat3']])

if preprocessing == 1:
    scaler = MinMaxScaler().fit(x)
    x = scaler.transform(x)
    
if preprocessing == 2:
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    
ones = np.ones(x.shape[0]).reshape(-1,1)
x = np.hstack([ones,x])
t = np.array(df[['Target']])

if choice == 1: # Trian with All Featurs
    wights , iter, wights_list = LR.gradient_descent(x,t,step_size,precision)
    print(f"wights: {wights[:,0]}")


if choice == 2: # Trian with Best Featurs
    df1 = df[['Feat1', 'Target']]
    df2 = df[['Feat2', 'Target']]
    df3 = df[['Feat3', 'Target']]
    featurs_Vs_target(df1,df2,df3)
    x1 = x[:,:2]
    wights , iter, wights_list = LR.gradient_descent(x1,t,step_size,precision)
    print(f"wights: {wights[:,0]}")
    

if choice == 3: # Normal Equation
    wights = normal_equations_solution(x,t)
    print(f"wights: {wights}")


if choice == 4: # Normal Equation
    LReg = LinearRegression(fit_intercept=False).fit(x,t)
    wights = LReg.coef_
    print(f"wights: {wights}")
    




# # HYPERPARAMETERS TUNING   
# step_sizes = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001])
# precisions = np.array([0.01, 0.001, 0.0001, 0.00001])
# parms = []
# itr = 0
# for lr in step_sizes:
#     for prec in precisions:
#         wights , iter, wights_list = LR.gradient_descent(x,t,lr,prec)
#         parms.append([wights, iter,lr,prec, wights_list])
#         itr += 1

# cost = LR.cost(x,t,parms[0][0])
# # print("parms", parms[0][0])
# parm = None
# iterations = 0
# w_list = None
# for prm in parms:
#     print("cost: ", LR.cost(x,t,prm[0]))
#     if LR.cost(x,t,prm[0]) < cost:
#         cost = LR.cost(x,t,prm[0])
#         parm = prm
#     if prm[1] > 10 and prm[1] < 1000:
#         iterations = prm[1]
#         w_list = prm[4]

# print(f"step_size = {prm[2]:.10f}".rstrip("0").rstrip("."))
# print(f"precision = {prm[3]:.10f}".rstrip("0").rstrip("."))
# print(f"minimum cost = {cost} | iterations = {prm[1]}")

# # COST VISUALISATION
# costs = []
# for w in w_list:
#     costs.append(LR.cost(x,t,w))
# costs_VS_iters(costs, iterations)






