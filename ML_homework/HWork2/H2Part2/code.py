import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import random
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from visualization import error_VS_degree, FeaurVsError
from regularized_polynomial import do_ridgeWithPoly, do_lassoWithPoly
from polynomial_regression import do_polynomailReg
import warnings

warnings.filterwarnings('ignore')
np.random.seed(17)
random.seed(17)

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example")

parser.add_argument('--dataset', type=str, default='hWork2.csv')

parser.add_argument('--preprocessing', type=int, default=1, #P4
    help='0 for no processing, 1 for min/max, 2 for standrizing')

parser.add_argument('--choice', type=int, default=2,
    help="1 for simple linerar regression"
         "2 for polynomial on all featurs with degrre [1,2,3,4] with cross featurs"
         "3 for polynomial on all featurs with degrre [1,2,3,4] with monomial featurs"
         "4 individual feature test"
         "5 for find best lambda with grid search"
         "6 Lasso selection")

parser.add_argument('--extra_args', type=str, default='0,3,6', help="extra values passed")

args = parser.parse_args()

extra_args = args.extra_args
dataset = args.dataset
preprocessing = args.preprocessing
choice = args.choice




df = pd.read_csv("/home/ayelidry/Desktop/ML/ML_homework/HWork2/H2Part2/data2_200x30.csv")



X = np.array(df.drop(columns="Target"))
Y = np.array(df[["Target"]])



if choice == 1: # Simple linear regression
    # Manual split 50% - 50%
    X_train = X[:100]
    X_test = X[100:]
    Y_train = Y[:100]
    Y_test = Y[100:]
    
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    LReg = LinearRegression(fit_intercept=True).fit(X_train,Y_train)


    print("intercept: ", LReg.intercept_)
    Predict1 = LReg.predict(X_train)
    print("Train RMSE: ", np.sqrt(mean_squared_error(Predict1, Y_train)))
    Predict2 = LReg.predict(X_test)
    print("Test RMSE: ", np.sqrt(mean_squared_error(Predict2, Y_test)))


if choice == 2: # Polynomail regresion
    TrainError, TestError = do_polynomailReg(X,Y,4,True)    
    # Visualisation
    error_VS_degree(TrainError, TestError, 4)
    
    
if choice == 3: # Mononmail regression
    TrainError, TestError = do_polynomailReg(X,Y,4,False)
    # Visualisation
    error_VS_degree(TrainError, TestError, 4)


if choice == 4: # individual fetures
    Dgr3Error = []
    for i in range(9):
        F1TrErr, F1TsErr = do_polynomailReg(X[:,i].reshape(-1,1),Y,3,True)
        Dgr3Error.append([F1TrErr[2],F1TsErr[2]])
        
    
    FsErr = np.array(Dgr3Error)
    # print(f"FsError: {FsErr}")
    FeaurVsError(FsErr)
    
if choice == 5: # Ridge with all  Features and cross val
    do_ridgeWithPoly(X,Y)
    
if choice == 6: # Lasso Selection
    best_model = do_lassoWithPoly(X,Y)
    
    data=df.drop(columns="Target")
    select_featurs = SelectFromModel(best_model)
    selected_features = data.columns[(select_featurs.get_support())]
    print("selected_features: \n\t", selected_features)
    
    df_new = data[selected_features]
    X_new = np.array(df_new)
    do_ridgeWithPoly(X_new,Y, degree_=1)

