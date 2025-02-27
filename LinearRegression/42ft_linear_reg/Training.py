import pandas as pd
import numpy as np
from visualize import *
from gradient_descent import GradientDescent
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse


arg_parse = argparse.ArgumentParser(description="simple argparser")
arg_parse.add_argument('--visualize', type=bool, default=False,
                       help=' add --visualize True to show the graph')

args = arg_parse.parse_args()
_visualize = args.visualize



file_name = "data.csv"
data_csv = pd.read_csv(file_name)
X = np.array(data_csv["km"]).reshape(-1,1)
Y = np.array(data_csv["price"]).reshape(-1,1)


# Train Val Split
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=88)


# Scaling
scaler_X = MinMaxScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_val = scaler_X.transform(X_val)


# Trianing with Gradient Descent
model = GradientDescent(fit_intercept=True,lr=1,pr=1e-09, max_itr=10000)
wights = model.fit(X_train, Y_train)
# print("GD wights: ", wights)


# Train Performence
Y_pr = model.predict(X_train)
print(f"----Train----------------")
print(f"\tGD.score: {model.score(X_train, Y_train)}\n")


# Test Performence
Y_pr = model.predict(X_val)
print(f"----Test-----------------")
print(f"\tGD.score: {model.score(X_val, Y_val)}")


# Visualize
if _visualize:
    X_scl = scaler_X.transform(X)
    Y_pr = model.predict(X_scl)
    visualize(X,Y,Y_pr)


# # Save Wights in file.csv
# Wights = wights[:,0]
# print(f"Wights: \n{Wights}")
# wights_df = pd.DataFrame({"wights":Wights})
# wights_df.to_csv("Wights.csv")
