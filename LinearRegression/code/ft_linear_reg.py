import pandas as pd
import numpy as np
from visualize import *
from gradient_descent import grad_descent, f_derive, cost_f
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


file_name = "/nfs/homes/aelidrys/Desktop/ml/LinearRegression/data_sets/car_price.csv"
data_csv = pd.read_csv(file_name)
data_csv = data_csv.sort_values(by="km")

X_raw = np.array(data_csv["km"]).reshape(24,1)
Y_raw = np.array(data_csv["price"]).reshape(24,1)


ones = np.ones((24, 1))
f2 = X_raw.reshape(24, 1)
scaler_X = MinMaxScaler().fit(f2)
f2_scld = scaler_X.transform(f2)
X_scld = np.hstack([ones,f2_scld])
scaler_Y = MinMaxScaler().fit(Y_raw)
Y_scld = scaler_Y.transform(Y_raw)

# unscaled the wights
def unscld_wights(w_scld, scaler_X, scaler_Y):
    X_min = scaler_X.data_min_
    X_max = scaler_X.data_max_
    Y_min = scaler_Y.data_min_
    Y_max = scaler_Y.data_max_
    wights = w_scld
    wights[1] = wights[1] * ((Y_max-Y_min) / (X_max-X_min))
    wights[0] = wights[0] * (Y_max-Y_min) + Y_min - (wights[1]*Y_min)
    return wights

# generate random wights between 0 and 1
init_wights = np.random.rand(2,1)

# Gradient Descent
wights_scld = grad_descent(X_scld, Y_scld, init_wights, lr=1)
wights = unscld_wights(wights_scld, scaler_X, scaler_Y)
_m = wights[1]
_c = wights[0]




# sklearn Model
reg = LinearRegression().fit(X_scld, Y_scld)
wights = unscld_wights(reg.coef_[0], scaler_X, scaler_Y)
m = wights[1]
c = wights[0]



#### testing #####
km = np.array(data_csv['km']).reshape(24, 1)
price = np.array(data_csv['price']).reshape(24, 1)
data = np.hstack([km, price])
indexs = np.random.choice(data.shape[0], 10, replace=False)
test_data = data[indexs, :]
t_gt = test_data[:, 1].reshape(10, 1)
t_pr = (test_data[:, 0] * m + c).reshape(10, 1)


MAPE = np.mean(np.abs((t_gt - t_pr) / t_gt)) * 100
precision = 100 - MAPE
print(f'MAPE: {MAPE:.2f}%')
print(f'precision: {precision:.2f}%')

## display an assist digram
visualize(m, c, X_raw, Y_raw, 0, 'green', 'Sklearn line')
visualize(_m, _c, X_raw, Y_raw, 1, 'red', 'GradientDescent line')





# mileage = input("enter the mileage: ")
# price = float(mileage) * m + c
# print("price = ", f"{price[0]:,}")
