import pandas as pd
import numpy as np
from visualize import *
from gradient_descent import grad_descent, f_derive, cost_f
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


file_name = "/nfs/homes/aelidrys/Desktop/ml/LinearRegression/linear_regr_42/data.csv"
data_csv = pd.read_csv(file_name)
data_csv = data_csv.sort_values(by="km")

X_raw = np.empty((24, 1))
X_raw = np.array(data_csv["km"])
label = np.array(data_csv["price"])
Y_raw = label.reshape(24,1)


ones = np.ones((24, 1))
f2 = X_raw.reshape(24, 1)
f2_scld = MinMaxScaler().fit_transform(f2)
X_scld = np.hstack([ones,f2_scld])
scaler_Y = MinMaxScaler()
Y_scld = scaler_Y.fit_transform(Y_raw)

# unscaled the wights
X_min = np.min(f2)
X_max = np.max(f2)
Y_min = np.min(Y_raw)
Y_max = np.max(Y_raw)
def unscld_wights(w_scld):
    wights = w_scld
    wights[1] = wights[1] * ((Y_max-Y_min) / (X_max-X_min))
    wights[0] = wights[0] * (Y_max-Y_min) + Y_min - (wights[1]*Y_min)
    return wights

# generate random wights between 0 and 1
init_wights = np.random.rand(2,1)

# launch gradient descent the learn the optimal wights
wights_scld = grad_descent(X_scld, Y_scld,
        init_wights, lr=1, pr=0.000000001)
wights = unscld_wights(wights_scld)
m = wights[1]
c = wights[0]

# print("m = ", f'{m[0]:,}')
# print('c = ', f'{c[0]:,}')

## display an assist digram
# visualize(m, c, X_raw, Y_raw)

# sklearn Model
reg = LinearRegression().fit(X_scld, Y_scld)
print(reg.coef_)
wights = unscld_wights(reg.coef_[0])
m = wights[1]
c = wights[0]



#### testing #####
km = np.array(data_csv['km']).reshape(24, 1)
price = np.array(data_csv['price']).reshape(24, 1)
data = np.hstack([km, price])
indexs = np.random.choice(data.shape[0], 10, replace=False)
test_data = data[indexs, :]
t_gt = test_data[:, 1]
t_gt = t_gt.reshape(10, 1)
t_pr = test_data[:, 0] * m + c
t_pr = t_pr.reshape(10, 1)

MAPE = np.mean(np.abs((t_gt - t_pr) / t_gt)) * 100
precision = 100 - MAPE
print('MAPE: ', MAPE)
print('precision: ', precision)
print('\n')






# mileage = input("enter the mileage: ")
# price = float(mileage) * m + c
# print("price = ", f"{price[0]:,}")
