import pandas as pd
import numpy as np
from visualize import *
from gradient_descent import grad_descent, f_derive
from sklearn.preprocessing import MinMaxScaler



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

# print(X_scld)
# exit()
# print("\n")
# print(Y_scld)
# visualize_points(X, Y)\

X_min = np.min(f2)
X_max = np.max(f2)
Y_min = np.min(Y_raw)
Y_max = np.max(Y_raw)
def unscld_wights(w_scld):
    wights = w_scld
    wights[1] = wights[1] * ((Y_max-Y_min) / (X_max-X_min))
    wights[0] = wights[0] * (Y_max-Y_min) + Y_min - (wights[1]*Y_min)
    return wights



wights_scld = grad_descent(X_scld, Y_scld,
        np.array([[50000], [20102]]), lr=1, pr=0.000000001)

wights = unscld_wights(wights_scld)
m = wights[1]
c = wights[0]
print("m = ", f'{m[0]:,}')
print('c = ', f'{c[0]:,}')

# visualize(m, c, X_raw, Y_raw)


mileage = input("enter the mileage: ")
price = float(mileage) * m + c
print("price = ", f"{price[0]:,}")
print("price_raw = ", price)
