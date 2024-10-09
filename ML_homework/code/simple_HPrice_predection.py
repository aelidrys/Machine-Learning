import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tools import unscld_wights

train_file_name = '../data_sets/simple_HPTrain.csv'
test_file_name = '../data_sets/simple_HPTest.csv'
target_file_name = '../data_sets/HPrice_target.csv'
df = pd.read_csv(train_file_name)
df_test = pd.read_csv(test_file_name)
df_target = pd.read_csv(target_file_name)

########### Train ###########
ones = np.ones((1460, 1))
X_train = np.array(df[['LotArea', 'YearBuilt']])
X_scaler = MinMaxScaler().fit(X_train)
X_train_scled = X_scaler.transform(X_train)
X_train_scled = np.hstack([ones, X_train_scled])
Y_train = np.array(df['SalePrice']).reshape(1460, 1)
reg = LinearRegression().fit(X_train_scled, Y_train)


########### Test ###########
ones = np.ones((1459, 1))
X_test = np.array(df_test[['LotArea', 'YearBuilt']]).reshape(1459, 2)
X_test_scled = MinMaxScaler().fit_transform(X_test)
X_test_scled = np.hstack([ones, X_test_scled])
target = np.array(df_target[['SalePrice']])
Pr_target = reg.predict(X_test_scled)
Error = (Pr_target - target)
cost = Error.T.dot(Error) / target.shape[0]
print('prediction: ', Pr_target) 
print('ERROR: ', cost) 





