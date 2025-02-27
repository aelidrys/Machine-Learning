from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


file_name = "data.csv"
data_csv = pd.read_csv(file_name)
X = np.array(data_csv["km"]).reshape(-1,1)
Y = np.array(data_csv["price"]).reshape(-1,1)

# git Statistics
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=88)
scaler_X = MinMaxScaler().fit(X_train)

# Choice Testing Set
X_test = X
Y_test = Y


X_test = scaler_X.transform(X_test)
X_test = np.hstack([np.ones((X_test.shape[0],1)),X_test])


w_df = pd.read_csv("Wights.csv")
wights = np.array(w_df['wights']).reshape(-1,1)

 
# Train Performence
Y_pr = np.dot(X_test,wights)
u = np.sum((Y_test - Y_pr) ** 2)
v = np.sum((Y_test - np.mean(Y_test)) ** 2)
precision = (1 - u / v) * 100
print(f"----Precision----------------")
print(f"\tprecision: {precision:.2f}%")