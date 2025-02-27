import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split

# Scaler
df = pd.read_csv("data.csv")
X = np.array(df["km"]).reshape(-1,1)
Y = np.array(df["price"]).reshape(-1,1)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=88)


# Scaling
scaler_X = MinMaxScaler().fit(X_train)


# Wights
w_df = pd.read_csv("Wights.csv")
wights = np.array(w_df['wights']).reshape(-1,1)


# Predict the Price
mileage = input("enter the mileage: ")
mlg_Features = np.array([int(mileage)]).reshape(-1,1)
mlg_Features = scaler_X.transform(mlg_Features)
mlg_Features = np.hstack([np.ones((1,1)),mlg_Features])

# print(f"-----wights--------\n{wights}")
# print(f"-----mlg_Features--------\n{mlg_Features}")
price = np.dot(mlg_Features,wights)[0][0]
print("price = ", f"{price:.5f}")

