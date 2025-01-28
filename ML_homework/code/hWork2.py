import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_name = '../data_sets/hWork2.csv'
df = pd.read_csv(file_name)
# print(df.info())


X = df.drop(columns=['Target']).to_numpy()
T = df['Target'].to_numpy().reshape(-1,1)

X_train, X_val, t_train, t_val = train_test_split(X, T, test_size=0.5, shuffle=False)

# print('X_train before scaling\n', X_train)
X_TrainScaled = MinMaxScaler().fit_transform(X_train)
X_TrainScaled = np.hstack([np.ones((100, 1)), X_TrainScaled])
# print('X_train after scaling\n', X_TrainScaled)

model = LinearRegression(fit_intercept=True)
reg = model.fit(X_TrainScaled, t_train)
# print('wights: ', reg.coef_)

X_ValScaled = MinMaxScaler().fit_transform(X_val)
X_ValScaled = np.hstack([np.ones((100, 1)), X_ValScaled])

t_predict = reg.predict(X_ValScaled)
# print('Prediction: \n\n', t_predict)

MAPE = np.mean(np.abs((t_predict - t_val) / t_val)) * 100
precision = 100 - MAPE
print(f'precision: {precision:.2f}%')
