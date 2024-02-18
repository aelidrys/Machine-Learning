import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def scaler(v_arr):
    min_v = np.min(v_arr)
    max_v = np.max(v_arr)
    for i, X in enumerate(v_arr):
        v_arr[i] = (X - min_v) / (max_v - min_v)
        print((X - min_v) / (max_v - min_v))
    return v_arr


def my_minmax_sc(data):
    max_v = np.max(data)
    min_v = np.min(data)
    n_clmn = data.shape[1]
    v_arrys = [data[:,i] for i in range(n_clmn)]
    for i, v_arr in enumerate(v_arrys):
        v_arr = scaler(v_arr)
        print(v_arr)
    # sc_data = (data - min_v) / (max_v - min_v)
    return np.array(v_arrys)

process = MinMaxScaler()
data = np.array([[1,20,3],
                 [4,15,6],
                 [7,8,90],
                 [9,11,2]])

my_sc_data = my_minmax_sc(data)

print("\ndata befor sklearn scaling\n" , data)
data_sc = process.fit_transform(data)
print("\ndata after sklearn scaling\n" , data_sc)
print("\ndata after my scaling\n" , my_sc_data)
