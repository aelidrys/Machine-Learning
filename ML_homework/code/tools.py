import numpy as np
import matplotlib.pyplot as plt


def unscld_wights(w_scld, scaler_X, scaler_Y):
    X_min = scaler_X.data_min_
    X_max = scaler_X.data_max_
    Y_min = scaler_Y.data_min_
    Y_max = scaler_Y.data_max_
    wights = w_scld
    wights[1] = wights[1] * ((Y_max-Y_min) / (X_max-X_min))
    wights[0] = wights[0] * (Y_max-Y_min) + Y_min - (wights[1]*Y_min)
    return wights