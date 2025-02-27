import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


save_path = 'scatter_plot.png'


def visualize(X, Y, Y_pr, color='red', label='line'):

    plt.figure(figsize=(10,6))
    
    plt.plot(X,Y_pr, label=label,color=color)
    plt.scatter(X,Y)
    plt.title('MSE')
    plt.xlabel("mileage")
    plt.ylabel("price")
    plt.legend()
    # plt.savefig(save_path, format='png', dpi=300)
    plt.show()


