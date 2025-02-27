import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display_points(X, Y, xlabel='X', ylabel='Y'):
    print("----display_points----")
    for x, y in zip(X, Y):
        plt.scatter(x, y, color='red')
    plt.plot(X, Y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



def costs_VS_iters(costs, iters):
    plt.figure(figsize=(10, 5))
    xItr = np.arange(iters+1)
    plt.plot(xItr, costs, marker='o', linestyle='-', color='b', label="House Size")
    plt.xlabel("House ID")
    plt.ylabel("Size (sq ft)")
    plt.title("House Sizes by ID")
    plt.legend()
    plt.grid(True)
    plt.show()

def featurs_Vs_target(df1,df2,df3):
    # Create figure of three plots
    fig, axis = plt.subplots(1, 3,figsize=(15,5))

    # Feat1
    df1.sort_values(by="Feat1", inplace=True)
    axis[0].set_title("Feat1 VS Target")
    axis[0].set_xlabel("Feat1")
    axis[0].set_ylabel("Target")
    axis[0].scatter(df1['Feat1'],df1['Target'], color="g")

    # Feat2
    df2.sort_values(by="Feat2", inplace=True)
    axis[1].scatter(df2['Feat2'],df2['Target'],linestyle='-', color="r")
    axis[1].set_title("Feat2 VS Target")
    axis[1].set_xlabel("Feat2")
    axis[1].set_ylabel("Target")

    # Feat3
    df3.sort_values(by="Feat3", inplace=True)
    axis[2].scatter(df3['Feat3'],df3['Target'],linestyle='-', color="b")
    axis[2].set_title("Feat3 VS Target")
    axis[2].set_xlabel("Feat3")
    axis[2].set_ylabel("Target")
    plt.tight_layout()
    plt.show()