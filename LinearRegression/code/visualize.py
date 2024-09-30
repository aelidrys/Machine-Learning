import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
save_path = os.path.join(current_directory,
    'LinearRegression/scatter_plot.png')
def visualize(m, c, X, Y):
    line_y = []
    X1 = X
    for x in X1:
        line_y.append(m * x + c)
    plt.scatter(X1,Y)
    plt.plot(X1,line_y, label='prediction line',color='red')
    plt.title('MSE')
    plt.xlabel("mileage")
    plt.ylabel("price")
    plt.legend()
    # plt.savefig(save_path, format='png', dpi=300)
    plt.show()









