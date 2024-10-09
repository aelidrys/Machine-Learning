import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
save_path = os.path.join(current_directory,
    'LinearRegression/scatter_plot.png')
def visualize(m, c, X, Y, dp=1, color='red', label='line'):
    line_y = []
    X1 = X
    for x in X1:
        line_y.append(m * x + c)
    plt.plot(X1,line_y, label=label,color=color)
    if dp:
        plt.scatter(X1,Y)
        plt.title('MSE')
        plt.xlabel("mileage")
        plt.ylabel("price")
        plt.legend()
        # plt.savefig(save_path, format='png', dpi=300)
        plt.show()









