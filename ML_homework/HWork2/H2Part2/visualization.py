import matplotlib.pyplot as plt
import numpy as np

def error_VS_degree(train_error, test_error, degree):
    plt.figure(figsize=(10, 5))
    xDegrees = np.arange(1,degree+1)
    plt.plot(xDegrees, train_error, linestyle='-', color='orange', label="Train Error")
    plt.plot(xDegrees, test_error, linestyle='-', color='b', label="Test Error")
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.title("Error VS Degree")
    plt.xticks(np.arange(1, degree+1))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
def FeaurVsError(FsErr):
    
    barWidth = 0.15
    indexs = np.array([0,1,2,3,4,5,6,7,8])
    br2 = [x + barWidth for x in indexs] 
    plt.subplots(figsize=(10, 5))
    plt.bar(indexs, FsErr[:,0], width=barWidth, color='green', label="Train")
    plt.bar(br2, FsErr[:,1], width=barWidth, color='b', label="Test")
    plt.xlabel("Featue IDs")
    plt.ylabel("RMSE")
    plt.title("Error VS Degree")
    plt.xticks([r for r in range(9)],
               ['0','1','2','3','4','5','6','7','8'])

    plt.legend()
    plt.grid(True)
    plt.show()



def alphaVSerror(alphas, errors):
    plt.figure(figsize=(10,5))
    plt.plot(alphas, errors, linestyle="-", label="test_error", color="b")
    plt.xlabel("aplhas")
    plt.ylabel("errors")
    plt.grid(True)
    plt.tight_layout()
    plt.title("alphas Vs errors")
    plt.show()
