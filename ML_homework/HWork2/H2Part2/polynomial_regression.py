import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from visualization import error_VS_degree

def monomials_poly_features(X, degree):
    i = 2
    X1 = X
    while i <= degree:
        X = np.hstack([X,X1**i])
        i +=1
    return X



def do_polynomailReg(X,Y,degree, poly=False):
    TrainError = []
    TestError = []
    for i in range(degree):
        if poly:
            X_new = PolynomialFeatures(i+1).fit_transform(X)
        else:
            X_new = monomials_poly_features(X,i+1)
        
        # Manual split 50% - 50%
        X_train = X_new[:100]
        X_test = X_new[100:]
        Y_train = Y[:100]
        Y_test = Y[100:]

        # Scaling
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train
        LReg = LinearRegression(fit_intercept=True).fit(X_train,Y_train)

        print("########## Degree: ",i+1, "##########")
        print("intercept: ",LReg.intercept_)
        Predict = LReg.predict(X_train)
        trainRMSE = np.sqrt(mean_squared_error(Predict, Y_train))
        print("Train RMSE: ", trainRMSE)
        TrainError.append(trainRMSE)
        
        Predict = LReg.predict(X_test)
        testRMSE = np.sqrt(mean_squared_error(Predict, Y_test))
        print("Test RMSE: ", testRMSE)
        TestError.append(testRMSE)
        print("####################")
        
    return TrainError, TestError


