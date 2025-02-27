from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from visualization import alphaVSerror
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

def do_ridgeWithPoly(X,Y,alpha=None, degree_=2):
    ridge_pipe = Pipeline([   
        ("scale", MinMaxScaler()),
        ("poly",PolynomialFeatures(degree=degree_)),
        ("ridge", Ridge()),
    ])
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    ridge_prms = {"ridge__alpha": alphas}
    GS = GridSearchCV(ridge_pipe, ridge_prms, scoring="neg_mean_squared_error", cv=4)
    GS.fit(X,Y)
    print("best_alpha", GS.best_params_['ridge__alpha'])
    print("best_score: ", GS.best_estimator_.score(X, Y))
    errors = np.sqrt(-GS.cv_results_["mean_test_score"])
    print("alphas: ", alphas)
    print("error", errors)
    alphaVSerror(np.log10(alphas), errors)
    
    
def do_lassoWithPoly(X,Y,alpha=None,data=None):
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10]
    
    # Manual split 50% - 50%
    X_train = X[:100]
    X_test = X[100:]
    Y_train = Y[:100]
    Y_test = Y[100:]

    # Scaling
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    models = []
    for alpha in alphas:
        model = Lasso(fit_intercept = True, alpha=alpha, max_iter=10000)
        model.fit(X_train,Y_train)
        models.append(model)   

    predict = models[0].predict(X_test)
    rmse = np.sqrt(mean_squared_error(predict, Y_test))
    best_model = None
    for model in models:
        predict = model.predict(X_test)
        testRMSE = np.sqrt(mean_squared_error(predict, Y_test))
        if rmse > testRMSE:
            best_model = model
            rmse = testRMSE
        # print("predict: ", testRMSE)
        
    predict = best_model.predict(X_test)
    testRMSE = np.sqrt(mean_squared_error(predict, Y_test))
    print("best_error: ", testRMSE)
    
    return best_model
    