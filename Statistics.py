#-----------------------------------------
#Playing with statistics in dataframes and testing out predictions with Linear Regression
#v.1 2020/2/7
#Armani Chien
#------------------------------------------
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.linear_model import LinearRegression



def plot():
    df.plot(kind='scatter', x='nx', y='ny', title='Statistics')
    plt.show()

def descriptives():
    print(df)
    print("sum")
    print("sum-" + str(df.sum()))
    print("min-" + str(df.min()))
    print("max-" + str(df.max()))
    print("mean-" + str(df.mean()))
    print("mode-" + str(df.mode()))
    print("median-" + str(df.median()))
    print("range-" + str(df.max() - df.min()))
    print(len(df))

def linearRegressionModel():
    # Converting df to an numpy array
    X = df.iloc[:, 0].values.reshape(-1, 1)
    Y = df.iloc[:, 1].values.reshape(-1, 1)
    linearRegressor = LinearRegression()
    linearRegressor.fit(X, Y)
    Y_pred = linearRegressor.predict(X)  # Make Prediction
    plt.scatter(X, Y)  # Drawing scatter plot
    plt.plot(X, Y_pred, color='red')  # Placing red line prediction on plot
    plt.title("Linear Regression with Predicted y")
    plt.show()

    r_squared = linearRegressor.score(X, Y)  # Obtain R Squared
    print('Coeff of determination:', r_squared)
    print('intercept:', linearRegressor.intercept_)
    print('slope:', linearRegressor.coef_)
    print('correlation:', df.corr(method='pearson'))

#Creating a dictionary for the linear Regression plot
d = {'nx': [5, 15, 25, 35, 45, 55], 'ny': [5, 20, 14, 32, 22, 38]}
df = pd.DataFrame(data=d)

linearRegressionModel()
