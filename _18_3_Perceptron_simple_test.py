# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180717", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

# Perceptron의 classification 확장 !

import numpy as np
import matplotlib.pyplot as plt
# import sklearn.linear_model.perceptron as p
from sklearn.linear_model import Perceptron

# Needed to show the plots inline
# %matplotlib inline
# Data

colormap = np.array(['r', 'k'])


def make_cal(X_train, y_train):
    # Create the model
    # 아래 두개의 모델을 비교해보자!
    #net = Perceptron(max_iter=100, random_state=None, eta0=0.002, fit_intercept=True, verbose=0)
    net = Perceptron(max_iter=100, random_state=True, eta0=0.002)
    net.fit(X_train, y_train)

    # Print the results
    print("Actual      :", str(y_train))
    print("Prediction  :", net.predict(X_train))
    print("Accuracy    :", net.score(X_train,y_train))

    # Plot the original data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colormap[y_train], s=40)

    # Output the values
    print("Coefficient 0 :", net.coef_[0,0])
    print("Coefficient 1 :", net.coef_[0,1])
    print("Bias          :", net.intercept_)                # 절편 = bias


    # Calc the hyperplane (decision boundary)
    plt.ylim([0, 10])
    ymin, ymax = plt.ylim()

    print('ymin :', ymin, 'ymax :', ymax)

    w1 = net.coef_[0]
    a1 = -w1[1] / w1[0]

    xx1 = np.linspace(ymin, ymax)
    yy1 = a1*xx1 - net.intercept_[0]/w1[0]
    print(xx1)
    print(yy1)
    # Plot the line
    plt.plot(yy1, xx1, 'k-')
    plt.show()





X_train = np.array([[2, 2], [1, 3], [2, 3], [5, 3], [7, 3], [2, 4],
                    [3, 4], [6, 4], [1, 5], [2, 5], [5, 5], [4, 6], [6, 6], [5, 9]])

# Labels
y_train = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])

make_cal(X_train, y_train)

