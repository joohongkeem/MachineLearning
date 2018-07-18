# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180717", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")


# 참고
#  DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
#  -> 0.21 버전부터는 사용되지 않을 것이므로 max_iter 을 사용해라!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Needed to show the plots inline
# %matplotlib inline
# Data

def make_cal(X_train, y_train, lb):

    print('----------------------'+lb+'----------------------')

    # Create the model
    net = Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
                            # 각 항이 무엇을 의미하는지 검색해보자!! ★★
                            # fit_intercept : 상수항을 사용할지!
    # 학습
    net.fit(X_train, y_train)

    # Print the results
    print("Prediction :", str(net.predict(X_train)))
    print("Actual     :", str(y_train))
    print("Accuracy   :", str(net.score(X_train, y_train)))



    plt.figure()
    # Plot the original data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colormap[y_train], s=40)
    plt.title("Perceptron [{0}] calculation  : ".format(lb))

    print("Coefficient :",net.coef_)

    # Output the values
    print("Coefficient 0 :", net.coef_[0,0])
    print("Coefficient 1 :", net.coef_[0,1])
    print("Bias          :", net.intercept_)


    # Calc the hyperplane (decision boundary)
    ymin, ymax = plt.ylim()
    w = net.coef_[0]
    print("w[0] =",w[0])
    print("w[1] =",w[1])
    a = -w[1] / w[0]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (net.intercept_[0]) / w[0]






    # Plot the line
    plt.plot(yy, xx, 'k-')
    #plt.show()

colormap = np.array(['r', 'k'])

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Labels
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_nand = np.array([1, 1, 1, 0])
y_xor = np.array([0, 1, 1, 0])


make_cal(X, y_and, 'and')
make_cal(X, y_or, 'or')                 # 실행할 때마다 직선이 조금씩 다르게 나온다!
make_cal(X, y_nand, 'nand')
make_cal(X, y_xor, 'xor')             # xor은 하나의 직선으로 구분할 수 없기 때문에 오류발생!!!

plt.show()
