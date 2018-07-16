# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:18:37 2017

@author: BEAST
"""

import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import style
#style.use("ggplot")
from sklearn import svm

def print_line(model, X) :
    w = model.coef_[0]
    print('weight :', w)
    a = -w[0] / w[1]
    b = model.intercept_[0]/w[1]
    xx = np.linspace(1,9)
    yy = a* xx - b
    plt.plot(xx, yy, 'k-', label ='non weighted div')
    plt.scatter(X[:, 0], X[:, 1], c ='g', label ="input data")
    plt.legend()


X_train = np.array([[1,2],[6,7],[1.4,1.9],[9,8],[1.1,0.7],[9,11]])
y_train = [0,1,0,1,0,1]

#plt.show()
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1])

svm_model = svm.SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)


X_test = np.array([[5,4.1],[6,2.0]])
print(X_test)

plt.figure()
plt.scatter(X_test[0,0],X_test[0,1],c ='r', label='test data label : {0}'.format(svm_model.predict(X_test[0].reshape(1,-1))[0]) )
plt.scatter(X_test[1,0],X_test[1,1],c ='b', label='test data label : {0}'.format(svm_model.predict(X_test[1].reshape(1,-1))[0]))
print_line(svm_model, X_train)
plt.show()