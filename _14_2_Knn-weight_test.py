# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180716","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")


# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)
X_train = np.sort(5 * np.random.rand(40, 1), axis=0)
x_test = np.linspace(0, 5, 500)[:, np.newaxis]
y_train = np.sin(X_train).ravel()

# Add noise to targets
y_train[::5] += 1 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model  
# 거리에 따른 가중치를 어떻게 줄것인가?
# uniform : 동일하게 , distance : 거리에 따른 가중치 조절
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    predict = knn.fit(X_train, y_train).predict(x_test)


    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train, y_train, c='k', label='data')
    plt.plot(x_test, predict, c='g', label = 'Prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNN (k = %i, weights = '%s')"%(n_neighbors,weights))
plt.show()