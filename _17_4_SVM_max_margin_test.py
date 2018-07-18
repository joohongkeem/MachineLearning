# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180717", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")


import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs

# we create 40 separable points
X_train, y_train = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X_train,y_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)

#print('XX :', XX.shape)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
#print('xy :' , xy, 'xy.shape ', xy.shape, 'XX.shape', XX.shape)
Z=clf.decision_function(xy).reshape(XX.shape)
#print('Z shape :', Z.shape)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
          linewidth=3, facecolors='red')
plt.show()