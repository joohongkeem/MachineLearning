# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180716","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

    # 확인해볼점 !!
    # Q. 과연 depth가 큰게 무조건 좋을까?
    #

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import pandas as pd
import pydot


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
# 1, 2, 3, 4, 5로 변경하여 대
max_depths = 1

# 자동으로 데이터셋을 분리해주는 함수
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state= 0)

#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

while max_depths <= 5 :
    # criterion='entropy' : information Gain을 사용하겠다는 의미,
    tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depths, random_state=0)

    # 훈련
    tree_model.fit(X_train, y_train)

    # 예측
    y_pred_tr = tree_model.predict(X_test)

    # 검증
    accuracy = accuracy_score(y_test, y_pred_tr)
    print('Accuracy: %.2f' % accuracy)

    plt.figure()
    print("max_depths :", max_depths)

    dot_data = tree.export_graphviz(tree_model, out_file=None, feature_names=[iris.feature_names[2], iris.feature_names[3]],
                                    class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('./_16_1_Tree[{0}].png'.format(max_depths))

    #Image(graph.create_png())


# 2차원 그래프 그리기 (150개 전체 데이터 기준 )
    resolution = 0.01
    markers = ('s', '^', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = tree_model.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title("Decision Tree Depth [{0}] \n Accuracy Score : {1}".format(max_depths, accuracy))
    max_depths += 1
    #plt.show()

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], s=80, label=cl)

plt.show()