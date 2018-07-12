# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180712","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import numpy as np

# - 코드 분석해보기
# - Linear 분석 그래프 그려보기
#


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sckit-learn 제공하는 셈풀 데이터 생성 함수
X_train, y_train = make_classification(n_features=1, n_redundant=0, n_informative=1,
                            n_clusters_per_class=1, random_state=4)


#print(X_train)

model = LogisticRegression().fit(X_train, y_train)

predicts = model.predict(X_train)
#계수 / 절편
print(model.coef_)
print(model.intercept_)



xx = np.linspace(-3, 3, 100)
# 시그모이드 함수에 적용..
sigm = sigmoid(model.coef_[0][0]*xx + model.intercept_[0])

precision, recall, thresholds = roc_curve(y_train, model.predict(X_train))

plt.plot(xx, sigm, label='sigmoid', c='red')
plt.scatter(X_train, y_train, marker='o', label='Training Data', s=100)

plt.scatter(X_train, predicts, marker='x', label='Predict Data', c=y_train, s=200, lw=2, alpha=0.5)
plt.xlim(-3, 3)
plt.legend()

plt.show()

plt.plot(precision, recall)
plt.show()