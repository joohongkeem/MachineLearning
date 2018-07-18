# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180718", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

# - Perceptron 클래스를 직접 만들어 Iris 데이터 예측에 적용한다.
# - Perceptron 클래스는 벡터의 내적으로 작성한다.
# - 2개의 특징 데이터를 이용하여 학습을 수행한다.


# perceptron.py
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

# Perceptron 모델을 직접 만들어 사용해 본다.
class Perceptron(object):
    # Perceptron 클래스의 생성자를 정의한다.
    # 생성자의 기본 값으로 rate = 0.01, epoch = 10
    def __init__(self, rate=0.01, epoch=10):
        self.rate = rate
        self.epoch = epoch

    def fit(self, X, y):
        # Fit training data
        # X : Training vectors, X.shape : [#samples, #features]
        # y : Target values, y.shape : [#samples]

        # 가중치를 numpy 배열로 정의한다.
        # X.shape[1]은 트레이닝 데이터의 입력값 개수를 의미한다.
        # 예를 들어 X가 4x2 배열인 경우, X.shpae의 값은 (4,2) 가되며
        # X.shape[1]의 값은 2가 된다.
        # 이 경우, self.w_는 np.zeors(3)이 될 것이고, 실제 값은 numpy 배열 [0. 0. 0.]이다.
        self.weight = np.zeros(1 + X.shape[1])          # weights   초기 [ 0. 0. 0.]   2개의 Feature, 1개의 상수.
        # 머신러닝 반복 회수에 따라 퍼셉트론의 예측값과 실제 결과값이 다른 오류 회수를 저장하기 위한 변수
        self.errors = []                                # Number of misclassifications

        print("X.shape :", X.shape)
        print("weight.shape :", self.weight.shape)
        print("weight :", self.weight)

        # self.epoch로 지정한 숫자만큼 반복한다.
        for i in range(self.epoch):
            # 초기 오류 횟수를 0으로 정의한다.
            err = 0

            # 트레이닝 데이터 세트 X와 결과값 y를 하나씩 꺼내서
            # xi, yarget 변수에 대입한다.
            # xi는 하나의 트레이닝 데이터의 모든 입력값 x1~xn을 의미한다.
            # 참고로 x0은 1로 정해져있다.

            # 데이터의 개수를 카운트해보자
            mycount = 0
            for xi, target in zip(X, y):
                mycount+=1
                print('xi.shape :', xi.shape)
                print('xi : ', xi)
                # Perceptron의 델타 규칙을 구현한다.
                # 델타 ->  퍼셉트론의 학습 : Wi = Wi + aXi(t-y) 에서 a(t-y) !!
                #       (a : 학습률, Xi : 입력노드 i에서의 입력 값, t : 목표값, y : 학습 출력값)
                # 델타는 w의 증분!
                #
                # 실제 결과값과 예측값에 대한 활성 함수 리턴값이 같게 되면 delta_w 는 0 이된다.
                # 따라서 트레이닝 데이터 xi의 값에 곱해지는 가중치에 delta_w * xi 값을 단순히 더함으로써 가중치를 업데이트 할 수 있는데,
                # 이는 결과값과 예측값의 활성 함수 리턴 값이 같아질 경우 0을 더하는 꼴이라 가중치의 변화가 없을 것이고
                # 결과값과 예측값의 활성 함수 리턴 값이 다를 경우 0이 아닌 유효한 값이 더해져서 가중치가 업데이트 될 것이다.
                # 마찬가지로 w0의 값에 해당하는 self.weight[0] 에는 x0이 1이므로 delta_w만 단순히 더하면 된다.
                delta_w = self.rate * (target - self.predict(xi))
                # print(delta_w)
                self.weight[1:] += delta_w * xi                 # Xi를 고려한다
                # print(self.weight[1:])
                self.weight[0] += delta_w                       # Xi를 고려하지 않는다.
                # print(self.weight[0])
                # delta_w 의 값이 0 이 아닌경우, err의 값을 1 증가시키고 다음 트레이닝 데이터로 넘어간다.
                err += int(delta_w != 0.0)          # target과 predict값이 다르면 에러에 추가한다 (같으면 패스)
            # 모든 트레이닝 데이터에 대해 1회 학습이 끝나면 self.errors에 발생한 오류 횟수를 추가한 후, 다음 학습을 반복한다.
            print('## {0}번째 학습 완료 - 학습 데이터 갯수 = {1} , 불일치 값의 갯수 = {2}'.format(i+1,mycount,err))
            self.errors.append(err)
        print(self.errors)
        return self

    # predict 함수 사용 시 내적 계산을 수행 함.
    # numpy.dot(x,y)는 벡터 x,y의 내적 또는 행렬 x,y의 곱을 리턴한다.
    # 따라서 net_input(self,X)는 트레이닝 데이터 X의 각 입력값과 그에 다른 가중치를 곱한 총 합,
    # 즉, summation(WiXi) 의 결과값을 구현한 것이다.
    def net_input(self, X):
        """Calculate net input"""
#        print('X : ' ,  X, ' self.weight[1:] :', self.weight[1:], ' self.weight[0] :',  self.weight[0])
        return np.dot(X, self.weight[1:]) + self.weight[0]     # Xi와 Wi를 내적해서 모두 더하고 , bias(절편)을 더한다.

    # net_input 함수를 통해 내적된 결과를 1과 -1로 표현한다
    def predict(self, X):
        """Return class label after unit step"""

        return np.where(self.net_input(X) >= 0.0, 1, -1)        # 결과값이 threshold(0.0) 이상이면 1 아니면 -1
                                                                 # -> 활성함수의 역할!

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# 0과 2 인덱스 값 사용  (100 X 2 )
X_train = df.iloc[0:100, [0, 2]].values
print(X_train)
# 목표 값 설정 (100 X 1)
y_train = df.iloc[0:100, 4].values
print(y_train)
y_train = np.where(y_train == 'Iris-setosa', -1, 1)
print(y_train)

plt.figure()
plt.scatter(X_train[:50, 0], X_train[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X_train[50:100, 0], X_train[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
#plt.show()

plt.figure()
pn = Perceptron(0.1, 10)
pn.fit(X_train, y_train)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
#plt.show()


plt.figure()
plot_decision_regions(X_train, y_train, classifier=pn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()