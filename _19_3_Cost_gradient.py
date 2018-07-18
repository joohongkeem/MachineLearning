# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180718", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")


# 경사하강법이라는 cost 함수를 통해, early stop을 구현하자!
#
#  * 경사 하강법(Gradient Descent : GD) ★
#  - 최적의 매개변수를 찾기 위해 손실 함수를 정의하고, 손실 함수가 최소값이 될 때의 매개변수를 찾는 방법
#    >> 신경망, 딥러닝 등에 가장 많이 사용
#

import matplotlib.pyplot as plt

# 평균제곱 오차를 구한다.
def cost(x, y, w) :
    c = 0

    # x에 포함된 데이터 만큼 평균 제곱 오차를 구한다.
    for i in range(len(x)) :
        hx = w * x[i]
        c += (hx - y[i])**2
    return c/len(x)


# X 범위(-2.0 ~ 5.0)에 대한 Cost 함수를 작성한다.
def cost_graph(X, Y) :
#    X = [1, 2, 3]
#    y = [1, 2, 3]
    for i in range(-20, 50) :
        w = i /10
        c_result = cost(X, y, w)
        print(w, c_result)
        plt.plot(w, c_result, 'ro')
    plt.show()


# 경사하강법 (배치 경사 하강법)
def gradient_descent(x, y, w) :
    delta = 0
    #
    for i in range(len(x)) :
        # Hypothesis
        hx = w * x[i]
        delta += (hx - y[i]) * x[i]
        print("delta :", delta , " hx :", hx, "y[",i,"] :", y[i], "x[",i,"] :", x[i])
    return delta/len(x)


def preidct(x, w) :
    for i in range(len(x)) :
        result = x[i] * w
        print('input ',x[i], 'predict :', result)


X = [1, 2, 3]
y = [1, 2, 3]
#X = [1, 2, 3, 4, 5]
#y = [1, 4, 6, 8, 10]


w = 10
a = 1   # 1일때 발산, 0.1로 테스트
          # a : 학습률, 값을 바꿔보자 (0~1)★★★
          # a = 1 로하고 아래 반복문을 range(10000)으로 하면 에러 - Overflow !!
          # a = 0.01로 하고 아래 반복문을 range(10000)으로 하면 괜찮다

for i in range(10000) :
    # Early Stop을 위하여 cost를 계산한다.
    c = cost(X, y, w)

    # Gredient Descent를 통해 가중치의 변위(Delta)를 구한다.
    g = gradient_descent(X, y, w)

    # 가중치 = 가중치 + 학습률 * 변위(Delta)     1 step w :   = 10 - 0.1 * 4.2 --> 5.8
    w -= a * g
    print(" w :", w)
    # early Stopping.... 중간에 탈출할 수 있는 방법임...(다양한 방법으로 적용필요)
    if c < 1.0e-25 :
  #      print("HERE!!")
        break
    print('Loop Count : [',i,']', 'Cost: [' , c, ']')

x =[6, 7]
preidct(x, w)

# Cost 함수를 시각화 해본다.
cost_graph(X, y)
