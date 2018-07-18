# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180717", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")



import numpy as np
import matplotlib.pyplot as plt

def and_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.7
    # 연산의 종류마다 theta의 값이 다르다

    # Hypothesis 만들기
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        if tmp <= theta:
            print(x1[0], 'AND', x1[1], '=', 0)
        elif tmp > theta:
            print(x1[0], 'AND', x1[1], '=', 1)

def or_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.2
    # 연산의 종류마다 theta의 값이 다르다

    # Hypothesis 만들기
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        if tmp <= theta:
            print(x1[0], 'OR', x1[1], '=', 0)
        elif tmp > theta:
            print(x1[0], 'OR', x1[1], '=', 1)

def nand_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.5
    # 연산의 종류마다 theta의 값이 다르다

    # Hypothesis 만들기
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        if tmp > theta:
            print(x1[0], 'NAND', x1[1], '=', 0)
        elif tmp <= theta:
            print(x1[0], 'NAND', x1[1], '=', 1)


X_train = np.array([[0,0],[1,0],[0,1],[1,1]])

and_cal(X_train)
or_cal(X_train)
nand_cal(X_train)