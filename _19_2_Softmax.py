# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180718", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")


import numpy as np


def softmax(a):
    print()
    print('a =',a)
    c = np.max(a)
    print('c =', c)
    exp_a = np.exp(a-c)         # overflow 방지
    print('exp_a =',exp_a)

    sum_exp_a = np.sum(exp_a)
    print('sum_exp_a =',sum_exp_a)
    y = exp_a / sum_exp_a
    print('y =',y)
    return y

# e100 은 0 dl 40 개가 넘는 큰값
def softmax1(a):
    print()
    print('a =', a)
    c = np.max(a)
    print('c =', c)
    exp_a = np.exp(a)  # overflow 방지
    print('exp_a =', exp_a)

    sum_exp_a = np.sum(exp_a)
    print('sum_exp_a =', sum_exp_a)
    y = exp_a / sum_exp_a
    print('y =', y)
    return y

a = np.array([0.3, 2.9, 4.0])
#a= np.array([1010, 1000, 990])
y= softmax1(a)
#y = softmax(a)


# Softmax 결과의미
# 1) 확률
# 2) 각 원소간의 관계 변하지 않음 : exp 함수가 단조 증가함수이기 때문에


print( "Softmax Result :", y)
print( "SUM : " , np.sum(y))


''' 
1. a = np.array([0.3, 2.9, 4.0]) 일 때,

    1) y=softmax(a) 의 경우
    
    a = [0.3 2.9 4. ]
    c = 4.0
    exp_a = [0.02472353 0.33287108 1.        ]
    sum_exp_a = 1.3575946101684189
    y = [0.01821127 0.24519181 0.73659691]
    Softmax Result : [0.01821127 0.24519181 0.73659691]
    SUM :  1.0

    2) y=softmax1(a) 의 경우
    
    a = [0.3 2.9 4. ]
    c = 4.0
    exp_a = [ 1.34985881 18.17414537 54.59815003]
    sum_exp_a = 74.1221542101633
    y = [0.01821127 0.24519181 0.73659691]
    Softmax Result : [0.01821127 0.24519181 0.73659691]
    SUM :  1.0

2. a= np.array([1010, 1000, 990]) 일 때,

    1) y=softmax(a) 의 경우
    
    a = [1010 1000  990]
    c = 1010
    exp_a = [1.00000000e+00 4.53999298e-05 2.06115362e-09]
    sum_exp_a = 1.0000454019909162
    y = [9.99954600e-01 4.53978686e-05 2.06106005e-09]
    Softmax Result : [9.99954600e-01 4.53978686e-05 2.06106005e-09]
    SUM :  1.0
    
    2) y=softmax1(a) 의 경우
    
    a = [1010 1000  990]
    c = 1010
    exp_a = [inf inf inf]
    sum_exp_a = inf
    y = [nan nan nan]
    Softmax Result : [nan nan nan]
    SUM :  nan
    C:/Users/bit-user/Desktop/joohong/MachineLearning/_19_2_Softmax.py:36: RuntimeWarning: invalid value encountered in true_divide
      y = exp_a / sum_exp_a
    
    >> 그냥 지수함수의 상승폭이 너무 크기때문에, 엄청 큰 값이 들어가서 에러발생!
       (꼼수)  exp_a = np.exp(a) 대신  exp_a = np.exp(a-c) 를 사용한다. (c=np.max(a)) 
'''