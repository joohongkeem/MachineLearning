# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180710","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

import numpy as np


# 미분
#
print("# 미분")
# 미분(접선 기울기) 함수
def getDifferential(f,x):
    h = 1e-4                        # x의 증분 값
    return (f(x+h) -f(x-h))/(2*h)  # 미분 방정식

# 미분을 구하기 위한 3차 방정식
def equation1(x):
    return 10*x**3 + 5*x**2 + 4*x  # a**b는 a의 b제곱을 의미한다

def equation2(x):
    return 0.01*x**2 + 0.1*x

#x = np.arange(-10.0,20.0,0.1)       # x 좌표 값의 범위를 지정한다 (-10부터 20까지 0.1간격)


print(getDifferential(equation1,5))     # 주어진 equation1에대해 좌표 5에서의 기울기를 구한다
                                        # 804가 나와야함
                                        # 출력값 : 804.0000000971759
print(getDifferential(equation2, 10))   # 원래 0.3이 나와야함
                                        # 출력 : 0.2999999999986347

print("-------------------------------------------------")

# 편미분
#
# F(x,y) = x**2 + y**2 일 때,
# 점 x=3, y=4에서의 편미분을 구하시오
print("# 편미분")

def function_1(x):
    return x**2 + 4.0 ** 2              # x에 대한 편미분
def function_2(y):
    return 3.0**2 + y**2                # y에 대한 편미분

print(getDifferential(function_1,3.0))      # Fx(3.0,4.0)
print(getDifferential(function_2,4.0))      # Fy(3.0,4.0)
