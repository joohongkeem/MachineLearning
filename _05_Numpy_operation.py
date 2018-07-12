# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180711","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

import numpy as np

x = np.array([1.0,2.0,3.0,4.0])
print(x)                                # [1. 2. 3. 4.]
print(type(x))                          # <class 'numpy.ndarray'>


x1 = np.array([[1,2,3],[4,5,6]])
print(x1)                               # [[1 2 3]
                                        #  [4 5 6]]
print(type(x1))                         # <class 'numpy.ndarray'>
print(x1.shape)                         # (2, 3)

x2 = np.zeros(10)
print(x2)                               # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(type(x2))                         # <class 'numpy.ndarray'>
print(x2.shape)                         # (10,)

x2 = np.zeros((3,2))
print(x2)                               # [[0. 0.]
                                        #  [0. 0.]
                                        #  [0. 0.]]
print(type(x2))                         # <class 'numpy.ndarray'>
print(x2.shape)                         # (3, 2)


x3 = np.ones(10)
print(x3)                               # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(type(x3))                         # <class 'numpy.ndarray'>
print(x3.shape)                         # (10,)

x3 = np.ones((3,2))
print(x3)                               # [[1. 1.]
                                        #  [1. 1.]
                                        #  [1. 1.]]
print(type(x3))                         # <class 'numpy.ndarray'>
print(x3.shape)                         # (3, 2)

print("-------------------------------------------------")

# 기본연산
# - 사칙연산은 기본적으로 각 항끼리 계산한다.

import numpy as np

x = np.array([[1,2,3],[4,5,6]])

x1 = x + x
print(x1)                               # [[ 2  4  6]
                                        #  [ 8 10 12]]
x2 = x1 - x
print(x)                                # [[1 2 3]
                                        #  [4 5 6]]
x3 = x * x
print(x3)                               # [[ 1  4  9]
                                        #  [16 25 36]]

x4 = x3 / x1            
print(x4)                               # [[0.5 1.  1.5]
                                        #  [2.  2.5 3. ]]

print("-------------------------------------------------")
# 브로드캐스팅
# - 배열(행렬) 간의 연산을 위하여 배열의 형태를 자동으로 일치 시켜주는 numpy의 기능

x = np.array([[1,2],[3,4]])
y = 10

x1 = x + y
print('x type :', type(x))             # x type : <class 'numpy.ndarray'>
print('y type :', type(y))             # y type : <class 'int'>
print(x1)                                # [[11 12]
                                         #  [13 14]]
                                         # >> int형 10인 y를 자동으로 [[10,10],[10,10]]으로 변환

x = np.array([[1,2],[3,4]])
y = np.array([10,20])

x1 = x + y
print('x shape :', x.shape)             # x shape : (2, 2)
print('y shape :', y.shape)             # y shape : (2,)
print(x1)                                 # [[11 22]
                                          #  [13 24]]
                                          # >> (2, ) 형태의 y를 자동으로 [[10,20],[10,20]]로 변환

print("-------------------------------------------------")

# 슬라이싱
# - 파이썬의 리스트에서 제공하는 슬라이싱과의 차이점에 주의하자
#

# 파이썬 리스트에서의 슬라이싱에 의한 값 변환
#
x1 = list(range(10))                        # 0 ~ 9 까지 10개의 원소를 갖는 배열 생성
print(x1)                                   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

x2 = x1[0:3]                                # x1 배열 중 0에서 4개의 원소 값을 슬라이싱
print(x2)                                   # [0, 1, 2]

x2[1] = 100
print(x2)                                   # [0, 100, 2]
                                            # >> 슬라이싱한 리스트에 변경 값 반영

print(x1)                                   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                                            # >> 원래의 리스트에는 변경된 값이 반영 안됨

# numpy에서의 slicing에 의한 값 변환
#
import numpy as np

x = np.array(range(10))
print(x,type(x))                                    # [0 1 2 3 4 5 6 7 8 9] <class 'numpy.ndarray'>

x2 = x[0:3]
print(x2,type(x2))                                  # [0 1 2] <class 'numpy.ndarray'>

x2[1] = 100
print(x2,type(x2))                                  # [  0 100   2] <class 'numpy.ndarray'>

print(x)                                            # [  0 100   2   3   4   5   6   7   8   9]
                                                    # >> 원래의 배열의 값이 변경!!! ★★★
                                                    #    (포인터의 개념으로 생각하자!!)