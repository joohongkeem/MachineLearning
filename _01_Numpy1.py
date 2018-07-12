# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180710","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

import numpy as np

'''
[1 2 3]
[4 5 6]
'''

a = np.array([[1,2,3],[4,5,6]]) # numpy 모듈
print(a.shape)                  # 행렬의 모양 출력 -> (2,3)
print(a)                        # 행렬 출력  -> [[1 2 3]
                                #                [4 5 6]]


b = np.array([[7,8],[9,10],[11,12]])
print(b.shape)


# c = a행렬과 b행렬의 내적
print("# 내적 결과")
c = np.dot(a,b)
print(c.shape)
print(c)

print("-------------------------------------------------")

# 전치행렬
print("# 전치행렬")
a = np.array([[1,2,3],[4,5,6]])
at = a.T
print(a)
print(at)

print("-------------------------------------------------")

import numpy.linalg as lin

# 역행렬
print('# 역행렬')
a = np.array([[1,2],[4,5]])
print(a)

ai = lin.inv(a)
print(ai.shape)
print(ai)

E = np.dot(a,ai)
print(E)            # [[1.00000000e+00 0.00000000e+00]
                    #  [2.22044605e-16 1.00000000e+00]]  을 단위행렬 E로 보면 된다~!

print("-------------------------------------------------")
