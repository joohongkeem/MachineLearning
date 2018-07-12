# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180712","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")



# 오그멘테이션 (Augmentation)
#

# 방법 1. statsmodels.api 사용
import statsmodels.api as sm
import numpy as np

X1 = np.array([[10], [20], [30], [50]])
# [[10]
#  [20]
#  [30]
#  [50]]
X_train = sm.add_constant(X1)  # 오그멘테이션
# X_train 출력하면 일캐나온다.
# [[ 1. 10.]
#  [ 1. 20.]
#  [ 1. 30.]
#  [ 1. 50.]]

# 방법 2. 직접 코딩


print(X1.shape)  # (4,1)

print(X1.shape[0])  # 4

print(np.ones((X1.shape[0], 1)))
# np.ones(10) 하면 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# np.ones((3,2)) 하면
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]
# 즉, np.ones((4,1)) 하면
# [[1.]
# [1.]
# [1.]
# [1.]]

X_train1 = np.hstack([np.ones((X1.shape[0], 1)), X1])

print(X_train1)
# [[ 1. 10.]
#  [ 1. 20.]
#  [ 1. 30.]
#  [ 1. 50.]]