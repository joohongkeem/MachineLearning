# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180711","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# One-Hot Encoder
#
# - 0 ~ K-1 의 값을 가지는 정수 스클라 값을 0 또는 1의 값을 갖는 K-차원 벡터로 변환한다.
# - 만약, 입력이 스칼라가 아니라 벡터이면, 각 원소에 대해 인코딩된 결과를 모두 연결한다.★
# - 각 원소의 위치 정보는 'feature_indices_' 속성에 저장된다.
# - 또 입력이 벡터인 경우에 특정한 열만 카테고리 값이면,
#   'categorical_features' 인수를 사용하여 인코딩이 되지 않도록 지정할 수 있다.
#       >> 이 때, 인코딩 결과의 순서가 바뀔 수 있으므로 주의한다.
#

# .fit : 데이터 프레임을 입력받아 결측값을 채워넣을 적합모형을 생성
# .transform : 생성된 적합모형을 적용할 데이터 프레임
#   >> 물론 .fit과 .transform은 동일한 모양을 가져야만 한다
#
# .fit_transform : .fit과 .transform 각각 실행한 것으로 한번에 묶어서 실행하는 단축키!
#
#       fit 메서드를 호출하면 다음과 같은 속성이 지정된다.
#           - n_values_ : 각 변수의 최대 클래스 갯수
#           - feature_indices_ : 입력이 벡터인 경우 각 원소를 나타내는 슬라이싱(slice) 정보
#           - active_features_ : 실제로 사용된 클래스 번호의 리스트

# (참고)
# - One Hot Encoding 결과는 메모리 절약을 위해 스파스 행렬(sparse matrix) 형식으로 출력되므로,
#   일반적인 배열로 바꾸려면 'toarray'메서드를 사용한다.

from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder()

X = np.array([[0],[1],[2]])
print(X)
'''
[[0]
 [1]
 [2]]
 '''

ohe.fit(X)
print(ohe.n_values_, ohe.feature_indices_, ohe.active_features_)
# [3] [0 3] [0 1 2]

X = np.array([[0, 0, 4], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
print(X)
'''
[[0 0 4]
 [1 1 0]
 [0 2 1]
 [1 0 2]]
 '''

ohe.fit(X)
print(ohe.n_values_, ohe.feature_indices_, ohe.active_features_)
# [2 3 5] [ 0  2  5 10] [0 1 2 3 4 5 6 7 9]

print(ohe.transform(X).toarray())
'''
[[1. 0. 1. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 1. 0. 1. 0. 0.]
 [0. 1. 1. 0. 0. 0. 0. 1. 0.]]
 '''

ohe = OneHotEncoder(categorical_features=[False, True, False])
print(ohe.fit_transform(X).toarray())
'''
[[1. 0. 0. 0. 4.]
 [0. 1. 0. 1. 0.]
 [0. 0. 1. 0. 1.]
 [1. 0. 0. 1. 2.]]
'''
print(ohe.n_values_, ohe.feature_indices_, ohe.active_features_)
# [3] [0 3] [0 1 2]