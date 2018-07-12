# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180711","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# Scikit-Learn 패키지
# - 머신 러닝의 수행 목적에 따라 다음과 같은 머신러닝 모형들을 제공
#
# 지도 학습(Supervised Learning) 모형
# 1) Generalized Linear Models ●
# 2) Linear and Quandratic Discriminat Analysis
# 3) Support Vector Machine ●
# 4) Stochastic Gradient Descent
# 5) Nearest Neighbor Algorithms ●
# 6) Gaussian Process
# 7) Naïve Bayes
# 8) Decision Trees ●
# 9) Ensemble methods(Random Forest)
#
# 비지도 학습(Unsupervised Learning) 모형
# 1) Gaussian mixture models
# 2) Manifold Learning
# 3) Clustering(K-Means) ●
# 4) Biclustering
# 5) Decomposing(PCA, LDA 등)
# 6) Covariance estimation
# 7) Novelty and Outlier Detection
# 8) Density Estimation\
#

# Scikit-Learn 패키지의 전처리
# - Scikit-Learn 패키지에서 제공하는 전처리 기능은 4가지이며,
#   preprocessing과 feature_extraction 패키지로 제공
#
#       - 스케일링 (Scaling)
#           >> 주어진 데이터의 측정 단위에 따른 오차 극복
#
#       - 인코딩 (Encoding)
#           >> 주어진 데이터에 대한 특징을 표현할 수 있도록
#              정수형으로 카테고리화 하기 위해 변환하는 기법
#

print("-------------------------------------------------")

# Scikit-Learn 패키지의 전처리 (Min-Max Scaling)
# - 최댓값과 최솟값을 기준으로 척도를 조정하여 0과 1 사이의 결과 값을 갖는 scaling 기법
#           >> x'_(min-max) = { x - x_(min) } / { x_(max) - x_(min) }
#
# Scikit-Learn 패키지의 전처리 (정규 표준화 척도 : Standard Scaling)
# - 입력 데이터들에 대해 0을 중심으로 표준편차가 1인 정규 분포를 따르게 변환하는 기법
#           >> x'_(standard) = { x - μ_(x) } / δ_(x)
#

# 1. Scikit_Learn 패키지의 전처리 (Min-Max Scaling)
#

from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale
import numpy as np
import pandas as pd


minmax_scaler = MinMaxScaler()


input_data = (np.arange(5, dtype=np.float)-2).reshape(-1,1)
    # np.arange(5) 사용?                            >> [0 1 2 3 4]
    # np.arange(5,dtype=np.float)                   >> [0. 1. 2. 3. 4.]
    # np.arange(5,dtype=np.float)-2                 >> [-2. -1. 0. 1. 2.]
    # (np.arange(5,dtype=np.float)-2).reshape(-1.1) >> [[-2.]
    #                                                   [-1.]
    #                                                   [ 0.]
    #                                                   [ 1.]
    #                                                   [ 2.]]
    #  >> numpy.resahpe 함수는 모양을 재설정해 줄 때 사용한다.
    #     이 때, 맨 앞에 -1을 넣으면, 다른 나머지 차원의 크기를 맞추고 남은 크기를
    #            해당 차원에 할당해준다는 의미!!
    #     (-1, 1) 을 넣었으므로 열을 1열로 하고 행은 알아서 할당!!

minmax_scale_data = minmax_scaler.fit_transform(input_data)
    # .fit : 데이터 프레임을 입력받아 결측값을 채워넣을 적합모형을 생성
    # .transform : 생성된 적합모형을 적용할 데이터 프레임
    #   >> 물론 .fit과 .transform은 동일한 모양을 가져야만 한다
    #
    # .fit_transform : .fit과 .transform 각각 실행한 것으로 한번에 묶어서 실행하는 단축키!
    #

print("평균: ",minmax_scale_data.mean(axis=0))            # 평균:  [0.5]
print("표준편차: ",minmax_scale_data.std(axis=0))        # 표준편차:  [0.35355339]


df1 = pd.DataFrame(np.hstack([input_data, minmax_scale(input_data)]),
                   columns = ["input_data", "minmax_scale(input_Data)"])
    # np.hstack : 행의 수가 같은 두 개 이상의 배열을 옆으로 연결하여,
    #             열의 수가 더 많아지는 배열을 만든다.
    #             연결할 배열은 하나의 리스트에 담아야 한다.
    # np.vstack : 열의 수가 같은 두 개 이상의 배열을 밑으로 연결하여,
    #             행의 수가 더 많아지는 배열을 만든다.
    #             연결할 배열은 하나의 리스트에 담아야 한다.

print(df1)
'''
   input_data  minmax_scale(input_Data)
0        -2.0                      0.00
1        -1.0                      0.25
2         0.0                      0.50
3         1.0                      0.75
4         2.0                      1.00
'''

print("-------------------------------------------------")

# 2. Scikit_Learn 패키지의 전처리 (Standard Scaling)
#

from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale
import numpy as np
import pandas as pd

stdscaler = StandardScaler()

input_data = (np.arange(5, dtype=np.float) - 2).reshape(-1, 1)
# np.arange(5) 사용?                            >> [0 1 2 3 4]
# np.arange(5,dtype=np.float)                   >> [0. 1. 2. 3. 4.]
# np.arange(5,dtype=np.float)-2                 >> [-2. -1. 0. 1. 2.]
# (np.arange(5,dtype=np.float)-2).reshape(-1.1) >> [[-2.]
#                                                   [-1.]
#                                                   [ 0.]
#                                                   [ 1.]
#                                                   [ 2.]]
#  >> numpy.resahpe 함수는 모양을 재설정해 줄 때 사용한다.
#     이 때, 맨 앞에 -1을 넣으면, 다른 나머지 차원의 크기를 맞추고 남은 크기를
#            해당 차원에 할당해준다는 의미!!
#     (-1, 1) 을 넣었으므로 열을 1열로 하고 행은 알아서 할당!!

stdscale_data = stdscaler.fit_transform(input_data)
# .fit : 데이터 프레임을 입력받아 결측값을 채워넣을 적합모형을 생성
# .transform : 생성된 적합모형을 적용할 데이터 프레임
#   >> 물론 .fit과 .transform은 동일한 모양을 가져야만 한다
#
# .fit_transform : .fit과 .transform 각각 실행한 것으로 한번에 묶어서 실행하는 단축키!
#

print("평균: ", stdscale_data.mean(axis=0))  # 평균:  [0.]
print("표준편차: ", stdscale_data.std(axis=0))  # 표준편차:  [1.]

df1 = pd.DataFrame(np.hstack([input_data, scale(input_data)]),
                   columns=["input_data", "stdscale(input_Data)"])
# np.hstack : 행의 수가 같은 두 개 이상의 배열을 옆으로 연결하여,
#             열의 수가 더 많아지는 배열을 만든다.
#             연결할 배열은 하나의 리스트에 담아야 한다.
# np.vstack : 열의 수가 같은 두 개 이상의 배열을 밑으로 연결하여,
#             행의 수가 더 많아지는 배열을 만든다.
#             연결할 배열은 하나의 리스트에 담아야 한다.

print(df1)
'''
   input_data  stdscale(input_Data)
0        -2.0             -1.414214
1        -1.0             -0.707107
2         0.0              0.000000
3         1.0              0.707107
4         2.0              1.414214

'''



print("-------------------------------------------------")

# 3. Scikit_Learn 패키지의 전처리 (One-Hot Encoding)
#   - 주어진 데이터가 0에서 k-에 대해 0과 1의 값으로 구성되는 k 차원의 벡터를 변환
#   - 주어진 데이터의 카테고리에서 한 요소만 1의 값을 갖는 벡터를 만드는 과정
#
#   [속성]
#   n_values_ - 각 특징 당 분류할 수 있는 개수
#   feature_indices_ - 입력 데이터가 벡터인 경우 각 원소를 나누기 위한 값
#   active_features_ - 실제로 분류를 위해 사용되는 색인 값
#
#   [메소드]
#   fit(X[,y])           - X 값에 대해 One hot Encoder를 맞춤
#                        - fit 이후 n_values_, feature_indices_, active_features_를 사용할 수 있음.
#   transform(X)         - X를 One hot Encoding을 사용하여 변환함.
#   fit_transform(X[,y]) - fit메소드 사용 후 transform 메소드를 사용하는 것과 같음.
#


# 1차원 벡터의 One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
import numpy as np
ohe = OneHotEncoder()
X = np.array([[0],[1],[2]])
print("X :",X)
print("x' onehot result :",ohe.fit_transform(X).toarray())

print("입력 값의 구분 갯수 :",ohe.n_values_)
print("원소들이 어떻게 나누어졌나 :", ohe.feature_indices_)
print("색인값 :", ohe.active_features_)

''' 
[출력결과]
X : 
[[2]
 [1]
 [0]]
x' onehot result : 
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
입력 값의 구분 갯수 : [3]
원소들이 어떻게 나누어졌나 : [0 3]
색인값 : [0 1 2]
'''

print("-------------------------------------------------")

# 4. Scikit_Learn 패키지의 전처리 (Label Binarizer)
# - 주어진 입력 데이터에서 문자열 라벨 정보에 대해 One hot Encoding을 하기 위해 사용되는 모듈
#

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()                       # LabelBinarizer 모듈 호출
X = ['A', 'B', 'C', 'D', 'A', 'B', 'F']
print(lb.fit(X))                            # abelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

    # 인코딩에 대한 분류 기준 정보
print('어떻게 분류 했나 :',lb.classes_)
    # LabelBinarizer에 의한 인코딩된 값 확인
print('0과 1로 변형된 값 :',lb.transform(X))

''' 
[출력결과]
어떻게 분류 했나 : 
['A' 'B' 'C' 'D' 'F']
0과 1로 변형된 값 :
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 0 0 1]]

 '''

print("-------------------------------------------------")

# 4. Scikit_Learn 패키지의 전처리 (Binarizer)
#   - 주어진 입력 값에 대해 0과 1의 값으로 인코딩 시
#     미리 선정된 임계 값(Threshold)을 기준으로 인코딩의 값을 결정하는 기법
#

    # Binarizer 모듈 선언
from sklearn.preprocessing import Binarizer

    # Binarizer 생성
binarizer = Binarizer()                 # Threshold default 값은 0
                                        # 0을 기준으로 크면 1, 작거나 같으면 0
    # 2차원 입력 데이터 선언
X = np.array([[1., -1.],[-1., 0.],[0., 2.]])

    # X 데이터 확인
print(X)
'''
[[ 1. -1.]
 [-1.  0.]
 [ 0.  2.]]
 '''

    # Binarizer 인코딩 변환 수행
print(binarizer.transform(X))
'''
[[1. 0.]
 [0. 0.]
 [0. 1.]]
 '''

print("-------------------------------------------------")

# 4-1. Scikit_Learn 패키지의 전처리 (Binarizer with Threshold)
    # Binarizer 모듈 선언
from sklearn.preprocessing import Binarizer

    # Binarizer 생성
binarizer = Binarizer(threshold=1.5)            # threshold 를 1.5로 설정해줌!!

    # 2차원 입력 데이터 선언
X = np.array([[1., -1.],[-1., 0.],[0., 2.]])

    # X 데이터 확인
print(X)
'''
[[ 1. -1.]
 [-1.  0.]
 [ 0.  2.]]
 '''

    # Binarizer 인코딩 변환 수행
print(binarizer.transform(X))
'''
[[0. 0.]
 [0. 0.]
 [0. 1.]]
 '''