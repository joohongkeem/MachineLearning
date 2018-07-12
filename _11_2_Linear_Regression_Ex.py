# -*- coding: utf-8 -*-
# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180712","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# 선형 회귀 분석 예시 5(Boston 주택 가격 예측)
# - Boston Housing DataSet에는 주택의 가격과 함께 범죄(CRIM), 집을 소유한 사람들의 나이(AGE) 등과 같은 다양한 정보도 같이 제공
# - Scikit-Learn 패키지에서는 파이썬 라이브러리를 통해 Boston Housing Dataset 데이터를 바로 가져올 수 있는 기능을 제공
# - Boston Housing Dataset은 13개의 속성과 1개의 주택 가격으로 구성

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# boston 객체를 불러온다.
boston = load_boston()

# boston 객체의 자세한 정보(Description) 호출
print(boston.DESCR)

print(type(boston.DESCR))
# <class 'str'>

print("boston.data.shape :", boston.data.shape, " type :", type(boston.data))
    # boston.data.shape : (506, 13)  type : <class 'numpy.ndarray'>
print("boston.target.shape :", boston.target.shape)
    # boston.target.shape : (506,)


# Training Data/Test Data 나누기 ★★★ (전체 데이터를 학습데이터/테스트데이터로 적절히 나눠준다)
# >> 나누는 이유? 오버피팅 방지 !!
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

print(X_train.shape)
    # (379, 13)
print(X_test.shape)
    # (127, 13)



# 선형 회귀 분석
#
# 1. 모델 객체 생성 : 모델 클래스를 선언하여 모델 객체를 생성
# 2. 학습 : 데이터셋 X, 타겟값 y를 입력으로 받아
#           fit(X,y) 함수를 이용하여 생성된 모델 객체를 학습시킨다
#

model_boston = LinearRegression().fit(X_train,y_train)       # 모델 객체 생성과 학습을 한번에!

print('model_boston의      Coef(가중치) :', model_boston.coef_ )
    # [-1.16869578e-01  4.39939421e-02 -5.34808462e-03  2.39455391e+00
    #  -1.56298371e+01  3.76145473e+00 -6.95007160e-03 -1.43520477e+00
    #   2.39755946e-01 -1.12937318e-02 -9.86626289e-01  8.55687565e-03
    #  -5.00029440e-01]
print('model_boston의 intercept( 절편 ) :', model_boston.intercept_ )
    # 36.98045533762074


# 테스트 데이터에 대한 예측 수행하기
#
# 3. 예측 : 하나 혹은 복수의 데이터 X를 입력받아 학습시킨 모델 객체를 이용하여
#           predict(X) 함수로 타겟값 y를 예측한다.
#

predictions = model_boston.predict(X_test)

# 평가
#
print("# Training Set 점수 ")
print(model_boston.score(X_train,y_train))
    # 0.7697448370563938
print("# Test Set 점수")
print(model_boston.score(X_test,y_test))
    # 0.6353620786674621                    R2_score와 결과값이 같다! >> .score하면 r2_score로 계산하는듯!

# 테스트 데이터와 테스트 목표 값을 넣어주면, 예측 값을 만들어 비교 해 준다.
# -> R2 Score ★★ (1이 되면 데이터가 똑같다!)
print("# R2_score ")
print(r2_score(y_test, predictions))
    # 0.6353620786674621

print("# mse ")
print(mean_squared_error(y_test, predictions))
    # 29.790559164238505


# get_error(y_test, predictions)
#plt.scatter(X_train, y_train, label = 'Training')
#plt.scatter(X_test, y_test, label = 'Test')
#plt.scatter(X_test, predictions, label = 'Test_Predict')
plt.scatter(y_test, predictions)
plt.xlabel(" Real Boston House Price")
plt.ylabel(" Predict Boston House Price")
plt.legend()
plt.show()