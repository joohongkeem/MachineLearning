# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180712","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# n_degree 의 값을 변화시켜가며 그래프 모양의 변화, 다항 회귀 분석의 r2_score의 변화를 살펴보자!
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# 데이터 생성
#

def make_nl_sample():
    np.random.seed(0)
    samples_number = 50
    X = np.sort(np.random.rand(samples_number))
    y = np.sin(2 * np.pi * X) + np.random.randn(samples_number) * 0.2
    X = X[:, np.newaxis]
    return (X, y)

X_train, y_train = make_nl_sample()

# 트레이닝 데이터의 분포 확인
#

plt.scatter(X_train, y_train, label='Training Data')
plt.title("# 1.Training Data")
plt.show()




# 선형 회귀 분석
#
# 1. 모델 객체 생성 : 모델 클래스를 선언하여 모델 객체를 생성
# 2. 학습 : 데이터셋 X, 타겟값 y를 입력으로 받아
#           fit(X,y) 함수를 이용하여 생성된 모델 객체를 학습시킨다
# 3. 예측 : 하나 혹은 복수의 데이터 X를 입력받아 학습시킨 모델 객체를 이용하여
#           predict(X) 함수로 타겟값 y를 예측한다.
#

model = LinearRegression().fit(X_train,y_train)
linear_predict = model.predict(X_train)

# 선형 회귀 분석 - 그래프 출력
#

plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, linear_predict, label='Linear Regression', color='b')
plt.legend()
plt.title("# 2.Linear Regression")
plt.show()

# 다항식 회귀 분석
#

# 1. 다항식 회귀 모델의 생성 및 훈련
poly_linear_model = LinearRegression()
    # 차수에 맞는 형태의 데이터 형태 변환
n_degree = 3
polynomial = PolynomialFeatures(n_degree)


# 2. 다항식 회귀 모델에 맞는 데이터 변환 및 학습
X_train_transformed = polynomial.fit_transform(X_train)
poly_linear_model.fit(X_train_transformed, y_train)

# 3. 다항식 회귀 모델에 X_train 값을 적용 (예측)
poly_predict = poly_linear_model.predict(X_train_transformed)

# 4. R2 스코어 계산 (선형 회귀모델과 다항식 회귀모델 각각)
linear_r2 = r2_score(y_train, linear_predict)
poly_r2 = r2_score(y_train, poly_predict)


# 다항식 회귀 분석 - 그래프 출력
#

plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, linear_predict, label='Linear Regression', color='b')
plt.plot(X_train, poly_predict, label='Poly Regression', color='r')
plt.title("Degree : {}\n linear_r2_score : {:.2e}\n poly_r2_score : {:.2e} ".format(n_degree, linear_r2, poly_r2))
plt.legend()
plt.show()

# linear_r2 : 4.72e-01
#
# degree :   2 일 때, poly_r2 : 4.84e-01
# degree :   3 일 때, poly_r2 : 9.06e-01
# degree : 100 일 때, poly_r2 : 9.71e-01
# degree :1000 일 때, poly_r2 : 9.70e-01      >> Overfitting ★★


