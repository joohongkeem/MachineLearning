# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180712","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")


# 선형 회귀 분석 예시1 (최소 제곱법 - 1)
#

import matplotlib.pyplot as plt
import numpy as np

def predict(x):                     # 예측값을 구하는 함수
    return w0 + w1 * x

sample_data = [[10, 25], [20, 45], [30, 65], [50, 105]]     # W0, W1을 구하기 위한 Data

X_train = []
y_train = []
X_train_a = []                           # Matplot 으로 그림을 그리기 위한 x축 좌표들
y_train_a = []                           # Matplot 으로 그림을 그리기 위한 y축 좌표들
total_size = 0                           # Sample Data의 총 개수(n)
sum_xy = 0                               # Σ(x*y)
sum_x = 0                                # Σ(x)
sum_y = 0                                # Σ(y)
sum_x_square = 0                         # Σ(x^2)

for row in sample_data:                   # row는 [10,25]->[20,45]->[30,65]->[50,105] 순서로 돈다
    X_train = row[0]
    y_train = row[1]
    X_train_a.append(row[0])
    y_train_a.append(row[1])
    sum_xy += X_train * y_train
    sum_x += X_train
    sum_y += y_train
    sum_x_square += X_train * X_train
    total_size += 1
w1 = (total_size * sum_xy - sum_x * sum_y) / (total_size * sum_x_square - sum_x * sum_x)
w0 = (sum_x_square * sum_y - sum_xy * sum_x) / (total_size * sum_x_square - sum_x * sum_x)
X_test = 40
y_predict = predict(X_test)
print("가중치: ", w1)
print("상수 : ", w0)
print("예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)



# 그래프 그려보기
#

x_new = np.arange(0,51)                             # 직선을 그리기 위한 0부터 50까지의 x data
y_new = predict(x_new)                              # 위의 x데이터를 대입한 예측 y결과
                                                    # >> 직선이 모든 점을 지나는지 확인할 수 있다!

plt.scatter(X_train_a, y_train_a, label = "data")           # 점을 찍는다!!
plt.scatter(X_test, y_predict, label="predict")
plt.plot(x_new, y_new,'r-', label = "regression")         # 그래프를 그린다!!
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Linear Regression")
plt.legend()                                # Data의 종류 표시 (data는 파란색, predict는 주황색)
plt.show()
y_predict

print("-------------------------------------------------")

# 선형 회귀 분석 예시2 (numpy 기반의 행렬 연산)
#

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def predict(x):
    return w0 + w1*x

X1 =np.array([ [10], [20],[30], [50]])
                        # [[10]
                        #  [20]
                        #  [30]
                        #  [50]]
y_label =np.array([ [25], [45],[65],  [105] ])

X_train = sm.add_constant(X1)   # 오그멘테이션
                                # X_train 출력하면 일캐나온다.
                                # [[ 1. 10.]
                                #  [ 1. 20.]
                                #  [ 1. 30.]
                                #  [ 1. 50.]]
w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_label)

print('w',w,sep='\n')
# 2 * 4 행렬 .  4*1  행렬 --> \ : 2 * 1 행렬
w0 = w[0]
w1 = w[1]

X_test = 40
y_predict = predict(X_test)
print("가중치: ", w1)
print("상수 : ", w0)
print("예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)


print("-------------------------------------------------")

# 선형 회귀 분석 예시3 (scikit_learn 라이브러리 사용)
#

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

    # 1. 모델 객체 생성 : 모델 클래스를 선언하여 모델 객체를 생성
model = LinearRegression(fit_intercept=True)        # 상수항이 있으면

    # 2. 학습 : 데이터셋 X, 타겟값 y를 입력으로 받아
    #           fit(X,y) 함수를 이용하여 생성된 모델 객체를 학습시킨다
X_train =np.array([ [10], [20],[30], [50]])
y_train =np.array([ [25], [45],[65], [105] ])
model.fit(X_train, y_train)

    # 3. 예측 : 하나 혹은 복수의 데이터 X를 입력받아 학습시킨 모델 객체를 이용하여
    #           predict(X) 함수로 타겟값 y를 예측한다.
X_test = 40
y_predict = model.predict(X_test)       # y_predict : 예측값을 넣었을 때의 결과값
y_pred = model.predict(X_train)         # y_pred : Sample Data를 넣었을 때의 결과값

    # mean_squared_error(predictions, targets)
    # Sample Data를 넣었을 때 결과값과 실제 y_train을 비교한다.
mse = mean_squared_error(y_pred, y_train)
print(mse)

print("가중치: ", model.coef_)
print("상수 : ", model.intercept_)
print("예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)


print("-------------------------------------------------")

# 선형 회귀 분석 예시4 (Random 데이터를 통한 분석)
#

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# scikit_learn 에서 제공하는 데이터 제공 함수
X_train, y_train, coef = \
    make_regression(n_samples=50, n_features=1, bias=50, noise=20,coef=True, random_state=1)
    # 입력
    # n_samples = 50        >> 표본 데이터의 갯수는 50개
    # n_features = 1        >> 독립변수(feature)의 차원은 1
    # n_targets = default   >> 종속변수(taget)의 차원은 1(default)
    # bias = 50             >> y절편은 50
    # noise = 20            >> 종속변수(출력)에 더해지는 오차의 표준편차
    # coef = True           >> 선형 모형의 계수도 출력
    # random_state = 1      >> 난수 발생용 시드값
    #
    # 출력
    # X : [n_samples, n_features] 형상의 2차원 배열 & 독립변수의 표본 데이터 행렬
    # y : [n_samples] 형상의 1차원 배열 또는 [n_samples, n_targets] 형상의 2차원 배열
    #     & 종속 변수의 표본 데이터 벡터 y
    # coef : [n_features] 형상의 1차원 배열 또는 [n_features, n_targets] 형상의 2차원 배열
    #        >> 선형 모형의 계수 벡터 w
model = LinearRegression(fit_intercept=True) # 상수항이 있으면

model.fit(X_train, y_train)


# 선형회귀 직선을 작성하기 위해 데이터 생성
# X_train 데이터의 최대, 최소 값 사이를 100의 데이터로 구분한다.
x_new = np.linspace(np.min(X_train), np.max(X_train), 100)
        # y = linspace(x1,x2)는 x1과 x2 사이에서 균일한 간격의 점 100개로 구성된 행 벡터를 반환합니다.
        # y = linspace(x1,x2,n)은 n개의 점을 생성합니다. 점 사이의 간격은 (x2-x1)/(n-1)입니다.
        # default = 50

# 1행 N열의 데이터를 N행 1열로 reshape
X_new = x_new.reshape(-1, 1)

# 그래프를 그리기 위한 y 예측 값 --> 직선을 그리기 위한 x 값과 그에 따른 y 값 정의
y_predict = model.predict(X_new)

# 예측 값
y_pred = model.predict(X_train)

mse = mean_squared_error(y_train, y_pred)
print('mse =',mse)

# 그래프 그려보기
plt.scatter(X_train, y_train, c='r', label="data")
plt.plot(X_new, y_predict, 'g-', label="regression")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()

