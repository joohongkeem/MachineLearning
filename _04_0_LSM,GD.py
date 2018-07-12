# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180710","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# 학습 기법
# - 머신 러닝은 주어진 데이터를 기반으로 모델을 학습하여 최적의 매개 변수 값을 구하는 과정
#
#       * 최소 제곱법(least squares:LSM)
#       - 직관적이며 간단하기 때문에 회기분석, 신경망 분석 등에 다양한 머신 러닝 및
#         통계 분석에 많이 사용되고 있는 기법
#
#       * 최우 추정법(Maximum likehood meathod : MLE)
#       - 확률이라는 개념을 통해 최적의 계수를 설정하는 기법으로,
#         주어진 데이터의 발생 가능성을(likelihood) 가장 크게 하는 모수(Parameter)을 찾는 방법
#
#       * 경사 하강법(Gradient Descent : GD) ★
#       - 최적의 매개변수를 찾기 위해 손실 함수를 정의하고, 손실 함수가 최소값이 될 때의 매개변수를 찾는 방법
#           >> 신경망, 딥러닝 등에 가장 많이 사용
#

print("-------------------------------------------------")

print("# 경사 하강법(Gradient Descent)")

# 경사 하강법(Gradient Descent)
# - 모든 차원가 모든 공간에서의 적용이 가능하다


x_old = 0
x_new = 6 # The algorithm starts at x=6
eps = 0.01 # step size
precision = 0.00001

def f_prime(x):
    return 4 * x**3 - 9 * x**2

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new = x_old - eps * f_prime(x_old)
print ("Local minimum occurs at ", x_new)