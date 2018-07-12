# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180710","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")
# 손실 함수
#   - 머신 러닝 모델에 의한 예측 값과 목표 값 사이의 차를 말하며, 이를 표현한 함수
#
#       * 손실이 크다 ? 예측 값과 목표 값의 차이가 크다
#       * 손실이 작다 ? 예측 값과 목표 값의 차이가 작다
#
print("-------------------------------------------------")
print("# 손실함수 - 평균 제곱 오차(Mean Squared Error)")
# 평균 제곱 오차(Mean Squared Error)
# - 실제 목표 값과 학습을 통한 머신 러닝의 출력 값 사이의 거리를 오차로 하는 값의 평균
#

import numpy as np

# 타겟(목표)
target = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]                 # 7번 숫자를 의미하는 목표 데이터 (one-hard Encoding)


# 머신 러닝 모델에 의한 예측 데이터
y1 = [0.1, 0, 0, 0.1, 0.05, 0.05, 0, 0.7, 0, 0]         # 0일 확률 : 10%, 3일 확률 : 10%, 4일 확률 : 5%, ...
y2 = [0, 0.1, 0.7, 0.05, 0.05, 0.1, 0, 0, 0, 0]

def mean_seq_error(t,y):
    return 1/10 * np.sum((t-y)**2)

print(mean_seq_error(np.array(target),np.array(y1)))                # 0.0575
print(mean_seq_error(np.array(target),np.array(y2)))                # 0.7575

# y1이 평균 제곱 오차값이 더 작으므로 y1이 더 좋은 예측모델
# y2의 손실이 훨씬 크다

print("-------------------------------------------------")
print("# 손실함수 - 교차 엔트로피 오차(Cross Entropy Error)")
# 교차 엔트로피 오차(Cross Entropy Error)
# - 로그 함수를 기반으로 오류(Error)를 정의한 교차 엔트로피
#       >> 인공 신경망 모델 등의 학습에서 자주 사용
#       >> 평균 제곱 오차에 비해 오차의 값에 더욱 민감


# 타겟(목표)
target = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]                 # 7번 숫자를 의미하는 목표 데이터 (one-hard Encoding)


# 머신 러닝 모델에 의한 예측 데이터
y1 = [0.1, 0, 0, 0.1, 0.05, 0.05, 0, 0.7, 0, 0]         # 0일 확률 : 10%, 3일 확률 : 10%, 4일 확률 : 5%, ...
y2 = [0, 0.1, 0.7, 0.05, 0.05, 0.1, 0, 0, 0, 0]

def cross_entropy(t,y):
    tmp = 1e-7                                          # 로그 0 에 의한 무한대 발생 방지 값
    return -np.sum(t*np.log(y+tmp))                    # cross entropy 계산
print(cross_entropy(np.array(target),np.array(y1)))     # 0.3566748010815999
print(cross_entropy(np.array(target),np.array(y2)))     # 16.11809565095832
