# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180717", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")


#   숫자인식 예제
#   PythonProgramming.net 참조



import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# 1. 데이터 생성
digits = datasets.load_digits()
X_train, y_train = digits.data, digits.target
print(X_train, X_train.sahpe, type(X_train), sep='\n')
print(y_train, y_train.shape, type(y_train), sep='\n')
'''
    X_train
[[ 0.  0.  5. ...  0.  0.  0.]
 [ 0.  0.  0. ... 10.  0.  0.]
 [ 0.  0.  0. ... 16.  9.  0.]
 ...
 [ 0.  0.  1. ...  6.  0.  0.]
 [ 0.  0.  2. ... 12.  0.  0.]
 [ 0.  0. 10. ... 12.  1.  0.]]
(1797, 64)                          # 64인 것은 8x8 이기 때문
<class 'numpy.ndarray'>

    y_train
[0 1 2 ... 8 9 8]
(1797,)
<class 'numpy.ndarray'>

'''
digits_index = 1252            # 1252번째에 있는 데이터로 확인해보자!
#print(y_train[1252])           # 1252번째 데이터의 진짜 숫자는 6


# 2. 모델 생성
svm_model = svm.SVC(gamma=0.0001, C=100)
    # gamma를 사용하는 커널이 있다 ex) 지수함수

# 3. 학습
svm_model.fit(X_train, y_train)




plt.imshow(digits.images[digits_index], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()