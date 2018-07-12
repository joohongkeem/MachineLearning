# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180712","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# 다항식 회귀 분석 예시 (Boston 주택 가격 예측)
# - 집값과 관련된 Feature 중 ‘LSTAT’의 값에 따른 집값의 변화를 예측해본다.

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# 보스톤 데이터 set 로드 (scikit-learn)
boston = load_boston()

# 머신러닝 훈련을 위한 데이터프레임 생성
bospd= pd.DataFrame(boston.data)
'''
         CRIM    ZN  INDUS  CHAS  ...      TAX  PTRATIO       B  LSTAT
0     0.00632  18.0   2.31   0.0  ...    296.0     15.3  396.90   4.98
1     0.02731   0.0   7.07   0.0  ...    242.0     17.8  396.90   9.14
2     0.02729   0.0   7.07   0.0  ...    242.0     17.8  392.83   4.03
3     0.03237   0.0   2.18   0.0  ...    222.0     18.7  394.63   2.94
4     0.06905   0.0   2.18   0.0  ...    222.0     18.7  396.90   5.33
5     0.02985   0.0   2.18   0.0  ...    222.0     18.7  394.12   5.21
6     0.08829  12.5   7.87   0.0  ...    311.0     15.2  395.60  12.43
7     0.14455  12.5   7.87   0.0  ...    311.0     15.2  396.90  19.15
8     0.21124  12.5   7.87   0.0  ...    311.0     15.2  386.63  29.93
9     0.17004  12.5   7.87   0.0  ...    311.0     15.2  386.71  17.10
10    0.22489  12.5   7.87   0.0  ...    311.0     15.2  392.52  20.45
11    0.11747  12.5   7.87   0.0  ...    311.0     15.2  396.90  13.27
12    0.09378  12.5   7.87   0.0  ...    311.0     15.2  390.50  15.71
13    0.62976   0.0   8.14   0.0  ...    307.0     21.0  396.90   8.26
14    0.63796   0.0   8.14   0.0  ...    307.0     21.0  380.02  10.26
15    0.62739   0.0   8.14   0.0  ...    307.0     21.0  395.62   8.47
16    1.05393   0.0   8.14   0.0  ...    307.0     21.0  386.85   6.58
17    0.78420   0.0   8.14   0.0  ...    307.0     21.0  386.75  14.67
18    0.80271   0.0   8.14   0.0  ...    307.0     21.0  288.99  11.69
19    0.72580   0.0   8.14   0.0  ...    307.0     21.0  390.95  11.28
20    1.25179   0.0   8.14   0.0  ...    307.0     21.0  376.57  21.02
21    0.85204   0.0   8.14   0.0  ...    307.0     21.0  392.53  13.83
22    1.23247   0.0   8.14   0.0  ...    307.0     21.0  396.90  18.72
23    0.98843   0.0   8.14   0.0  ...    307.0     21.0  394.54  19.88
24    0.75026   0.0   8.14   0.0  ...    307.0     21.0  394.33  16.30
25    0.84054   0.0   8.14   0.0  ...    307.0     21.0  303.42  16.51
26    0.67191   0.0   8.14   0.0  ...    307.0     21.0  376.88  14.81
27    0.95577   0.0   8.14   0.0  ...    307.0     21.0  306.38  17.28
28    0.77299   0.0   8.14   0.0  ...    307.0     21.0  387.94  12.80
29    1.00245   0.0   8.14   0.0  ...    307.0     21.0  380.23  11.98
..        ...   ...    ...   ...  ...      ...      ...     ...    ...
476   4.87141   0.0  18.10   0.0  ...    666.0     20.2  396.21  18.68
477  15.02340   0.0  18.10   0.0  ...    666.0     20.2  349.48  24.91
478  10.23300   0.0  18.10   0.0  ...    666.0     20.2  379.70  18.03
479  14.33370   0.0  18.10   0.0  ...    666.0     20.2  383.32  13.11
480   5.82401   0.0  18.10   0.0  ...    666.0     20.2  396.90  10.74
481   5.70818   0.0  18.10   0.0  ...    666.0     20.2  393.07   7.74
482   5.73116   0.0  18.10   0.0  ...    666.0     20.2  395.28   7.01
483   2.81838   0.0  18.10   0.0  ...    666.0     20.2  392.92  10.42
484   2.37857   0.0  18.10   0.0  ...    666.0     20.2  370.73  13.34
485   3.67367   0.0  18.10   0.0  ...    666.0     20.2  388.62  10.58
486   5.69175   0.0  18.10   0.0  ...    666.0     20.2  392.68  14.98
487   4.83567   0.0  18.10   0.0  ...    666.0     20.2  388.22  11.45
488   0.15086   0.0  27.74   0.0  ...    711.0     20.1  395.09  18.06
489   0.18337   0.0  27.74   0.0  ...    711.0     20.1  344.05  23.97
490   0.20746   0.0  27.74   0.0  ...    711.0     20.1  318.43  29.68
491   0.10574   0.0  27.74   0.0  ...    711.0     20.1  390.11  18.07
492   0.11132   0.0  27.74   0.0  ...    711.0     20.1  396.90  13.35
493   0.17331   0.0   9.69   0.0  ...    391.0     19.2  396.90  12.01
494   0.27957   0.0   9.69   0.0  ...    391.0     19.2  396.90  13.59
495   0.17899   0.0   9.69   0.0  ...    391.0     19.2  393.29  17.60
496   0.28960   0.0   9.69   0.0  ...    391.0     19.2  396.90  21.14
497   0.26838   0.0   9.69   0.0  ...    391.0     19.2  396.90  14.10
498   0.23912   0.0   9.69   0.0  ...    391.0     19.2  396.90  12.92
499   0.17783   0.0   9.69   0.0  ...    391.0     19.2  395.77  15.10
500   0.22438   0.0   9.69   0.0  ...    391.0     19.2  396.90  14.33
501   0.06263   0.0  11.93   0.0  ...    273.0     21.0  391.99   9.67
502   0.04527   0.0  11.93   0.0  ...    273.0     21.0  396.90   9.08
503   0.06076   0.0  11.93   0.0  ...    273.0     21.0  396.90   5.64
504   0.10959   0.0  11.93   0.0  ...    273.0     21.0  393.45   6.48
505   0.04741   0.0  11.93   0.0  ...    273.0     21.0  396.90   7.88
'''

# 데이터프레임의 속성값을 따로 저장
bospd.columns = boston.feature_names
'''
Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'],
      dtype='object')
'''

# boston 객체의 자세한 정보(Description) 호출
# print(boston.DESCR)
'''
Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive
    
    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
**References**

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
'''


# 훈련 데이터 설정 (인구가 낮은 정도)
X_train = bospd[['LSTAT']].values           # bospd 데이터프레임에서 'LSTAT' 속성의 데이터들을 X_train 에 넣는다.
'''
X_train 은 (506,1) 의 <class 'numpy.ndarray'> 
'''

y_train = boston.target
'''
y_train 은 (506,) 의 <class 'numpy.ndarray'> 
'''


# 최소값에서 최대값까지 1씩 증가 데이터 생성
X_test  = np.arange(X_train.min(), X_train.max(), 1)[:, np.newaxis]
    # np.arange(X_train.min(), X_train.max(), 1)은
    #
    # [ 1.73  2.73  3.73  4.73  5.73  6.73  7.73  8.73  9.73 10.73 11.73 12.73
    #  13.73 14.73 15.73 16.73 17.73 18.73 19.73 20.73 21.73 22.73 23.73 24.73
    #  25.73 26.73 27.73 28.73 29.73 30.73 31.73 32.73 33.73 34.73 35.73 36.73
    #  37.73]
    # >> (37,) 의 <class 'numpy.ndarray'>
    #
    # np.arange(X_train.min(), X_train.max(), 1)[:, np.newaxis]은
    #
    # [[ 1.73]
    #  [ 2.73]
    #  [ 3.73]
    #    ...
    #  [35.73]
    #  [36.73]
    #  [37.73]]
    # >> (37,1)의 <class 'numpy.ndarray'>
    #

# 선형 회귀 분석
#
Linear_model = LinearRegression()               # 1. 선형 회귀 모델
Linear_model.fit(X_train, y_train)              # 2. 데이터 학습
Linear_predict = Linear_model.predict(X_test)   # 3. 예측

# 2차 다항식 회귀 분석
#
Poly2_model = LinearRegression()                # 1. 선형 회귀 모델
Poly2 = PolynomialFeatures(degree = 2)          # - 2차
X_train_Trans2 = Poly2.fit_transform(X_train)   # - 2차에 맞게 데이터 변형

Poly2_model.fit(X_train_Trans2, y_train)        # 2. 데이터 학습

X_test_Trans2 = Poly2.fit_transform(X_test)     # - 데이터 예측을위한 test데이터를 2차로 변환
X_test_Pre2 = Poly2_model.predict(X_test_Trans2)# 3. 데이터 예측



# 5차 다항식 회귀 분석
#
Poly5_model = LinearRegression()                # 1. 선형 회귀 모델
Poly5 = PolynomialFeatures(degree = 5)          # - 5차
X_train_Trans5 = Poly5.fit_transform(X_train)   # - 5차에 맞게 데이터 변형

Poly5_model.fit(X_train_Trans5, y_train)        # 2. 데이터 학습

X_test_Trans5 = Poly5.fit_transform(X_test)     # - 데이터 예측을위한 test데이터를 5차로 변환
X_test_Pre5 = Poly5_model.predict(X_test_Trans5)# 3. 데이터 예측


# 평가.... R2
#
print(r2_score(y_train, Linear_model.predict(X_train)))         # 0.5441462975864799
print(r2_score(y_train, Poly2_model.predict(X_train_Trans2)))   # 0.6407168971636611
print(r2_score(y_train, Poly5_model.predict(X_train_Trans5)))   # 0.6816897416931837


plt.scatter(X_train, y_train, label='Training Data', c='grey')
plt.plot(X_test, Linear_predict, linestyle='-', label='Linear Regression', c='green')
plt.plot(X_test, X_test_Pre2, linestyle='--', label='2 Degress Poly Regression', c='red')
plt.plot(X_test, X_test_Pre5, linestyle=':', label='5 Degress Regression',c='blue')
plt.xlabel(" LSTAT")
plt.ylabel(" HOUSE Price")
plt.legend()
plt.show()