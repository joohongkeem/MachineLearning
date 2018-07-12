# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180711","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# 데이터 입출력 기능
# - 다양한 형식 (CSV, txt, Excell, JSON 등)의 데이터 입/출력 기능을 제공
#


print("-------------------------------------------------")

import pandas as pd

# 1. CSV파일
# - 몇 가지 필드를 쉼표(,)로 구분한 텍스트 데이터 및 텍스트 파일이다.
#
# Ex)
# 연도 | 제조사 |                모델              |       설명        |  가격
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 1997 | Ford   | E350                             | ac, abs, moon     | 3000.00
# 1999 | Chevy  | Venture "Extended Edition"       |                   | 4900.00
# 1999 | Chevy  | Venture "Extended Edition, Big"  |                   | 5000.00
# 1996 | Jeep   | Grand Cherokee                   | air, moon roof    | 4799.00
#
# 위의 데이터 표를 csv형식으로 표현하면
#
# 연도,제조서,모델,설명,가격
# 1997,Ford,E350,"ac, abs, moon",3000.00
# 1999,Chevy,"Venture ""Extended Edition""","",4900.00
# 1999,Chevy,"Venture ""Extended Edition, Big""",,5000.00
# 1996,Jeep,Grand Cherokee,"air, moon roo",4799.00
#

# 현재 디렉토리의 iris.csv 파일을 읽기 위해 read.csv 함수 호출
#
iris = pd.read_csv('./iris.csv',
                   names = ['sl','sw','pl','pw','regression'])
print(iris)
print(type(iris))

'''
      sl   sw   pl   pw      regression
0    5.1  3.5  1.4  0.2     Iris-setosa
1    4.9  3.0  1.4  0.2     Iris-setosa
2    4.7  3.2  1.3  0.2     Iris-setosa
3    4.6  3.1  1.5  0.2     Iris-setosa
4    5.0  3.6  1.4  0.2     Iris-setosa
5    5.4  3.9  1.7  0.4     Iris-setosa
6    4.6  3.4  1.4  0.3     Iris-setosa
7    5.0  3.4  1.5  0.2     Iris-setosa
8    4.4  2.9  1.4  0.2     Iris-setosa
9    4.9  3.1  1.5  0.1     Iris-setosa
10   5.4  3.7  1.5  0.2     Iris-setosa
11   4.8  3.4  1.6  0.2     Iris-setosa
12   4.8  3.0  1.4  0.1     Iris-setosa
13   4.3  3.0  1.1  0.1     Iris-setosa
14   5.8  4.0  1.2  0.2     Iris-setosa
15   5.7  4.4  1.5  0.4     Iris-setosa
16   5.4  3.9  1.3  0.4     Iris-setosa
17   5.1  3.5  1.4  0.3     Iris-setosa
18   5.7  3.8  1.7  0.3     Iris-setosa
19   5.1  3.8  1.5  0.3     Iris-setosa
20   5.4  3.4  1.7  0.2     Iris-setosa
21   5.1  3.7  1.5  0.4     Iris-setosa
22   4.6  3.6  1.0  0.2     Iris-setosa
23   5.1  3.3  1.7  0.5     Iris-setosa
24   4.8  3.4  1.9  0.2     Iris-setosa
25   5.0  3.0  1.6  0.2     Iris-setosa
26   5.0  3.4  1.6  0.4     Iris-setosa
27   5.2  3.5  1.5  0.2     Iris-setosa
28   5.2  3.4  1.4  0.2     Iris-setosa
29   4.7  3.2  1.6  0.2     Iris-setosa
..   ...  ...  ...  ...             ...
120  6.9  3.2  5.7  2.3  Iris-virginica
121  5.6  2.8  4.9  2.0  Iris-virginica
122  7.7  2.8  6.7  2.0  Iris-virginica
123  6.3  2.7  4.9  1.8  Iris-virginica
124  6.7  3.3  5.7  2.1  Iris-virginica
125  7.2  3.2  6.0  1.8  Iris-virginica
126  6.2  2.8  4.8  1.8  Iris-virginica
127  6.1  3.0  4.9  1.8  Iris-virginica
128  6.4  2.8  5.6  2.1  Iris-virginica
129  7.2  3.0  5.8  1.6  Iris-virginica
130  7.4  2.8  6.1  1.9  Iris-virginica
131  7.9  3.8  6.4  2.0  Iris-virginica
132  6.4  2.8  5.6  2.2  Iris-virginica
133  6.3  2.8  5.1  1.5  Iris-virginica
134  6.1  2.6  5.6  1.4  Iris-virginica
135  7.7  3.0  6.1  2.3  Iris-virginica
136  6.3  3.4  5.6  2.4  Iris-virginica
137  6.4  3.1  5.5  1.8  Iris-virginica
138  6.0  3.0  4.8  1.8  Iris-virginica
139  6.9  3.1  5.4  2.1  Iris-virginica
140  6.7  3.1  5.6  2.4  Iris-virginica
141  6.9  3.1  5.1  2.3  Iris-virginica
142  5.8  2.7  5.1  1.9  Iris-virginica
143  6.8  3.2  5.9  2.3  Iris-virginica
144  6.7  3.3  5.7  2.5  Iris-virginica
145  6.7  3.0  5.2  2.3  Iris-virginica
146  6.3  2.5  5.0  1.9  Iris-virginica
147  6.5  3.0  5.2  2.0  Iris-virginica
148  6.2  3.4  5.4  2.3  Iris-virginica
149  5.9  3.0  5.1  1.8  Iris-virginica

[150 rows x 5 columns]
<class 'pandas.core.frame.DataFrame'>           -> DataFrame 형태로 읽어준다!!!
'''


# 데이터 입출력 기능(CSV 형식) - head(), tail() 함수

print(iris.head())
print(type(iris.head()))
'''
    sl   sw   pl   pw   regression
0  5.1  3.5  1.4  0.2  Iris-setosa
1  4.9  3.0  1.4  0.2  Iris-setosa
2  4.7  3.2  1.3  0.2  Iris-setosa
3  4.6  3.1  1.5  0.2  Iris-setosa
4  5.0  3.6  1.4  0.2  Iris-setosa
<class 'pandas.core.frame.DataFrame'>
                        >> 위에서부터 default(5)개 출력
                           iris.head(n)으로 쓰면 n개 출력
'''
print(iris.tail(3))
print(type(iris.tail(3)))
'''
      sl   sw   pl   pw      regression
147  6.5  3.0  5.2  2.0  Iris-virginica
148  6.2  3.4  5.4  2.3  Iris-virginica
149  5.9  3.0  5.1  1.8  Iris-virginica
<class 'pandas.core.frame.DataFrame'>
                        >> 아래서부터 3개 출력
                           그냥 iris.tail()로 쓰면 default(5)개 출력
'''

print("-------------------------------------------------")

# 2. txt 파일
# - sep의 값을 기준으로 필드를 구분
#


# iris2.txt 파일은
# 5.1   3.5   1.4   0.2   Iris-setosa
# 4.9   3.0   1.4   0.2   Iris-setosa
# 4.7   3.2   1.3   0.2   Iris-setosa
# 4.6   3.1   1.5   0.2   Iris-setosa
# 의 내용이 담겨있다.

iris2 = pd.read_table('./iris2.txt', sep = '\s+',
                      names = ['sl','sw','pl','pw','regression'])
        # csv파일을 읽을땐, pd.read_csv 였지만, txt파일을 읽을땐 read_table !!! ★★
        # sep에 적혀있는 '\s+' 를 기준으로 필드를 구분한다.
        #       >> \s+ : 길이가 정해져 있지 않은 공백
        #       >> 주로, ',', '\t' 등을 사용
print(iris2)
print(type(iris2))
'''
    sl   sw   pl   pw   regression
0  5.1  3.5  1.4  0.2  Iris-setosa
1  4.9  3.0  1.4  0.2  Iris-setosa
2  4.7  3.2  1.3  0.2  Iris-setosa
3  4.6  3.1  1.5  0.2  Iris-setosa
<class 'pandas.core.frame.DataFrame'>
'''

# skiprow() 메서드
# - 읽어들이지 않을 row(행)를 지정해 줄 수 있음
#
iris3 = pd.read_table('./iris2.txt', sep = '\s+',
                      names = ['sl','sw','pl','pw','regression'],
                      skiprows=[0,2,3])

print(iris3)
print(type(iris3))
'''
0  4.9  3.0  1.4  0.2  Iris-setosa
<class 'pandas.core.frame.DataFrame'>
'''