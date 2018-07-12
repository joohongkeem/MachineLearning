# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180711","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# Pandas 패키지
# - 데이터 분석 작업의 편의성을 제공하는 다양한 기능을 가지고 있음.
#       >> 데이터 처리를 위한 Series 클래스, DataFrame 클래스 등 제공
#       >> 편리한 데이터 입출력 기능 제공 등
#

print("-------------------------------------------------")

# 데이터프레임(DataFrame) 클래스
# - 다수의 컬럼으로 구성된 2차원 형태의 자료구조로,
#   2차원 구조의 데이터를 쉽게 조작할 수 있는 기능을 제공
#
from pandas import Series,DataFrame

# dict형을 data로 이용
data = {'지역':['서울','서울','서울','인천','인천'],
        '연도':[2010, 2011, 2012, 2011, 2012],
        '가격':[10, 15, 20, 3, 5]}

# DataFrame 생성
df = DataFrame(data)

print(df)
'''
   지역    연도  가격
0  서울  2010  10
1  서울  2011  15
2  서울  2012  20
3  인천  2011   3
4  인천  2012   5
                    >> 인덱스가 자동으로 부여된다!
                       다양한 생성 방법이 있지만, 위의 경우엔 dict 자료형을 이용함
'''

# 인덱스열, 컬럼정보 지정
df1 = DataFrame(data, columns=['area','year','price'],
                index=['first','second','third','fourth','fifth'])
'''
print(df1)하면 XXXXXXX
       area year price
first   NaN  NaN   NaN
second  NaN  NaN   NaN
third   NaN  NaN   NaN
fourth  NaN  NaN   NaN
fifth   NaN  NaN   NaN
'''

df1 = DataFrame(data, columns=['지역','연도','가격'],
                index=['first','second','third','fourth','fifth'])
print(df1)
'''
        지역    연도  가격
first   서울  2010  10
second  서울  2011  15
third   서울  2012  20
fourth  인천  2011   3
fifth   인천  2012   5
'''

print("-------------------------------------------------")

# 객체의 컬럼 정보 확인
print(df1.columns)          # Index(['지역', '연도', '가격'], dtype='object')
print(type(df1.columns))    # <class 'pandas.core.indexes.base.Index'>

# 객체의 데이터 정보 확인
print(df1.values)
print(type(df1.values))
'''
[['서울' 2010 10]
 ['서울' 2011 15]
 ['서울' 2012 20]
 ['인천' 2011 3]
 ['인천' 2012 5]]
 <class 'numpy.ndarray'>
 '''

# 객체의 인덱스 정보 확인
print(df1.index)              # Index(['first', 'second', 'third', 'fourth', 'fifth'], dtype='object')
print(type(df1.index))        # <class 'pandas.core.indexes.base.Index'>


print("-------------------------------------------------")

# 데이터 프레임의 인덱싱 (컬럼 이름 Label로)

# 'DataFrame객체명.컬럼명' 을 통한 인덱싱
print(df1.연도)
print(type(df1.연도))
'''
first     2010
second    2011
third     2012
fourth    2011
fifth     2012
Name: 연도, dtype: int64
<class 'pandas.core.series.Series'>
'''

# 'DataFrame객체명[컬럼명]' 을 통한 인덱싱
print(df1['연도'])
print(type(df1['연도']))
'''
first     2010
second    2011
third     2012
fourth    2011
fifth     2012
Name: 연도, dtype: int64
<class 'pandas.core.series.Series'>                 시리즈 형태!
'''

print("-------------------------------------------------")

# 데이터 프레임의 인덱싱 (리스트 형태로)

print(df1[["연도","지역"]])     # 리스트를 통한 인덱싱
print(type(df1[["연도","지역"]]))
'''
          연도  지역
first   2010  서울
second  2011  서울
third   2012  서울
fourth  2011  인천
fifth   2012  인천
<class 'pandas.core.frame.DataFrame'>                 데이터프레임 형태!
'''


print("-------------------------------------------------")

df2 = DataFrame(data, columns=['지역','연도','가격','인구'],
                 index=['일빠','이빠','삼빠','사빠','오빠'])

print(df2)
print(type(df2))

'''    
지역    연도  가격   인구
일빠  서울  2010  10  NaN
이빠  서울  2011  15  NaN
삼빠  서울  2012  20  NaN
사빠  인천  2011   3  NaN
오빠  인천  2012   5  NaN
<class 'pandas.core.frame.DataFrame'>
                >> 인구는 data에 없으므로, NaN(Not a Number)이 출력
'''

df2.인구 = 1000       # 변경할 DataFrame의 컬럼에 대하여 인덱싱을 통한 값 지정
print(df2)
print(type(df2))
'''
    지역    연도  가격    인구
일빠  서울  2010  10  1000
이빠  서울  2011  15  1000
삼빠  서울  2012  20  1000
사빠  인천  2011   3  1000
오빠  인천  2012   5  1000
<class 'pandas.core.frame.DataFrame'>
'''

# 시리즈객체를 사용하여 데이터를 변경하기
#

    # 변경을 위한 인덱스와 변경 값을 Series 객체로 생성하여 지정한다.
val = Series([500,1500],index=['일빠','삼빠'])
    # DataFrame 객체의 컬럼 인덱스 지정을 통해 컬럼 값을 변경하다.
df2.인구 = val

print(df2)
print(type(df2))
'''
    지역    연도  가격      인구
일빠  서울  2010  10   500.0
이빠  서울  2011  15     NaN
삼빠  서울  2012  20  1500.0
사빠  인천  2011   3     NaN
오빠  인천  2012   5     NaN
<class 'pandas.core.frame.DataFrame'>
                                            >> '일빠' '삼빠'를 제외하곤 NaN ★★
'''


print("-------------------------------------------------")

# 컬럼 삭제
# - del '열의 인덱스'(삭제하고자 하는 DataFrame 객체의 열 지정)
#

print(df2)
print(type(df2))
'''
    지역    연도  가격      인구
일빠  서울  2010  10   500.0
이빠  서울  2011  15     NaN
삼빠  서울  2012  20  1500.0
사빠  인천  2011   3     NaN
오빠  인천  2012   5     NaN
<class 'pandas.core.frame.DataFrame'>
'''

del df2['인구']

print(df2)
print(type(df2))
'''
    지역    연도  가격
일빠  서울  2010  10
이빠  서울  2011  15
삼빠  서울  2012  20
사빠  인천  2011   3
오빠  인천  2012   5
<class 'pandas.core.frame.DataFrame'>
                >> '인구' column이 삭제되었다.
'''

print("-------------------------------------------------")

# 좌우 행렬 바꾸기 -> '전치 행렬'
#

print(df2.T)
print(type(df2.T))
'''
      일빠    이빠    삼빠    사빠    오빠
지역    서울    서울    서울    인천    인천
연도  2010  2011  2012  2011  2012
가격    10    15    20     3     5
<class 'pandas.core.frame.DataFrame'>
'''

print("-------------------------------------------------")

import pandas as pd
# DataFrame 객체 다루기 - count
#

    # None 값을 포함한 총 5개의 요소로 구성된 Series 객체
s = pd.Series([1,2,3,None,5])
print(s.count())                    # 4
                                    # >> None이 아닌 데이터 개수만 카운팅함
print(type(s.count()))              # <class 'numpy.int32'>
    # 2개의 열로 구성된 딕셔너리 객체 선언
data = {'연도':[2010,2011,2012,2011,2012],
        '가격':[10, 15, 20, None, 5]}

    # DataFrame 객체 생성
s1 = pd.DataFrame(data)
print(s1)
print(type(s1))
'''
     연도    가격
0  2010  10.0
1  2011  15.0
2  2012  20.0
3  2011   NaN
4  2012   5.0
<class 'pandas.core.frame.DataFrame'>
'''
print(s1.count())
print(type(s1.count()))
'''
연도    5
가격    4
dtype: int64
<class 'pandas.core.series.Series'>
'''


print("-------------------------------------------------")

import pandas as pd
# DataFrame 객체 다루기 - sum(열 합계)
#


    # 2개의 열로 구성된 딕셔너리 객체 선언
data = {'연도':[2010,2011,2012,2011,2012],
        '가격':[10, 15, 20, None, 5]}

    # DataFrame 객체 생성
s1 = pd.DataFrame(data)
print(s1)
print(type(s1))
'''
     연도    가격
0  2010  10.0
1  2011  15.0
2  2012  20.0
3  2011   NaN
4  2012   5.0
<class 'pandas.core.frame.DataFrame'>
'''
print(s1.sum())
print(type(s1.sum()))
'''
연도    10056.0
가격       50.0
dtype: float64
<class 'pandas.core.series.Series'>
'''

print("-------------------------------------------------")
import pandas as pd
# DataFrame 객체 다루기 - sum(행 합계)
#


    # 2개의 열로 구성된 딕셔너리 객체 선언
data = {'연도':[2010,2011,2012,2011,2012],
        '가격':[10, 15, 20, None, 5]}

    # DataFrame 객체 생성
s1 = pd.DataFrame(data)
s2 = s1.T
print(s2)
print(type(s2))
'''
         0       1       2       3       4
연도  2010.0  2011.0  2012.0  2011.0  2012.0
가격    10.0    15.0    20.0     NaN     5.0
<class 'pandas.core.frame.DataFrame'>
'''
print(s2.sum(axis=1))               # axis = 1 을 넣으면 행으로 더한다. ★★★
print(type(s1.sum(axis=1)))
'''
연도    10056.0
가격       50.0
dtype: float64
<class 'pandas.core.series.Series'>
'''

