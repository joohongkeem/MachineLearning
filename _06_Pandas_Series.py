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

# 시리즈(Series) 클래스
# - 1차원 배열과 비슷한 형태이지만, 각 데이터에 대한 인덱스(Index) 정보가 붙어 있는 형태의 자료
#

from pandas import Series

# 1차원 배열 형태의 시리즈 객체 생성
house_price = Series([10,20,30,40,50])

# Series 객체의 데이터 확인
print(house_price)
        # 0    10
        # 1    20
        # 2    30
        # 3    40
        # 4    50
        # dtype: int64

# 인덱싱을 통한 Series 객체의 데이터 확인
print(house_price[1])                       # 20

print("-------------------------------------------------")

# 인덱스 정보와 같이 생성하기 ★
house_price = Series([10,20,30,40,50],
                     index = ["강원","인천","전라","제주","서울"])

print(house_price)
print(type(house_price))
        # 강원    10
        # 인천    20
        # 전라    30
        # 제주    40
        # 서울    50
        # dtype: int64
        # <class 'pandas.core.series.Series'>

print(house_price["서울"])                 # 50
print(type(house_price["서울"]))           # <class 'numpy.int64'>


# index 속성, value 속성 ★
print(house_price.index, type(house_price.index))
        # Index(['강원', '인천', '전라', '제주', '서울'], dtype='object') <class 'pandas.core.indexes.base.Index'>
print(house_price.values, type(house_price.values))
        # [10 20 30 40 50] <class 'numpy.ndarray'>


print("-------------------------------------------------")

# 시리즈(Series) 클래스의 연산
# - Series 객체의 각 데이터 요소에 대한 산술 연산을 모두 지원함
#

print(house_price + 5)
'''
강원    15
인천    25
전라    35
제주    45
서울    55
dtype: int64
'''
print(house_price - 5)
'''
강원     5
인천    15
전라    25
제주    35
서울    45
dtype: int64
'''

print(house_price * 5)
'''
강원     50
인천    100
전라    150
제주    200
서울    250
dtype: int64
'''
print(house_price / 5)
'''
강원     2.0
인천     4.0
전라     6.0
제주     8.0
서울    10.0
dtype: float64
'''

print("-------------------------------------------------")

# 시리즈(Series) 클래스의 인덱싱
#

print(house_price[[0,3]])
'''
강원    10
제주    40
dtype: int64
    >> house_price[0] 과 house_price[3] 출력
'''
# 시리즈(Series) 클래스의 슬라이싱
#
print(house_price[0:3])
'''
강원    10
인천    20
전라    30
dtype: int64
'''