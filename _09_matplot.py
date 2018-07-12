# Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180711","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# 맷플롯 패키지
# - 데이터 시각화를 통해 데이터 분석을 지원하는 라이브러리
# - 유용한 수학과 그래프를 지원하는 파이썬 라이브러리
#   >> 이를 통해 데이터를 다루고, 시각화 하기 쉬워짐
#

print("-------------------------------------------------")


# 예제 1. Matplot 을 이용한 단순한 그래프 작성하기
#       - matplotlib의 pyplot 모듈을 사용해야 한다.
#       - 간단하게 1, 4, 9, 16, 25, 36 을 입력값으로 하는 그래프를 그려보도록 한다.
#

    # matplotlib의 pyplot 모듈을 불러온다.
import matplotlib.pyplot as plt

    # 그래프를 그리기 위한 데이터
inputValue = [1,4,9,16,25,36]

    # plt.plot() 함수를 통한 입력값 전달
plt.plot(inputValue)

    # 그래프를 화면에 호출
plt.show()


print("-------------------------------------------------")

# 예제 2. Matplot 을 이용한 단순한 그래프 작성하기
#       - Plot 함수에 x, y축 값을 모두 전달하여 작성하기
#

    # matplotlib의 pyplot 모듈을 불러온다.
import matplotlib.pyplot as plt


    # 그래프를 그리기 위한 데이터 - x좌표
inputValue = [1,2,3,4,5,6]
    # 그래프를 그리기 위한 데이터 - y좌표
outputValue = [1,4,9,16,25,36]

    # plt.plot() 함수에 x, y 좌표 값 전달
plt.plot(inputValue,outputValue)

    # 그래프를 화면에 호출
plt.show()


print("-------------------------------------------------")

# 예제 3. Matplot 을 이용한 단순한 그래프 작성하기 - 그래프 꾸미기
#       - 도형의 굵기, x축, y축 라벨등을 정할 수 있다.
#

    # matplotlib의 pyplot 모듈을 불러온다.
import matplotlib.pyplot as plt


inputValue = [1,2,3,4,5,6]
outputValue = [1,4,9,16,25,36]

    # 그래프 꾸미기
plt.plot(inputValue,outputValue,linewidth = 5)      # 그래프 라인 두께 변경
plt.title("Get Square", fontsize= 20)             # 그래프 타이틀 설정
plt.xlabel("Input Values",fontsize = 15)          # 그래프 x축 라벨 설정
plt.ylabel("Square Values",fontsize = 15)         # 그래프 y축 라벨 설정
plt.tick_params(axis=1, labelsize = 15)             # 그래프 x, y축 눈금 레이블 표시 값 크기 변경

    # 그래프를 화면에 호출
plt.show()

print("-------------------------------------------------")

# 예제 4. Matplot 을 이용한 단순한 그래프 작성하기 - 점 그리기
#       - 도형의 굵기, x축, y축 라벨등을 정할 수 있다.
#

    # matplotlib의 pyplot 모듈을 불러온다.
import matplotlib.pyplot as plt


inputValue = [1,2,3,4,5,6]
outputValue = [1,4,9,16,25,36]

    # x와 y를 입력 값으로 점 그리기
plt.scatter(inputValue,outputValue)

    # 그래프를 화면에 호출
plt.show()


print("-------------------------------------------------")

# Matplot을 이용한 이미지 그리기
# - Matplot의 pyplot에는 이미지관련 함수인 imshow() 를 이용하여, 이미지를 그려본다.
# - 오류 발생 시 Image 패키지의 설치가 필요함
#

from matplotlib.image import imread

img = imread('./test.jpg')
plt.imshow(img)
plt.show()


