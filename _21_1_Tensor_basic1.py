# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf

# • Tensorflow
# − 구글에서 머신 러닝 분석 및 시각화를 위해 개발한 오픈소스 라이브러리
# − 텐서플로우(TensorFlow™)는 데이터 플로우 그래프(Data flow graph)를 사용하여 수
#   치 연산을 하는 오픈소스 소프트웨어 라이브러리
# − 그래프의 노드(Node)는 수치 연산을 나타내고 엣지(edge)는 노드 사이를 이동하는
#   다차원 데이터 배열(텐서,tensor)를 나타냄
# − 유연한 아키텍처로 구성되어 있어 코드 수정없이 데스크탑, 서버 혹은 모바일 디바
#   이스에서 CPU나 GPU를 사용하여 연산을 구동시킬 수 있음
# − 텐서플로우는 원래 머신러닝과 딥 뉴럴 네트워크 연구를 목적으로 구글의 인공지능
#   연구 조직인 구글 브레인 팀의 연구자와 엔지니어들에 의해 개발됨
#
# -> 그래프의 개념이 중요하며, + Tensor라는 객체
#
# • Tensors
# - Tensorflow의 프로그램에서 사용하는 모든 데이터의 구조
# - op의 연산을 위해서는 작업(op)간에는 tensor만 주고 받을 수 있음
# - n차원 배열이나 리스트의 형태임

a = tf.constant(3)
b = tf.Variable(5)

add = tf.add(a,b)


print('a :',a)              # a : Tensor("Const:0", shape=(), dtype=int32)
print('b :',b)              # b : <tf.Variable 'Variable:0' shape=() dtype=int32_ref>
print(add)                   # Tensor("Add:0", shape=(), dtype=int32)

                             # -> 값이 나오지않고 객체가 나왔다!!


# Tensorflow의 수행 구조
#
# 1. 변수의 선언 및 값 설정
#    -> X = tf.Variable([1,2,3], dtype=tf.float32)
#
# 2. 세션(Session)의 사용
# - 세션 : 텐서플로우에서 연사을 수행하기 위한 실행 환경
#       - Session() : 세션 객체를 생성하기 위한 생성자
#       - run() : 인자로 주어진 변수에 대해 계산을 수행한다.
#       - close() : 세션에서 사용한 resource를 반환하기 위한 함수, with로 대체 가능
#

print('\n# Session\n#')


sess = tf.Session()

# 초기화 (전체 변수 초기화)
sess.run(tf.global_variables_initializer())
    # 초기화 하지 않으면 constatn(상수) 는 출력이 되지만
    # 변수(Variable)은 출력이 되지 않는다 !!!


print('a :', sess.run(a))           # a : 3
print('b :', sess.run(b))           # b : 5
print('add :', sess.run(add))       # add : 8


# 반드시 close()를 해줘야 한다.
sess.close()