# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf


# 그래프 작성
a = tf.placeholder(tf.float32)
print(a, type(a))
b = tf.placeholder(tf.float32)
print(b, type(b))
c = tf.placeholder(tf.float32)
print(c, type(c))
add = tf.add(a,b)
print(add, type(add))


# 실행부
sess = tf.Session()
    # 초기화
sess.run(tf.global_variables_initializer())

    # 세션 실행 시 feed를 통해 add함수 수행
    #
    # Feeds
    # - graph의 tensor들 (상수, 변수) 에 값을 전달하는 메커니즘
    # - Feed 데이터는 run()으로 전달되어 사용됨
    # - tf.placeholder()를 사용하여 특정 작업을 feed 작업으로 지정함
    #

print(sess.run(add, feed_dict={a:4, b:5}))      # 9.0 출력
#print(sess.run(add))                           # 이렇게만 치면 에러 발생(값을 모른다)
