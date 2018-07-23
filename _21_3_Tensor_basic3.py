# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf

# 세션을 실행할 때 순서에 따라 변수가 어떻게 변하는지 확인

value = tf.Variable(0)
one = tf.constant(1)

state = tf.add(value, one)
update = tf.assign(value, state) # state 값을 value에 할당하고
                                 # 할당된 값을 리턴하기 위한 함수

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for _ in range(3):
    print('---------------------------------------')
    print("update :",sess.run(update), "state :",sess.run(state))
'''
---------------------------------------
update : 1 state : 2
---------------------------------------
update : 2 state : 3
---------------------------------------
update : 3 state : 4
'''
    #print("state :",sess.run(state), 'update :',sess.run(update))
'''
---------------------------------------
state : 1 update : 1
---------------------------------------
state : 2 update : 2
---------------------------------------
state : 3 update : 3
'''

            # 왜 출력값이 다른지 잘 생각해보자 ! 텐서플로우는 그래프의 흐름이 중요하므로 순서가 가장 중요하다!!ㅉ


sess.close()