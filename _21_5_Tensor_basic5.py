# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf

# 원하는 구구단의 숫자를 입력받아 구구단을 완성
#

def nine_1(dan):
    left = tf.placeholder(tf.int32)
    right = tf.placeholder(tf.int32)

    multi = tf.multiply(left, right)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('# {}단'.format(dan))
    for i in range (1, 10):
        result= sess.run(multi, feed_dict={left:dan, right : i})
        print('{} x {} = {}'.format(dan,i, result))
    sess.close()


#nine_1(9)


# left를 인자로 받아 계산

def nine_2(dan):
    right = tf.placeholder(tf.int32)                # datatype을 잘맞춰주자!
    multi = tf.multiply(dan, right)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('# {}단'.format(dan))
    for i in range(1,10):
        result = sess.run(multi,feed_dict={right:i})
        print('{} x {} = {}'.format(dan,i, result))
    sess.close()

nine_2(8)

