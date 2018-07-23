# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

# placeholder를 이용해서 구구단을 만들어보자.
#

import tensorflow as tf

n = tf.placeholder(tf.int32)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


for i in range(1, 10, 1):
    print(i,'단 ->',sess.run((n)*(i),feed_dict={n:[1,2,3,4,5,6,7,8,9]}))


sess.close()