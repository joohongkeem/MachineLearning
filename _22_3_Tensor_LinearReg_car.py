# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf
import numpy as np


cars = np.loadtxt('./_22_3_cars.csv', delimiter=',', unpack=True)

X_train = cars[0]
y_train = cars[1]

w = tf.Variable(tf.random_uniform([1],-1,1))
b = tf.Variable(tf.random_uniform([1],-1,1))

hypothesis = w * X_train + b
cost = tf.reduce_mean((hypothesis - y_train) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
train = optimizer.minimize(loss = cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(train)

ww, bb = sess.run(w), sess.run(b)
print('w :', ww, 'b :', bb, 'cost :',sess.run(cost))

sess.close()