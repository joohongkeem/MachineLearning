# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf

x1 = [1., 0., 3., 0., 5.]           # 공부시간
x2 = [0., 2., 0., 4., 0.]           # 출석시간
y = [1., 2., 3., 4., 5.]            # 시험점수



w1 = tf.Variable(tf.random_uniform([1], -1, 1))
w2 = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

# hypothesis 작성
hypothesis = w1 * x1 + w2 * x2 + b
    # w1(1,1) * x1(1,5) -> (1, 5)
    # w2(1,1) * x2(1,5) -> (1, 5)
    # -> 즉 hypothesis는 (1, 5)의 형태

# cost
cost = tf.reduce_mean((hypothesis - y)**2)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)

# train
train = optimizer.minimize(loss = cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("init w1 :",sess.run(w1),"\ninit w2 :",sess.run(w2))

for i in range(1000) :
    sess.run(train)
    print(i,'- cost)', sess.run(cost))

print("Hypothesis ->",sess.run(hypothesis),'')
print('   w1 :',sess.run(w1))
print('   w2 :',sess.run(w2))
print('    b :',sess.run(b))
print(' cost :',sess.run(cost))

X_test = [0,4]

ww1, ww2, bb = sess.run(w1), sess.run(w1), sess.run(b)

print("Predict -> ",ww1*X_test[0] + ww2*X_test[1] + bb,'')