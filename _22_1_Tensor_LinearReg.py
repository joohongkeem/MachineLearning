# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf

# 데이터의 갯수를 바꿔보고, 학습률도 바꿔보고 , epoch를 바꿔보며 결과를 확인해보자
# 1. 데이터의 갯수가 많으면 학습률이 작게, epoch를 크게해야한다 ! (발산 방지)

X_train = [1,2,3,4,5,6,7,8,9,10]
y_train = [1,2,3,4,5,6,7,8,9,10]

# 가중치
# 균등 분포로 랜덤하게 숫자를 만들기 위함
w = tf.Variable(tf.random_uniform([1],-1,1))    # random_uniform(shape, min, max)
b = tf.Variable(tf.random_uniform([1],-1,1))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))          # 값을 찍어보면 [-1과 1 사이의 아무런 숫자]
print(sess.run(b))          # 값을 찍어보면 [-1과 1 사이의 아무런 숫자]
sess.close()

x_test = [17,38]

# hypothesis 설정
# hypothesis = tf.add(tf.multiply(x,w), b)와 같은 의미
hypothesis = w * X_train + b

# 평균 제곱 오차를 구한다.
cost = tf.reduce_mean((hypothesis - y_train)**2)    #reduce_mean(~~~) : ~~~의 값의 평균을 구해준다

# 옵티마이저를 지정한다 -> 학습방법의 지정!!
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    # 경사하강법을 사용하자!
    # - 학습률 : 보폭 !!
    #   >> 기울기가 2.5 이고 학습률이 0.01이면 이전 지점으로부터 0.025 떨어진 지점으로 !

# 학습을 시킨다
train = optimizer.minimize(loss=cost)
print('train :',train)                  # train : name: "GradientDescent"
                                          # op: "NoOp"
                                          # input: "^GradientDescent/update_Variable/ApplyGradientDescent"
                                          # input: "^GradientDescent/update_Variable_1/ApplyGradientDescent"
print('type(train) :',type(train))     # type(train) : <class 'tensorflow.python.framework.ops.Operation'>


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# epoch를 100번 수행
for i in range(1000):
    sess.run(train)
    print(' # {}번째 학습 ------------------------------'.format(i+1))
    print('   w = {}\n   b = {}\n   cost = {}'.format(sess.run(w),sess.run(b),sess.run(cost)))

for i in range(len(x_test)):
    print("x =", x_test[i],"predict =",sess.run(w*x_test[i] + b))
sess.close()