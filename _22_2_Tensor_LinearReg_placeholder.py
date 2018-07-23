# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

import tensorflow as tf


X_train = [1, 2, 3]
y_train = [1, 2, 3]

# 가중치
# 균등 분포로 랜덤하게 숫자를 만들기 위함
w = tf.Variable(tf.random_uniform([1],-1,1))    # random_uniform(shape, min, max)
b = tf.Variable(tf.random_uniform([1],-1,1))

x = tf.placeholder(tf.float32)

X_test = [5,7]

# hypothesis 설정
# hypothesis = tf.add(tf.multiply(x,w), b)와 같은 의미
hypothesis = w * x+ b

# 평균 제곱 오차를 구한다.
cost = tf.reduce_mean((hypothesis - y_train)**2)    #reduce_mean(~~~) : ~~~의 값의 평균을 구해준다

# 옵티마이저를 지정한다 -> 학습방법의 지정!!
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
    # 경사하강법을 사용하자!
    # - 학습률 : 보폭 !!
    #   >> 기울기가 2.5 이고 학습률이 0.01이면 이전 지점으로부터 0.025 떨어진 지점으로 !

# 학습을 시킨다
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# epoch를 100번 수행
for i in range(1000):
    sess.run(train,feed_dict={x:X_train})
    print(' # {}번째 학습 ------------------------------'.format(i+1))
#    print('   w = {}\n   b = {}\n   cost = {}'.format(sess.run(w),sess.run(b),sess.run(cost)))
        # -> 에러 발생!
    print('w :',sess.run(w, feed_dict={x:X_train}))
    print('b :', sess.run(b, feed_dict={x: X_train}))
    print('cost :', sess.run(cost, feed_dict={x: X_train}))

for i in range(len(X_test)):
    print("x =", X_test[i],"predict =",sess.run(w*X_test[i] + b, feed_dict = {x:X_test}))

sess.close()