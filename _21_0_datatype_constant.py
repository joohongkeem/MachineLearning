# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180719", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")

# - 텐서플로우TensorFlow의 기본 데이터 구조인 텐서Tensor는 보통 다차원 배열이라고 말합니다.
# - 텐서플로우에는 세 가지의 핵심 데이터 구조인 상수Constant, 변수Variable, 플레이스홀더Placeholder가 있습니다.
# - 텐서플로우의 텐서를 다차원 배열로 많이 설명하지만 이는 맞기도 하고 틀리기도 합니다.
#   이로 인해 다소 오해도 발생합니다.
#   C++ API에서 말하는 텐서는 다차원 배열에 가깝습니다. 메모리를 할당하고 데이터 구조를 직접 챙깁니다.
#   하지만 파이썬 API 입장에서 텐서는 메모리를 할당하거나 어떤 값을 가지고 있지 않으며
#   계산 그래프의 연산(Operation) 노드(Node)를 가리키는 객체에 가깝습니다.
#   우리가 주로 다루는 파이썬 API의 텐서는 넘파이(NumPy)의 다차원 배열 보다는
#   어떤 함수를 의미하는 수학 분야의 텐서에 더 비슷합니다.

import tensorflow as tf


# 상수 텐서 하나를 만들어보자!

g1 = tf.Graph()

with g1.as_default():
    c1 = tf.constant(1, name="c1")
    # tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
    #       - value : 상수의 값
    #       - dtype : 상수의 데이터형.
    #       - shape : 행렬의 차원을 정의, shape=[3,3]으로 정의하면 이 상수는 3x3 행렬을 저장하게 된다.
    #       - name : 상수의 이름을 정의한다.
    #
    # 즉, c1은 "c1"이란 이름을 가지고 정수값 1을 가진 상태이다

print(type(c1))     # tensorflow.python.framework.ops.Tensor
                    # >> c1 은 tensorflow.python.framework.ops 밑에 있는 Tensor 클래스의 객체이다

print(c1)           # Tensor("c1:0", shape=(), dtype=int32)
                    # >> 텐서플로우가 그래프를 만들고, 실행하는 두 단계 구조를 갖고 있으므로
                    #    c1을 출력하면 상수값 1이 출력되는 것이 아니고,
                    #    이 상수가 텐서라는 것과, 이름, 크기(여기선 스칼라), 데이터 타입이 출력

print(c1.op)        # 파이썬의 텐서는 연산 노드를 가리킨다고 했는데, c1 텐서는 어떨까?
'''
name: "c1"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 1
    }
  }
}

- 텐서의 op 속성에는 그 텐서에 해당하는 연산이 할당되어 있다.
  >> 연산의 타입은 Const이고, 이름은 "c1" 이구나!
- 속성으로 int_val : 1인 텐서를 가지고 있다!
  >> 이 텐서는 tf.constant로 만들었을 때, 지정한 상수값 1을 가지고 있다!!
  >> But, c1은 tf.Tensor 타입인데 그 안에 왜 tensor의 값이 또 있을까?
  >> c1 텐서와 c1.op의 노드 속성에 있는 텐서가 다른 위치에서 같은 이름을 사용하고 있다!
  >> c1이라는 텐서는 연산 Const 노드를 가리키는 포인터와 같다!
'''


a = tf.constant(1)
b = tf.constant(10)
c = tf.constant(5)

d = a*b+c               # 이 때, d = a*b+c 는 계산을 수행하는 것이 아니라,
                        # a-> [*]   ->  [+]
                        # b->          c->      라는 그래프를 정의하는 것!
                        #
                        # -> 실제 값을 뽑아내려면, 이 정의된 그래프에 값을 넣어서 실행하는데
                        #    이 때, 세션(Session)을 생성하여 그래프를 실행해야 한다!!!
                        #    세션은 그래프를 인자로 받아서 실행을 해주는 일종의 러너(Runner) ★★★

print(d)                # Tensor("add:0", shape=(), dtype=int32)

sess = tf.Session()
result = sess.run(d)
print(result)           # 15