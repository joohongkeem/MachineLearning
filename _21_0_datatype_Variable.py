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


# 변수 텐서 하나를 만들어보자!
#
# - y = W*x + b 라는 학습용 가설이 있을 때, W와 b는 학습을 통해서 구해야 하는 변수!!
#

g2 = tf.Graph()

with g2.as_default():
    v1 = tf.Variable(initial_value=10, name="v1")


print(type(v1))                     # <class 'tensorflow.python.ops.variables.Variable'>

print(v1)                           # <tf.Variable 'v1:0' shape=() dtype=int32_ref>

print(v1.op)

'''
name: "v1"
op: "VariableV2"
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "shape"
  value {
    shape {
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}


'''


print(g2.as_graph_def())