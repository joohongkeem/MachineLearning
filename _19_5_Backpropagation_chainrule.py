# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180718", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")



class Add:
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class Mul:
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x


def apple_graph():
    apple_price = 100
    apple_count = 2
    tax = 1.1

    layer_apple = Mul()
    layer_tax = Mul()

    # 순전파
    apple_total = layer_apple.forward(apple_price, apple_count)
    total = layer_tax.forward(apple_total, tax)
    print('[Forward]\n','total =', total)

    # 역전파    오류에 의한 영향
    d_total = 1.0
    # 세금 계산 층에서  갯수에 대한 영향, 세금에 대한 영향
    d_apple_total, d_tax = layer_tax.backward(d_total)
    #  사과 계산 계층에서의 사고 갯수에 대한 영향과 세금에 대한 영향
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
    print('[Backward]\n', 'd_apple_price ={0} \nd_apple_count={1}\nd_tax={2}'.format(d_apple_price, d_apple_count, d_tax))


def fruit_graph():
    apple_price = 100
    apple_count = 2
    mango_price = 150
    mango_count = 3
    tax = 1.1

    layer_apple = Mul()
    layer_mango = Mul()
    layer_fruit = Add()
    layer_tax = Mul()

    # 순전파
    # 사과 계산 층에서의 갯수에 따른 계산
    apple_total = layer_apple.forward(apple_price, apple_count)
    mango_total = layer_mango.forward(mango_price, mango_count)
    # 사과와 망고의 계산 합 층
    fruit_total = layer_fruit.forward(apple_total, mango_total)
    # 세금 계산 충
    total = layer_tax.forward(fruit_total, tax)
    print('[Forward]\n','total =', total)


    # 역전파
    # 1개 변화했을때의 영향
    d_total = 1.0
    d_fruit_total, d_tax = layer_tax.backward(d_total)

    d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
    d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)

    print('      backward :', "d_fruit_total [",d_fruit_total,"] d_tax [", d_tax,"]")
    print('fruit backward :', "d_apple_total [",d_apple_total,"] d_mango_total [", d_mango_total,"]")
    print('apple backward :', "d_apple_price [",d_apple_price,"] d_apple_count [", d_apple_count,"] d_tax[", d_tax, "]")
    print('Mango backward :', "d_mango_price [", d_mango_price, "] d_mango_count [", d_mango_count,"]")


# 오류 역전파에 의해 주어진 목표 가격에 대한 과일 갯수를 계산해 보자
# -> 사과를 2개 갖고 있고, 망고를 3개 갖고 있고, tax=1.1 일 때,
#    target(720)원을 벌고싶으면 사과와 망고의 가격을 어느정도로 정해야 하는가!
#
# -> 편미분을 하나도 사용하지 않고 역전파를 구현했다 !! ★★
def back_propagation():
    apple_count = 2
    mango_count = 3
    tax = 1.1

    target = 720  # 목표치...

    # weights       (가격)
    apple_price = 100
    mango_price = 150

    layer_apple = Mul()
    layer_mango = Mul()
    layer_fruit = Add()
    layer_tax = Mul()

    for i in range(10000):
        # 순전파
        apple_total = layer_apple.forward(apple_price, apple_count)
        mango_total = layer_mango.forward(mango_price, mango_count)
        fruit_total = layer_fruit.forward(apple_total, mango_total)
        total = layer_tax.forward(fruit_total, tax)
        print('foward :', total)


        # 역전파
        #d_total = 1.0
        d_total = total - target  # 목표 값과의 차이
        print('backward :\nd_total({0}) = total({1}) - target({2})'.format(d_total, total, target))
        d_fruit_total, d_tax = layer_tax.backward(d_total)

        d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
        d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
        d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)

        apple_price -= 0.1 * d_apple_price
        mango_price -= 0.1 * d_mango_price

        print('{} : apple_price = {:.2f} & mango_price = {:.2f}'.format(i, apple_price, mango_price))


apple_graph()
'''
[Forward]
 total = 220.00000000000003
[Backward]
 d_apple_price =2.2 
 d_apple_count=110.00000000000001
 d_tax=200.0
 
 --> 영향도를 계산하는 것이므로 숫자가 저렇게 나오는것이 맞다!
'''


fruit_graph()
'''
[Forward]
 total = 715.0000000000001
[Backward]
      backward : d_fruit_total [ 1.1 ] d_tax [ 650.0 ]
fruit backward : d_apple_total [ 1.1 ] d_mango_total [ 1.1 ]
apple backward : d_apple_price [ 2.2 ] d_apple_count [ 110.00000000000001 ] d_tax[ 650.0 ]
Mango backward : d_mango_price [ 3.3000000000000003 ] d_mango_count [ 165.0 ]

 --> 영향도를 계산하는 것이므로 숫자가 저렇게 나오는것이 맞다!
'''
back_propagation()
'''
영향도 계산이 아닌 실제 가격으로 계산!
잘 읽어보면 위의 두개함수는 d_total = 1.0 으로 두어서 영향도를 확인하지만,
아래의 함수는 target=720이라는 목표치를 준다.'''