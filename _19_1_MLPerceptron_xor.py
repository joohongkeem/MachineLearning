# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180718", "Made by joohongkeem#".rjust(38), sep=' ', end='\n')
print("-------------------------------------------------")


# Perceptron으로 해결할 수 없었던 XOR연산을 Multi Layer Perceptron을 통해 구현해보자!

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns


def plot_mlp(ppn):
    # plt.figure(figsize=(12, 8), dpi=60)
    #    model = Perceptron(n_iter=10, eta0=0.1, random_state=1).fit(X, y)
    model = ppn
    XX_min = X[:, 0].min() - 1;
    XX_max = X[:, 0].max() + 1;
    YY_min = X[:, 1].min() - 1;
    YY_max = X[:, 1].max() + 1;
    XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
    ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    cmap = matplotlib.colors.ListedColormap(sns.color_palette("Set3"))
    plt.contourf(XX, YY, ZZ, cmap=cmap)
    plt.scatter(x=X[y == 0, 0], y=X[y == 0, 1], s=200, linewidth=2, edgecolor='k', c='y', marker='^', label='0')
    plt.scatter(x=X[y == 1, 0], y=X[y == 1, 1], s=200, linewidth=2, edgecolor='k', c='r', marker='s', label='1')
    plt.xlim(XX_min, XX_max)
    plt.ylim(YY_min, YY_max)
    plt.grid(False)
    plt.legend()


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # Labels
print('X.shape =', X.shape)     #(4, 2)

y = np.array([0, 1, 1, 0])        #(4, )
print('y.shape =', y.shape)

X_test = np.array([[1,0],[1,0],[0,0],[0,1],[0,0],[1,1],[0,0],[0,1],[1,1],[0,0],[0,1],[1,1],[1,0],[0,0]])

    # ★★
    # 입력노드가 2개 -> x1, x2
    # hidden layer의 노드가 4, 2, 4, 2, 4개
    # 출력노드가 1개 -> y
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[4,2,4,2,4]).fit(X,y)
    # 각 인자가 무엇을 의미하는지 꼭 찾아보기 ★★
    # solver : 알고리즘 {‘lbfgs’, ‘sgd’, ‘adam’}
    # random_state : 랜덤 상태
    # activation : 적용함수 ex) 'relu'
print(type(mlp.hidden_layer_sizes))                             # <class 'list'>
    # hidden_layer_sizes=[100] : 은닉층 수 (default = (100,))
    # hidden_layes_sizes=[10,10] : 10개짜리 은닉층 2개

    # 그래프 출력
plot_mlp(mlp)


print("학습 결과 :", mlp.predict(X_test) )

    # 입력 층 +  출력층 +  히든층 :
    # hidden_layer_sizes=[4,2,4,2,4]이므로 입력층(1) + 출력층(1) + 히든충(5) = 7 출력
print("신경망 깊이 :", mlp.n_layers_)

print('mlp.coefs_ :',mlp.coefs_)
'''
mlp.coefs_ : 
   [array([[ 1.25767365,  7.4081325 , -0.48508712,  1.78697461], [ 0.33259654, 10.19443487, -0.27299191,  1.45265875]]), 
    array([[-0.53776663,  0.11713066], [-3.9529173 , -5.74768107], [-1.09274618,  0.34271499], [ 0.80019629,  1.13074517]]), 
    array([[-0.07119706,  4.75979116, -0.70553714, -0.83929002], [-0.65918429,  4.13214628,  0.0403819 , -2.42530894]]), 
    array([[-0.88941294,  0.21742377], [16.06765559,  0.44862478], [ 0.82017232,  0.33605549], [-0.1384039 , -0.11638281]]), 
    array([[10.16792029,  0.56547549, -0.53529518, -0.68584946], [-0.32786492, -0.29184184,  0.12974354, -0.11348182]]), 
    array([[ 0.01505998], [-0.19245845], [-0.06821035], [-0.51745979]])
   ]
   
   현재 구조는
   입력    히든    히든    히든   히든   히든   출력
            ㅇ1            ㅇ            ㅇ
    ㅇ      ㅇ2     ㅁ1     ㅇ     ㅇ      ㅇ
    ㅇ      ㅇ3     ㅁ2     ㅇ     ㅇ      ㅇ    ㅇ
            ㅇ4             ㅇ             ㅇ
    
    ㅇ1 -> ㅁ1 = -0.53776663
    ㅇ2 -> ㅁ1 = -3.9529173
    ㅇ3 -> ㅁ1 = -1.09274618
    ㅇ4 -> ㅁ1 = 0.80019629
'''
    # MLP의 계층별 가중치 확인
print('len(mlp.coefs_) :', len(mlp.coefs_))

print("mlp.n_outputs_ :", mlp.n_outputs_)
print("mlp.classes_ :", mlp.classes_)
#for i in range(len([mlp.coefs_])):
i = 0
while i<len(mlp.coefs_):
    print('----------------------------------------')
    number_neurons_in_layer = mlp.coefs_[i].shape[1]
    print("number_neurons_in_layer :", number_neurons_in_layer, ' \ni : ', i)
    for j in range(number_neurons_in_layer):
        weights = mlp.coefs_[i][:, j]
        print(i, j, weights, end=", ")
        print()
    print()
    i+=1
plt.show()