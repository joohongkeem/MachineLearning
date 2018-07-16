# -*- coding: utf-8 -*-
#  Run module(Hot key : F5)
print("-------------------------------------------------")
print("# 20180716","Made by joohongkeem#".rjust(38),sep=' ',end='\n')
print("-------------------------------------------------")

# 1. 초기 k은 데이터 오브젝트 중에서 무작위로 뽑힌다.
# 2. 각 k중에 가까이 있는 평균값을 기준으로 묶인다.
# 3. k개의 클러스터의 중심점을 기준으로 평균값이 재조정된다.
# 4. 중심점이 이동하지 않을 때 까지 2~3 과정을 반복한다.
#


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score

X_train = np.array([[10, 2],[5, 8],[7, 2], [4, 7],[3, 9],[2, 9],
              [0, 8], [1, 7],[9, 3], [7, 4],[6, 2],[4, 6],
              [1, 9], [2, 7],[10, 3],[8, 4],[9, 1],[5, 3],[11,3],[9,2]
              ])
    # 목표 값이 없으므로 y_train이 존재하지 않는다.
    #   >> clustering 은 '비지도 학습'
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], s=100, label='input data')  # X1 값 , X2 값 선정
init_point = np.array([[0,0],[10,10]])
plt.scatter(init_point[0, 0], init_point[0, 1], label='init point 1', c='r', s=100)
plt.scatter(init_point[1, 0], init_point[1, 1], label='init point 2', c='b', s=100)
plt.legend()

#plt.show()

#epoche = 100
#plt.figure()
i = 1
while  i <= 5 :
    model = KMeans(n_clusters=2, init=init_point, n_init=1, max_iter=i, random_state=0) # 모델 설정
            # init= 값을 따로 지정해주지 않으면, 알아서 random으로 지정해준다.
            #  But, 다른 방식을 사용해서 지정할 수도 있다!
            #
            # max_iter 항목을 없애고 while문도 없앤다면
            # 알아서 centroid가 움직이지 않을때까지 clustering을 한다. default = 300
            #


    model.fit(X_train)  # 학습

    c0, c1 = model.cluster_centers_  # 중심점 찾기
    print("c0 ", c0, "C1", c1)
    print(model.labels_)
    plt.figure()
#    plt.subplot(1, 4, i)
    plt.scatter(X_train[model.labels_==0,0], X_train[model.labels_==0,1], s=100, marker='v', c='r', label='class 0')
    plt.scatter(X_train[model.labels_==1,0], X_train[model.labels_==1,1], s=100, marker='^', c='b', label='class 1')
    plt.scatter(c0[0], c0[1], s=100, c="r", label='Centriod 0')
    plt.scatter(c1[0], c1[1], s=100, c="b", label='Centriod 1')
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.title("{4} Loop \n Centriod 0 [{0} , {1}]\n Centriod 1 [{2}, {3}]".format(c0[0],c0[1], c1[0], c1[1], i))
    plt.legend()
#    plt.show()
    i += 1
plt.show()

def kmeans_check_data(c0, c1):
    df = pd.DataFrame(np.hstack([X_train,
                      np.linalg.norm(X_train - c0, axis=1)[:, np.newaxis],
                                np.linalg.norm(X_train - c1, axis=1)[:, np.newaxis],
                                model.labels_[:, np.newaxis]]),
                    columns=["x0", "x1", "distance 0", "distance 1", "cluster"])
    return df

result = kmeans_check_data(c0, c1)
print(result)



#print(np.linalg.norm(X - c0, axis=1)[:, None])  # 가로를 세로로 세

print(model.score(X_train))

print("silhouette_score :", silhouette_score(X_train, model.labels_,
                                      metric='euclidean',
                                      sample_size=len(X_train)))