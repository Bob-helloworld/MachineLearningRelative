import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2


# 从文件中读取数据
def read_data(path):
    data = pd.read_table(path, header=None, names=['x', 'y'])
    print("read data:" + str(data))
    # 没有表头，read_table去Tab
    x = data['x']
    y = data['y']
    # plt.subplot(2,1,1)
    plt.scatter(x, y)
    return data


# 计算每个点距离各类别中心点的距离
def distance(data, centers):
    #     data: 80x2, centers: kx2
    dist = np.zeros((data.shape[0], centers.shape[0]))  # 出来的是80*4，即每个点相对4个质心的距离
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i, j] = np.sqrt(np.sum((data.iloc[i, :] - centers[j]) ** 2))  # 共80个样本，每个样本的在两个特征维度上，
            # 分别对k个质心求距离然后求和，类似矩阵乘法，
            # 所以距离矩阵为80x4
    return dist


# 计算每个点离哪个质心最近，返回对应质心的标签
def near_center(data, centers):
    dist = distance(data, centers)
    # 得到的dist行为80个点，列为每个点到4个质心的距离。然后取最小距离，得到对应质心的label。
    near_cen = np.argmin(dist, 1)
    return near_cen, dist


# 使用kmeans聚类
def kmeans(data, k):
    # step 1: init. centers
    seed = int(time.time())
    np.random.seed(seed)
    random_centers = data.iloc[np.random.choice(data.shape[0], size=k, replace=False), :]  # 随机产生k个质心
    centers = np.array(random_centers)
    near_cen = [0]
    for i in range(100):  # 最多做100次迭代
        old_class = near_cen
        # step 2: 点归属
        near_cen, dist = near_center(data, centers)
        # step 3：簇重心更新
        for ci in range(k):  # 每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均
            centers[ci] = data[near_cen == ci].mean()
        # 比较样本点所属类别是否发生改变，near_cen即new_class
        diff = old_class - near_cen
        # print("old_class:" + str(old_class))
        # print("new_class:" + str(near_cen))
        # print("diff:"+str(diff))
        # 若样本点类别不在发生改变，退出循环
        if diff.sum() == 0:
            break
    return centers, near_cen, dist


if __name__ == '__main__':
    path = 'testdata.txt'
    data = read_data(path)
    centers, near_cen, dist = kmeans(data, 4)
    print(f"centers={centers}, \nnear_cen={near_cen}, \ndist={dist}\n")
    # random_change_class(centers, near_cen, dist)
    plt.figure()
    x = data['x']
    y = data['y']
    plt.scatter(x, y, c=near_cen)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=500, c='r')
    plt.show()
