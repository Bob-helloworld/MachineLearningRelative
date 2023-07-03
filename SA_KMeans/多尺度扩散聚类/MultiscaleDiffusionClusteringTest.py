import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def target_function(near_cen, dist):
    target_func = 0
    # 每个点距离自己类别中心点的距离之和
    for i in range(len(dist)):
        target_func = target_func + dist[i, near_cen[i]]
        # print("dest_func is " + str(target_func))
    return target_func


# 使用kmeans聚类
def multiscale_diffusion_clustering(data, k, alpha, d_min, run_times):
    # step 1: init. centers
    seed = int(time.time())
    np.random.seed(seed)
    random_centers = data.iloc[np.random.choice(data.shape[0], size=k, replace=False), :]  # 随机产生k个质心
    centers = np.array(random_centers)
    # step 2: 点归属
    near_cen, dist = near_center(data, centers)
    # step 3：计算系统浓度
    density = target_function(near_cen, dist)
    # 选择数据集样本点之间的最大欧式距离作为初始扩散尺度,
    d = np.max(distance(data, np.array(data)))
    best_centers = centers
    best_near_cen = near_cen
    best_dist = dist
    # print(f"random_centers={random_centers}, \ndist_0={d}")
    while d > d_min:
        for i in range(run_times):
            # step 4：生成高斯随机数
            seed = int(time.time())
            np.random.seed(seed)
            random_number = np.random.normal(0, 1, [len(centers), 2])
            # step 5：簇的扩散
            new_centers = best_centers + d * random_number
            # print(f"random_number={random_number}, \nnew_centers={new_centers} , \nbest_centers={best_centers}")
            # step 6：计算比较浓度的变化
            new_near_cen, new_dist = near_center(data, new_centers)
            new_density = target_function(new_near_cen, new_dist)
            print(f"d={d}, \nnew_density={new_density} , \nbest_centers={best_centers}")
            # 若样本点类别不在发生改变，退出循环
            if new_density < density:
                density = new_density
                best_centers = new_centers
                best_near_cen = new_near_cen
                best_dist = new_dist
        d = alpha * d
    return best_centers, best_near_cen, best_dist


if __name__ == '__main__':
    path = 'testdata.txt'
    data = read_data(path)
    # 运动点个数
    k = 4
    # 尺度下降系数
    alpha = 0.5
    # 最小尺度
    d_min = 0.1
    # 每个尺度循环次数
    run_times = 500
    f = open('best.txt', 'a+')
    print(f"参数配置：k={k}, alpha={alpha}, d_min={d_min}, run_times={run_times}", file=f)
    centers, near_cen, dist = multiscale_diffusion_clustering(data, k, alpha, d_min, run_times)
    target_func = target_function(near_cen, dist)
    print(f"F={target_func}, \ncenters={centers}, \nnear_cen={near_cen}, \ndist={dist}\n", file=f)
    f.close()
    plt.figure()
    x = data['x']
    y = data['y']
    plt.scatter(x, y, c=near_cen)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=500, c='r')
    plt.show()
