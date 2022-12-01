import math
import random

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
    np.random.seed(2022)
    centers = np.random.choice(np.arange(-5, 5, 0.1), (k, 2))  # 随机产生质心
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


def target_function(near_cen, dist):
    target_func = 0
    # 每个点距离自己类别中心点的距离之和
    for i in range(len(dist)):
        target_func = target_func + dist[i, near_cen[i]]
        # print("dest_func is " + str(target_func))
    return target_func


def random_change_class(centers, near_cen, dist):
    # 提取每行的最小值即每个样本点距离每个类中心的最小距离，然后选取k个距离最大的样本点随机更改其类别
    tmp = dist.min(axis=1)
    print("分类后的dist：" + str(tmp.sum()))
    k = len(centers)
    # k个最大数值的索引
    change_sample = np.argpartition(tmp, -k)[-k:]
    print("change_sample is " + str(change_sample))
    for i in range(len(change_sample)):
        # 将k个距离最大的样本点随机更改其类别
        new_class = random.randint(0, k - 1)
        print(str(change_sample[i]) + " sample is changed,old_class is " + str(
            near_cen[change_sample[i]]) + " new_class is " + str(new_class))
        near_cen[change_sample[i]] = new_class
    return near_cen


class SA:
    def __init__(self, func, data, centers, near_cen, dist, iter=100, T0=100, Tf=0.01, alpha=0.5):
        self.func = func  # 目标函数
        self.data = data  # 样本点数据
        self.centers = centers  # 类中心点数组
        self.near_cen = near_cen  # 各样本点类别
        self.dist = dist  # 各样本点距各类别中心点距离二维数组
        self.iter = len(data)  # 内循环迭代次数,即为L =100
        self.alpha = alpha  # 降温系数，alpha=0.99
        self.T0 = self.func(near_cen, dist)  # 初始温度为初始目标函数值
        self.Tf = Tf  # 温度终值Tf为0.01
        self.T = self.T0  # 当前温度
        self.most_best = []
        self.x = {}  #
        self.y = {}
        self.z = {}
        self.best1 = {}  #
        self.best2 = {}
        self.best3 = {}
        """
        random()这个函数取0到1之间的小数
        如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
        该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
        （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
        """
        self.history = {'f': [], 'T': []}

    def generate_new(self, centers, near_cen, dist):
        # 提取每行的最小值即每个样本点距离每个类中心的最小距离，然后选取k个距离最大的样本点随机更改其类别
        tmp = dist.min(axis=1)
        # print("分类后的target：" + str(tmp.sum()))
        k = len(centers)
        # k个最大数值的索引
        # change_sample = np.argpartition(tmp, -k)[-k:]
        # 只改变一个最大距离样本点的类别
        change_sample = np.argmax(tmp)
        # print("change_sample is " + str(change_sample))
        for _ in range(1):
            # 将k个距离最大的样本点随机更改其类别
            new_class = random.randint(0, k - 1)
            # print(str(change_sample[i]) + " sample is changed,old_class is " + str(
            #     near_cen[change_sample[i]]) + " new_class is " + str(new_class))
            near_cen[change_sample] = new_class
        # 随机改变k个样本点的类别后重新计算中心点
        for ci in range(len(centers)):  # 每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均
            centers[ci] = self.data[near_cen == ci].mean()
        # 更改中心点后重新计算每个样本点与中心点的距离
        for i in range(len(self.data)):
            for j in range(len(centers)):
                dist[i, j] = np.sqrt(np.sum((self.data.iloc[i, :] - centers[j]) ** 2))  # 共80个样本，每个样本的在两个特征维度上，
                # 分别对k个质心求距离然后求和，类似矩阵乘法，
                # 所以距离矩阵为80x4
        return centers, near_cen, dist

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random.random() < p:
                return 1
            else:
                return 0

    def best(self, x, y, z):  # 获取最优目标函数值
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(len(x)):
            f = self.func(x[i], y[i])
            f_list.append(f)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        self.history['f'].append(self.func(self.near_cen, self.dist))
        self.history['T'].append(self.T)
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:
            # 内循环迭代
            for i in range(self.iter):
                f = self.func(self.near_cen, self.dist)  # f为迭代一次后的值
                new_centers, new_near_cen, new_dist = self.generate_new(self.centers, self.near_cen,
                                                                        self.dist)  # 产生新解
                f_new = self.func(new_near_cen, new_dist)  # 产生新值
                if self.Metrospolis(f, f_new):  # 判断是否接受新值
                    # 如果接受新值，则更新新的类别信息
                    self.centers = new_centers
                    self.near_cen = new_near_cen
                    self.dist = new_dist
                    # 记录当前解信息, .copy()防止后面追加的列表覆盖前面的列表
                    self.x[i] = new_near_cen.copy()
                    self.y[i] = new_dist.copy()
                    self.z[i] = new_centers.copy()
                else:
                    self.x[i] = self.near_cen.copy()
                    self.y[i] = self.dist.copy()
                    self.z[i] = self.centers.copy()
            # 迭代L次记录在该温度下最优解
            ft, ft_idx = self.best(self.x, self.y, self.z)
            print(f"第{count}次最佳F={ft}, \nnear_cen={self.x[ft_idx]}, "
                  f"\ndist={self.y[ft_idx]}, \ncenters={self.z[ft_idx]}\n")
            self.best1[count] = self.x[ft_idx]
            self.best2[count] = self.y[ft_idx]
            self.best3[count] = self.z[ft_idx]
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            count += 1
        # 得到最优解
        f_best, idx = self.best(self.best1, self.best2, self.best3)
        print(f"最终结果F={f_best}, \nnear_cen={self.best1[idx]}, \ndist={self.best2[idx]}, \ncenters={self.best3[idx]}\n")
        # for _ in range(10):  # 做10次迭代
        #     # step 2: 点归属
        #     near_cen, dist = near_center(data, centers)
        #     # step 3：簇重心更新
        #     for ci in range(len(centers)):  ##每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均
        #         centers[ci] = data[near_cen == ci].mean()
        # target_func = self.func(near_cen, dist)
        # print(f"最终结果:F={target_func}\ncenters={centers}, \nnear_cen={near_cen}, \ndist={dist}\n")
        ##没有表头，read_table去Tab
        x = data['x']
        y = data['y']
        # plt.subplot(2,1,1)
        plt.scatter(x, y)
        plt.figure()
        plt.scatter(x, y, c=near_cen)
        plt.scatter(self.best3[idx][:, 0], self.best3[idx][:, 1], marker='x', s=500, c='r')


if __name__ == '__main__':
    path = 'testdata.txt'
    data = read_data(path)
    # step1 先进行一次kmeans聚类获得初始解并获得目标函数值（每个点距离自己类别中心点的距离之和）
    centers, near_cen, dist = kmeans(data, 4)
    target_func = target_function(near_cen, dist)
    print(f"F={target_func}, \ncenters={centers}, \nnear_cen={near_cen}, \ndist={dist}\n")
    # random_change_class(centers, near_cen, dist)
    sa = SA(target_function, data, centers, near_cen, dist)
    sa.run()
    plt.figure()
    plt.plot(sa.history['T'], sa.history['f'])
    plt.title('SA')
    plt.xlabel('T')
    plt.ylabel('f')
    plt.gca().invert_xaxis()
    plt.show()
