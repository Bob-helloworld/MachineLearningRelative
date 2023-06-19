import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

##假设k取4
data = pd.read_table('testdata.txt', header=None, names=['x', 'y'])
##没有表头，read_table去Tab
x = data['x']
y = data['y']
# plt.subplot(2,1,1)
plt.scatter(x, y)


def distance(data, centers):
    #     data: 80x2, centers: kx2
    dist = np.zeros((data.shape[0], centers.shape[0]))  ## 出来的是80*4，即每个点相对4个质心的距离
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i, j] = np.sqrt(np.sum((data.iloc[i, :] - centers[j]) ** 2))  ##共80个样本，每个样本的在两个特征维度上，
            # 分别对k个质心求距离然后求和，类似矩阵乘法，
            # 所以距离矩阵为80x4

    return dist


def near_center(data, centers):  ##得到每个点离哪个质心最近，返回对应质心的label
    dist = distance(data, centers)
    print("dist:")
    print(dist[0,1])
    near_cen = np.argmin(dist, 1)  ##得到的dist行为80个点，列为每个点到4个质心的距离。然后取最小距离，得到对应质心的label。
    return near_cen


def kmeans(data, k):
    # step 1: init. centers
    np.random.seed(2022)
    centers = np.random.choice(np.arange(-5, 5, 0.1), (k, 2))  ##随机产生质心

    for _ in range(10):  # 做10次迭代
        # step 2: 点归属
        near_cen = near_center(data, centers)
        # step 3：簇重心更新
        for ci in range(k):  ##每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均
            centers[ci] = data[near_cen == ci].mean()
    return centers, near_cen


# 参考万有引力公式,假设质量都一样，假设两个焦点为两个星球，计算每个样本 到每两个星球对 引力之和（暂时用标量之和）。谁大就属于那个类。

def new_distance(data, centers, focus1, focus2):
    #     data: 80x2, centers: kx2
    dist1 = np.zeros((data.shape[0], centers.shape[0]))  ## 出来的是80*4，即每个点相对4个质心的距离
    dist2 = np.zeros((data.shape[0], centers.shape[0]))

    # 考虑第一个焦点的引力
    for i in range(len(data)):
        for j in range(len(focus1)):
            # dist[i, j] = np.sqrt(np.sum((data.iloc[i, :] - centers[j]) ** 2)) ##共80个样本，每个样本的在两个特征维度上，
            #                                                                 # 分别对k个质心求距离然后求和，类似矩阵乘法，
            #                                                                # 所以距离矩阵为80x4

            # 1/r^2
            dist1[i, j] = 1 / (np.sum((data.iloc[i, :] - focus1[j]) ** 2) + 0.1 ** 6)

    # # 考虑第二个焦点的引力
    for i in range(len(data)):
        for j in range(len(focus2)):
            # 1/r^2
            dist2[i, j] = 1 / (np.sum((data.iloc[i, :] - focus2[j]) ** 2) + 0.1 ** 6)

    return dist1 + dist2


def new_near_center(data, centers, focus1, focus2):  ##得到每个点离哪个质心最近，返回对应质心的label

    dist = new_distance(data, centers, focus1, focus2)

    near_cen = np.argmax(dist, 1)  # 引力最大
    return near_cen


# 计算新的质心以及椭圆焦点
def adjust_center(data, near_cen, k):
    # fig = plt.figure()
    centers = np.random.choice(np.arange(-5, 5, 0.1), (k, 2))
    focus1 = np.random.choice(np.arange(-5, 5, 0.1), (k, 2))
    focus2 = np.random.choice(np.arange(-5, 5, 0.1), (k, 2))
    cdata = []
    for i in range(k):
        cdata.append([data[near_cen == i].values])

    for ci in range(k):  ##每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均

        # a = np.array([[1, -2.7], [5, 5], [2, 3], [1, 4]]).astype(np.float32)
        # 计算并返回指定点集的最小区域边界斜矩形。
        rect = cv2.minAreaRect(cdata[ci][0].astype(np.float32))
        # 中心坐标
        centers[ci] = rect[0]
        height = rect[1][0]
        width = rect[1][1]
        theta = rect[2]

        c = np.sqrt(np.abs(width ** 2 - height ** 2) / 4)

        box = cv2.boxPoints(rect)

        # box[0] 为 最左的点
        if np.linalg.norm(box[0] - box[1]) >= np.linalg.norm(box[0] - box[3]):

            cpoint1 = (box[0] + box[3]) / 2
            cpoint2 = (box[1] + box[2]) / 2
            # 长边
            a = 0
            if height >= width:
                a = height
            else:
                a = width

        else:
            cpoint1 = (box[0] + box[1]) / 2
            cpoint2 = (box[2] + box[3]) / 2
            # 长边
            a = 0
            if height >= width:
                a = height
            else:
                a = width
        focus1[ci] = c / a * (cpoint1 - rect[0]) + rect[0]
        focus2[ci] = c / a * (cpoint2 - rect[0]) + rect[0]

    #     plt.scatter(box[0][0], box[0][1], marker='s', s=100, c='r')
    #     plt.scatter(box[1][0], box[1][1], marker='s', s=100, c='g')
    #     plt.scatter(box[2][0], box[2][1], marker='s', s=100, c='b')
    #     plt.scatter(box[3][0], box[3][1], marker='s', s=100, c='y')
    #     # 最下的位置开始
    #     prect = plt.Rectangle(box[1], height, width,  theta, fill=False, edgecolor='red', linewidth=1)
    #     plt.gca().add_patch(prect)
    #
    #
    # x = data['x']
    # y = data['y']
    # plt.scatter(x, y, c=near_cen)
    # plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=500, c='r')
    # plt.scatter(focus1[:, 0], focus1[:, 1], marker='x', s=500, c='g')
    # plt.scatter(focus2[:, 0], focus2[:, 1], marker='x', s=500, c='b')

    # # 簇重心更新
    # for ci in range(k):  ##每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均
    #     centers[ci] = data[near_cen == ci].mean()
    return centers, focus1, focus2


def new_kmeans(data, k):
    # step 1: init. centers
    np.random.seed(2022)
    centers = np.random.choice(np.arange(-5, 5, 0.1), (k, 2))  ##随机产生质心

    # 第一次迭代使用kmeans的方式
    # step 2: 点归属
    near_cen = near_center(data, centers)

    centers, focus1, focus2 = adjust_center(data, near_cen, k)

    # near_cen = new_near_center(data, near_cen, centers)

    # 从第二次迭代开始用新的迭代方式。即用最小包含该类所有点的椭圆质心为新质心（该版本用最小外接矩阵近似，以该矩阵的椭圆为所求椭圆）

    for _ in range(10 - 1):  # 做10次迭代

        # 聚类后的数据
        cdata = []
        for i in range(k):
            cdata.append([data[near_cen == i]])

        # step 2: 点归属
        near_cen = new_near_center(data, centers, focus1, focus2)
        # step 3：簇重心更新
        centers, focus1, focus2 = adjust_center(data, near_cen, k)
        # for ci in range(k): ##每次点划分完之后，安照步骤，需要重新寻找各个簇的质心，即求平均
        #     centers[ci] = data[near_cen == ci].mean()

    return centers, near_cen, focus1, focus2


plt.figure()
centers, near_cen = kmeans(data, 4)
# plt.subplot(2,1,2)
plt.scatter(x, y, c=near_cen)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=500, c='r')
print(centers)
print(near_cen)

centers, near_cen, focus1, focus2 = new_kmeans(data, 4)
plt.figure()
# plt.subplot(2,1,2)
plt.scatter(x, y, c=near_cen)
plt.scatter(centers[:, 0], centers[:, 1], marker='2', s=500, c='r')
plt.scatter(focus1[:, 0], focus1[:, 1], marker='2', s=500, c='g')
plt.scatter(focus2[:, 0], focus2[:, 1], marker='2', s=500, c='b')
plt.show()
