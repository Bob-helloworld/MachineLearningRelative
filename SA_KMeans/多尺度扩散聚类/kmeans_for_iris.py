import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 加载数据集，是一个字典类似Java中的map
from KmeansTest.kmeans import kmeans
from 多尺度扩散聚类.MultiscaleDiffusionClusteringTest import multiscale_diffusion_clustering, target_function

lris_df = datasets.load_iris()

# 挑选出前两个维度作为x轴和y轴，你也可以选择其他维度
x_axis = lris_df.data[:, 0]
y_axis = lris_df.data[:, 2]

# 将数组合并为DataFrame
data = pd.DataFrame({'x': x_axis, 'y': y_axis})

print(data)

# 运动点个数
k = 3
# 尺度下降系数
alpha = 0.5
# 最小尺度
d_min = 0.01
# 每个尺度循环次数
run_times = 500
f = open('best.txt', 'a+')
print(f"参数配置：k={k}, alpha={alpha}, d_min={d_min}, run_times={run_times}", file=f)
centers, near_cen, dist = multiscale_diffusion_clustering(data, k, alpha, d_min, run_times)
target_func = target_function(near_cen, dist)
print(f"F={target_func}, \ncenters={centers}, \nnear_cen={near_cen}, \ndist={dist}\n", file=f)
f.close()
plt.figure(1)
x = data['x']
y = data['y']
plt.scatter(x, y, c=near_cen)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=500, c='r')
plt.savefig('figs/multiscale_diffusion_clustering.png')


# centers, near_cen, dist = kmeans(data, 3)
# print(f"centers={centers}, \nnear_cen={near_cen}, \ndist={dist}\n")
# # random_change_class(centers, near_cen, dist)
# plt.figure(2)
# x = data['x']
# y = data['y']
# plt.scatter(x, y, c=near_cen)
# plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=500, c='r')
# plt.savefig('figs/kmeans.png')

# 这里已经知道了分3类，其他分类这里的参数需要调试
model = KMeans(n_clusters=3)


# 训练模型
model.fit(lris_df.data)

# 选取行标为100的那条数据，进行预测
prddicted_label = model.predict([[6.3, 3.3, 6, 2.5]])

# 预测全部150条数据
all_predictions = model.predict(lris_df.data)
plt.figure(3)
# 打印出来对150条数据的聚类散点图
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()