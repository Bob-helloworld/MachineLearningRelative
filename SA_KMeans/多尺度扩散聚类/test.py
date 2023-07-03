import numpy as np
import matplotlib.pyplot as plt

def generate_circular_data(num_samples, radius, noise_std):
    angles = np.linspace(0, 2 * np.pi, num_samples)
    x = radius * np.cos(angles) + np.random.normal(0, noise_std, num_samples)
    y = radius * np.sin(angles) + np.random.normal(0, noise_std, num_samples)
    return x, y
num_samples = 200
num_classes = 3
radius = 5.0
noise_std = 0.2

data = []
labels = []

for class_id in range(num_classes):
    x, y = generate_circular_data(num_samples, radius * (class_id + 1), noise_std)
    data.append(np.column_stack((x, y)))
    labels.append(np.full(num_samples, class_id))

data = np.concatenate(data)
labels = np.concatenate(labels)

def generate_cluster_data(num_samples, centers, cluster_std):
    data = []
    labels = []

    for class_id, center in enumerate(centers):
        # 生成随机数据点
        x = np.random.normal(center[0], cluster_std, num_samples)
        y = np.random.normal(center[1], cluster_std, num_samples)

        # 添加数据点和标签
        data.append(np.column_stack((x, y)))
        labels.extend([class_id] * num_samples)

    return np.concatenate(data), np.array(labels)

num_samples = 100
centers = [(0, 0), (10, 0), (0, 10), (10, 10)]
cluster_std = 1.0

data, labels = generate_cluster_data(num_samples, centers, cluster_std)


# 保存坐标到文件
np.savetxt('data_points.txt', data, fmt='%.6f')

plt.scatter(data[:, 0], data[:, 1], c=labels, s=5)
plt.axis('equal')
plt.show()
