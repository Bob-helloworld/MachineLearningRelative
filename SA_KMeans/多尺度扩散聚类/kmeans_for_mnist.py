import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 加载手写数字数据集（MNIST）
digits = load_digits()
X = digits.data
y = digits.target

# 使用PCA降维到2维，方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# 将数组合并为DataFrame
# data = pd.DataFrame({'x': x_axis, 'y': y_axis})
print(X)
# 使用K-means算法进行聚类，这里假设聚类数为10（对应0-9的手写数字）
kmeans = KMeans(n_clusters=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)
accuracy = accuracy_score(y, y_kmeans)
print("分类准确度：", accuracy)

# 可视化聚类结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of MNIST Data')
plt.show()
