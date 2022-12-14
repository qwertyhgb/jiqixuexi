import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# from sklearn import datasets
from sklearn.datasets import load_iris

# 引入鸢尾花的数据
iris = load_iris()
X = iris.data[:]
# 打印数据  
# print(X)
# print(X.shape)

# 将鸢尾花数据可视化
plt.scatter(X[:, 0], X[:, 1], c="blue", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
