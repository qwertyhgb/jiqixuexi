import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 读取数据
samples = np.loadtxt("E:\\MyCode\\机器学习\\data\\kmeansSamples.txt")
clustering = DBSCAN(eps=5, min_samples=5).fit(samples)
print(clustering.labels_)
print(clustering)

# 画图
plt.scatter(samples[:, 0], samples[:, 1], c=clustering.labels_+1.5, linewidths=np.power(clustering.labels_+1.5, 2))
plt.show()
