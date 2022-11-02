"""
线性回归
温度与花朵数量关系例子
"""

# 温度
temperatures = [15, 20, 25, 30, 35, 40]
# 花朵
flowers = [136, 140, 155, 160, 157, 175]

import numpy as np


def least_square(X, Y):
    """
    para X:矩阵,样本特征矩阵
    para Y:矩阵,标签向量
    return:矩阵,回归系数
    """
    W = (X * X.T).I * X * Y.I
    return W


X = np.mat([1, 1, 1, 1, 1, 1], temperatures)
print(X)

Y = np.mat(flowers)
print(Y)

W = least_square(X, Y)
print(W)

# 画图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(temperatures, flowers, color="green", label="花朵数量", linewidth=2)
plt.plot(temperatures, flowers, linewidth=1)
x1 = np.linspace(15, 40, 100)
y1 = W[1, 0] * x1 + W[0, 0]
plt.plot(x1, y1, color="red", label="拟合直线", linewidth=2, linestyle=':')
plt.legend(loc='lower right')
plt.show()

new_tempera = [18, 22, 33]
new_tempera = (np.mat(new_tempera)).T
pro_num = W[1, 0] * new_tempera + W[0, 0]
print(pro_num)
