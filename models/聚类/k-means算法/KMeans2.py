from random import sample
import numpy as np
import matplotlib.pyplot as plt


def L2(vecXi, vecXj):
    """
    计算欧氏距离
    para vecXi vecXj 点坐标
    """
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))


def KMeans(S, K, distMeas=L2):
    """
    para S: 样本集 多维数组
    para K: 簇个数
    para distMeas: 欧氏距离
    return sampleTag: 一维数组, 存储样本对应的簇标记
    return clusterCents: 一维数组,各簇中心
    return SSE: 误差平方和
    """

    # 样本总数
    m = np.shape(S)[0]  # 0表示S的行数
    sampleTag = np.zeros(m)

    # 随即产生K个初始簇中心
    n = np.shape(S)[1]  # 1表示S的列数
    # clusterCents = np.mat([[-1.93964824, 2.33260803], [7.79822795, 6.72621783], [10.64183154, 0.20088133]])
    clusterCents = np.mat(np.zeros((K, n)))
    for j in range(n):
        minJ = min(S[:, j])
        rangeJ = float(max(S[:, j]) - minJ)
        clusterCents[:, j] = np.mat(minJ + rangeJ * np.random.rand(K, 1))

    sampleTagChanged = True  # 点发生改变
    SSE = 0.0  #

    while sampleTagChanged:  # 如果没有点发生分配结果改变,则结束
        sampleTagChanged = False
        SSE = 0.0

        # 计算每个样本点到簇中心的距离
        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(K):
                d = distMeas(clusterCents[j, :], S[i, :])
                if d < minD:
                    minD = d
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex
            SSE += minD ** 2
        print(SSE)

        # 重新计算簇中心
        for i in range(K):
            ClustI = S[np.nonzero(sampleTag[:] == i)[0]]
            clusterCents[i, :] = np.mean(ClustI, axis=0)
    return clusterCents, sampleTag, SSE


if __name__ == '__main__':
    samples = np.loadtxt("E:\\MyCode\\机器学习\\data\\kmeansSamples.txt")
    clusterCents, sampleTag, SSE = KMeans(samples, 3)

    plt.scatter(samples[:, 0], samples[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
    plt.show()
    print(clusterCents)
    print(SSE)
    # print(samples)
