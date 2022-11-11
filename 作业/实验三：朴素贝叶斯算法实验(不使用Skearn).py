# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : LiMinG
# @Email : 2168884970@qq.com
# -------------------------------
# @File : 实验三：朴素贝叶斯算法实验(不使用Sklearn).py
# @Software : PyCharm
# @Time : 2022/11/11 20:03
# -------------------------------

import numpy as np
import pandas as pd


def Y_prob(Y):  # 好瓜和坏瓜的各自的比重
    y = Y.values
    true_prob = sum(y) / len(y)
    return {0: 1 - true_prob, 1: true_prob}


def x_y_prob(feature, y):  # 求p(特征|类别)，用字典的形式存储返回
    x_y = {}
    n = len(y)
    for f in feature.columns:
        # for i in range(feature_num(feature[f])): #x_f = i
        for i in feature[f].value_counts().keys():
            index = (y == 1)
            index_True = y[y == True].index
            index_False = y[y == False].index

            sample_True = data.loc[index_True, f] == i
            prob_true = len(sample_True.loc[sample_True == True]) / len(index_True)
            strings_True = str(f) + '=' + str(i) + '|' + 'y=1'

            sample_False = data.loc[index_False, f] == i
            prob_False = len(sample_False.loc[sample_False == True]) / len(index_False)
            strings_False = str(f) + '=' + str(i) + '|' + 'y=0'

            x_y[strings_True] = prob_true
            x_y[strings_False] = prob_False
    return x_y


if __name__ == '__main__':
    data = pd.read_csv('E:\MyCode\机器学习\data\data_word.csv')
    feature = data.columns.values[:-1]
    x_y = x_y_prob(data[feature], data['好瓜'])  # 用字典存储p(特征|类别)
    y_prob = Y_prob(data['好瓜'])
    # 要预测的数据
    test_data = ['青绿', '蜷缩', '沉闷', '稍糊', '凹陷', '硬滑']
    # 将测试数据转化为字典可以搜索的格式
    test_true = [str(feature[i]) + '=' + str(test_data[i]) + '|' + 'y=1' for i in range(len(feature))]
    test_false = [str(feature[i]) + '=' + str(test_data[i]) + '|' + 'y=0' for i in range(len(feature))]
    p_true = y_prob[1]
    p_false = y_prob[0]
    for i in range(len(feature)):  # 计算每一类的概率
        p_true *= x_y[test_true[i]]
        p_false *= x_y[test_false[i]]
    print('特征为：{}的瓜，是好瓜的概率为：{:.8f}，不是好瓜的概率为：{:.8f}'.format(test_data, p_true, p_false))
    print('朴素贝叶斯最后预测：' + '是好瓜' if p_true > p_false else '不是好瓜')
