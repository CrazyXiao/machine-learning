#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
    描述性统计
    描述性统计是容易操作，直观简洁的数据分析手段。
    但是由于简单，对多元变量的关系难以描述。
    现实生活中，自变量通常是多元的：决定体重不仅有身高，还有饮食习惯，肥胖基因等等因素。
    通过一些高级的数据处理手段，我们可以对多元变量进行处理，
    例如特征工程中，可以使用互信息方法来选择多个对因变量有较强相关性的自变量作为特征，
    还可以使用主成分分析法来消除一些冗余的自变量来降低运算复杂度。
"""
from numpy import array, cov, corrcoef
from numpy import mean, median, ptp, var, std
from scipy.stats import mode
from numpy.random import normal, randint

data1 = randint(0, 10, size=10)
print(data1)
print('平均值', mean(data1))
print('众数', mode(data1))
print('中位数', median(data1))
print('极差', ptp(data1))
print('方法', var(data1))
print('标准差', std(data1))
print('变异系数',  mean(data1) / std(data1))
# 偏差程度 定义z-分数（Z-Score）为测量值距均值相差的标准差数目
print('偏差程度:', (data1 -mean(data1)) / std(data1))

print('---------------------------------')
data2 = randint(0, 10, size=10)
data3 = randint(0, 10, size=10)
data = array([data1, data2, data3])
print(data)

#计算两组数的协方差
#参数bias=1表示结果需要除以N，否则只计算了分子部分
#返回结果为矩阵，第i行第j列的数据表示第i组数与第j组数的协方差。对角线为方差
print('协方差', cov(data, bias=1))

#计算两组数的相关系数
#返回结果为矩阵，第i行第j列的数据表示第i组数与第j组数的相关系数。对角线为1
print('相关系数', corrcoef(data))


