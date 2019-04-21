#!/usr/bin/env python

"""
    无量纲处理

"""
import numpy as np


"""
在进行特征选择之前，一般会先进行数据无量纲化处理，这样，表征不同属性（单位不同）的各特征之间才有可比性，
如1cm 与 0.1kg 你怎么比？无量纲处理方法很多，使用不同的方法，对最终的机器学习模型会产生不同的影响。
"""
# 归一化
from sklearn.preprocessing import MinMaxScaler
x = np.array([[1,-1,2],[2,0,0],[0,1,-1]])
x1 = MinMaxScaler().fit_transform(x)
print(x1)

# 标准化
from sklearn.preprocessing import StandardScaler
x = np.array([[1,2,3],[4,5,6],[1,2,1]])
x1 = StandardScaler().fit_transform(x)
print(x1)
scaler = StandardScaler().fit(x)
# print(scaler)
print(scaler.mean_)

# 正则化
from sklearn import preprocessing
normalizer = preprocessing.Normalizer().fit(x)
print(normalizer.transform(x))
