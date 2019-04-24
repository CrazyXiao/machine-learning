#!/usr/bin/env python


"""
    iris数据集
    特征工程

"""

from sklearn.datasets import load_iris

#导入IRIS数据集
iris = load_iris()

#特征矩阵
print(iris.data)

#目标向量
print(iris.target)
print(iris.target.shape)

# from sklearn.preprocessing import StandardScaler
# print(StandardScaler().fit_transform(iris.data))

# from sklearn.preprocessing import Binarizer
#二值化，阈值设置为3，返回值为二值化后的数据
# print(Binarizer(threshold=3).fit_transform(iris.data))

from sklearn.preprocessing import OneHotEncoder
print(OneHotEncoder().fit_transform(iris.target.reshape((-1,1))).toarray())

from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer
#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
print(Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data))))
