#!/usr/bin/env python

"""
    scikit-learn
    随机梯度下降
    随机梯度下降法相比批量梯度下降，每次用一个样本调整参数，逐渐逼近，效率高
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# 数据多才可以更好的拟合
x = [[150], [200], [250], [300], [350], [400], [600]] * 100
y = [[6450],[7450],[8450],[9540],[11450],[12450],[17450]] * 100

# 正则化 将特征映射到方差为1 均值为0的标准正太分布中去
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)

# training set
x_train = x[:-2]
y_train = y[:-2]

# test set
x_test = x[-2:]
y_test = y[-2:]

# training
model = SGDRegressor()
model.fit(x_train, y_train.ravel())
print("系数：{} 截距：{} 方差：{}".format(model.coef_, model.intercept_, np.mean((model.predict(x_test)-y_test)**2)))
print("预测结果：{} 得分：{}".format(model.predict(x_test), model.score(x_test, y_test)))

plt.title("simple linear regression")
plt.scatter(x, y, color='black')
plt.plot(x, model.predict(x), color='blue')
plt.xlabel('x')
plt.ylabel('price')
plt.show()