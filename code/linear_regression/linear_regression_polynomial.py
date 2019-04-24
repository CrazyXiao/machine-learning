#!/usr/bin/env python

"""
    多项式回归问题
    可以通过调整degree参数来比较训练后的结果
    并以此判断哪种是最合适的模型
    但是要防止过拟合现象
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 代表使用几次多项式回归模型
degree = 2

# 自己构造数据集
x = np.linspace(5, 30, 20)
y = (5+ x**2) + np.random.randn(20) * 5
x = x[:,np.newaxis]

# training set
x_train = x[:-2]
y_train = y[:-2]

# test set
x_test = x[-2:]
y_test = y[-2:]

# 用多项式对x做变换 degree代表 n次多项式
quadratic_featurizer = PolynomialFeatures(degree=degree)
x_quadratic = quadratic_featurizer.fit_transform(x)
x_train_quadratic = x_quadratic[:-2]
x_test_quadratic = x_quadratic[-2:]
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)

print("系数：{} 误差：{}".format(regressor_quadratic.coef_, regressor_quadratic.intercept_))
print("预测结果：{} 得分：{}".format(regressor_quadratic.predict(x_test_quadratic), regressor_quadratic.score(x_test_quadratic, y_test)))


plt.scatter(x, y, color='black')
plt.plot(x, regressor_quadratic.predict(x_quadratic), 'g-')
plt.show() # 展示图像