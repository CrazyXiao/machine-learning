#!/usr/bin/env python

"""
    scikit-learn
    多元线性回归

"""

import numpy as np
from numpy.linalg import inv
from numpy import  dot, transpose
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_ceot(x,y):
    """
        根据x,y求解多元线性方程
    """
    return dot(inv(dot(transpose(x), x)), dot(transpose(x),y))

x = 100 * np.random.rand(100,3)
y = dot(x, [1,2,3]) + np.random.randn(100)

# training set
x_train = x[:-2]
y_train = y[:-2]

# test set
x_test = x[-2:]
y_test = y[-2:]

# training
model = LinearRegression()
model.fit(x_train, y_train)
print("系数：{} 截距：{} 方差：{}".format(model.coef_, model.intercept_, np.mean((model.predict(x_test)-y_test)**2)))
print("预测结果：{} 得分：{}".format(model.predict(x_test), model.score(x_test, y_test)))

plt.title("multi linear regression")
plt.scatter(x[:,2], y, color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


