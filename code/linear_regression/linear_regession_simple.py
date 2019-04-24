#!/usr/bin/env python

"""
    scikit-learn
    一元线性回归
    举例：
    经典房价问题
    x 房子平方英尺
    y 价格
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = [[150], [200], [250], [300], [350], [400], [600]]
y = [6450,7450,8450,9540,11450,12450,17450]

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

plt.title("simple linear regression")
plt.scatter(x, y, color='black')
plt.plot(x, model.predict(x), color='blue')
plt.xlabel('x')
plt.ylabel('price')
plt.show()

